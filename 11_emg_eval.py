import os
import csv
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
"""
PLOTS Median BIS across EMG
As well as .05 to .95 Quantile Band
For EMG_irregular, EMG_regular and all Regions
For BIS (measured) and BIS predicted (XGB)



"""

TARGET_COL = "BIS/BIS"
EMG_COL = "BIS/EMG"
DATA_DIR = "data"
SUFFIX = "_rawdata.parquet"
CLINICAL_INFO = "__clinical_info.csv"

MODEL_DIR = "model"

XGB_MODEL_PATH = os.path.join(MODEL_DIR, "bis_xgb_model.json")
XGB_META_PATH = os.path.join(MODEL_DIR, "bis_xgb_meta.json")

OUT_DIR = "plots_emg_vs_bis"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 12
})


# Model loading
def load_xgb_bundle():
    with open(XGB_META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    model = xgb.XGBRegressor()
    model.load_model(XGB_MODEL_PATH)

    features = meta["features"]
    return model, features, meta


# Clinical / split info
def collect_case_info(dataset):
    case_info = {}
    file_path = os.path.join(DATA_DIR, CLINICAL_INFO)

    with open(file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row.get("Dataset") == dataset:
                cid = row.get("caseid")
                ane_time = row.get("ane_intro_end")

                if cid and ane_time:
                    case_info[cid] = {
                        "ane_intro_end": float(ane_time),
                        "age": row.get("age"),
                        "sex": row.get("sex"),
                        "lbm": row.get("lbm"),
                    }

    return case_info


def create_mask(df, ane_intro_end, label_filter=None):
    mask = df["Time"] > ane_intro_end
    if label_filter is not None:
        mask &= (df["label"] == label_filter)
    return mask


def add_static_features(df, meta):
    for k in ["age", "sex", "lbm"]:
        df[k] = meta.get(k)
    return df


def coerce_static_types(df):
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
    if "lbm" in df.columns:
        df["lbm"] = pd.to_numeric(df["lbm"], errors="coerce")
    if "sex" in df.columns:
        df["sex"] = df["sex"].astype(str).str.lower().map({"m": 1, "f": 0})
    return df


# Data loading
def load_testing_rows(case_info, label_filter=None):
    """
    Load testing rows after ane_intro_end, optionally filtered by label.
    Keeps EMG, target, label, time, statics, and all columns present in raw data.
    """
    rows = []

    for cid, meta in case_info.items():
        path = os.path.join(DATA_DIR, cid + SUFFIX)
        if not os.path.exists(path):
            continue

        df = pd.read_parquet(path)
        df = df.loc[create_mask(df, meta["ane_intro_end"], label_filter)].copy()

        if len(df) == 0:
            continue

        df = add_static_features(df, meta)
        df = coerce_static_types(df)
        df["caseid"] = cid

        rows.append(df)

    if not rows:
        raise RuntimeError("No testing rows loaded. Check paths, split, and mask.")

    return pd.concat(rows, axis=0, ignore_index=True)


def build_numeric_feature_matrix(df, features):
    """
    Extract features in exact order and coerce all to numeric.
    """
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns required by model: {missing}")

    X_df = df.loc[:, features].copy()
    for c in X_df.columns:
        X_df[c] = pd.to_numeric(X_df[c], errors="coerce")

    return X_df


def make_prediction_frame(df, xgb_model, xgb_features):
    """
    Build a frame with EMG, observed BIS, XGB pred.
    Keeps rows where all required quantities are finite.
    """
    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column missing: {TARGET_COL}")
    if EMG_COL not in df.columns:
        raise KeyError(f"EMG column missing: {EMG_COL}")

    out = df[["caseid", "Time", "label", EMG_COL, TARGET_COL]].copy()
    out[EMG_COL] = pd.to_numeric(out[EMG_COL], errors="coerce")
    out[TARGET_COL] = pd.to_numeric(out[TARGET_COL], errors="coerce")

    X_xgb_df = build_numeric_feature_matrix(df, xgb_features)
    X_xgb = X_xgb_df.to_numpy(dtype=np.float32)

    xgb_valid = np.isfinite(out[TARGET_COL].to_numpy(dtype=np.float32))
    xgb_valid &= np.isfinite(out[EMG_COL].to_numpy(dtype=np.float32))
    xgb_valid &= np.all(np.isfinite(X_xgb), axis=1)

    xgb_pred = np.full(len(df), np.nan, dtype=np.float32)
    if xgb_valid.any():
        xgb_pred[xgb_valid] = xgb_model.predict(X_xgb[xgb_valid]).astype(np.float32)

    out["xgb_pred"] = xgb_pred

    keep = np.isfinite(out[EMG_COL].to_numpy(dtype=np.float32))
    keep &= np.isfinite(out[TARGET_COL].to_numpy(dtype=np.float32))
    keep &= np.isfinite(out["xgb_pred"].to_numpy(dtype=np.float32))

    out = out.loc[keep].copy()

    if len(out) == 0:
        raise RuntimeError("No aligned finite rows available for EMG/BIS comparison.")

    return out


# EMG binning
def choose_emg_edges(emg_values, max_bins=35, min_count_hint=1000):
    """
    Robust, data-driven EMG bins:
    - trim to 1st..99th percentile for edge construction
    - choose at most max_bins bins
    - fallback to linear spacing if needed
    """
    emg = pd.to_numeric(pd.Series(emg_values), errors="coerce").to_numpy(dtype=float)
    emg = emg[np.isfinite(emg)]

    if len(emg) == 0:
        raise RuntimeError("No finite EMG values found.")

    lo = np.percentile(emg, 1)
    hi = np.percentile(emg, 99)

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(emg))
        hi = float(np.nanmax(emg))

    if hi <= lo:
        hi = lo + 1.0

    n = len(emg)
    approx_bins = max(12, min(max_bins, int(round(n / max(min_count_hint, 1)))))
    approx_bins = min(max_bins, max(12, approx_bins))

    edges = np.linspace(lo, hi, approx_bins + 1)

    edges[0] = min(edges[0], float(np.nanmin(emg)))
    edges[-1] = max(edges[-1], float(np.nanmax(emg)))

    edges = np.unique(edges)
    if len(edges) < 3:
        mn = float(np.nanmin(emg))
        mx = float(np.nanmax(emg))
        if mx <= mn:
            mx = mn + 1.0
        edges = np.linspace(mn, mx, 13)

    return edges


def summarize_by_emg_bin(df, emg_edges, min_rows_per_bin=50):
    """
    Per EMG bin:
      - center
      - count
      - mean of true BIS
      - 5th/95th quantiles of true BIS
      - mean of XGB pred
      - 5th/95th quantiles of XGB pred
    """
    work = df.copy()
    work["emg_bin"] = pd.cut(
        work[EMG_COL],
        bins=emg_edges,
        include_lowest=True,
        right=False,
        duplicates="drop"
    )

    grp = work.groupby("emg_bin", observed=False)

    summary = grp.agg(
        n=(TARGET_COL, "size"),
        emg_mean=(EMG_COL, "mean"),

        bis_mean=(TARGET_COL, "mean"),
        bis_q05=(TARGET_COL, lambda s: s.quantile(0.05)),
        bis_q95=(TARGET_COL, lambda s: s.quantile(0.95)),

        xgb_mean=("xgb_pred", "mean"),
        xgb_q05=("xgb_pred", lambda s: s.quantile(0.05)),
        xgb_q95=("xgb_pred", lambda s: s.quantile(0.95)),
    ).reset_index()

    summary = summary.loc[summary["n"] >= min_rows_per_bin].copy()
    summary = summary.sort_values("emg_mean").reset_index(drop=True)

    for c in ["bis_q05", "bis_q95", "xgb_q05", "xgb_q95"]:
        summary[c] = pd.to_numeric(summary[c], errors="coerce")

    if len(summary) == 0:
        raise RuntimeError("No EMG bins survived min_rows_per_bin threshold.")

    return summary


def attach_bin_quantile_lines(pred_df, emg_edges, summary_df):
    """
    Attach the bin-level quantile lines back to each sample.
    Each sample gets the bis_q95 and xgb_q95 of its EMG bin.
    """
    out = pred_df.copy()
    out["emg_bin"] = pd.cut(
        out[EMG_COL],
        bins=emg_edges,
        include_lowest=True,
        right=False,
        duplicates="drop"
    )

    bin_quantiles = summary_df[["emg_bin", "bis_q95", "xgb_q95"]].copy()
    out = out.merge(bin_quantiles, on="emg_bin", how="left")

    return out


def print_roi_sample_count(pred_df, emg_edges, summary_df, split_name, emg_threshold=35.0):
    """
    ROI definition:
      - EMG > emg_threshold
      - measured BIS > binwise xgb_q95 line
      - xgb_pred < binwise bis_q95 line
    Prints the number of samples in that area.
    """
    work = attach_bin_quantile_lines(pred_df, emg_edges, summary_df)

    roi_mask = (
        (work[EMG_COL] > emg_threshold) &
        np.isfinite(pd.to_numeric(work["xgb_q95"], errors="coerce")) &
        np.isfinite(pd.to_numeric(work["bis_q95"], errors="coerce")) &
        (work[TARGET_COL] > work["xgb_q95"]) &
        (work["xgb_pred"] < work["bis_q95"])
    )

    n_roi = int(roi_mask.sum())
    n_total_emg = int((work[EMG_COL] > emg_threshold).sum())

    print(f"\n[ROI] {split_name}")
    print(f"  Condition: {EMG_COL} > {emg_threshold}")
    print(f"             {TARGET_COL} > binwise XGB 95th quantile")
    print(f"             xgb_pred < binwise BIS 95th quantile")
    print(f"  Samples in ROI         : {n_roi}")
    print(f"  Samples with EMG > {emg_threshold}: {n_total_emg}")

    return work.loc[roi_mask].copy()


# Plotting
def replace_second_last_tick_with_unit(ax, axis, unit):
    ax.tick_params(axis='both', which='both', length=0)

    if axis == "y":
        ticks = ax.get_yticks()
        labels = [f"{t:g}" for t in ticks]
        if len(labels) >= 2:
            labels[-2] = f"{unit}"
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)

    elif axis == "x":
        ticks = ax.get_xticks()
        labels = [f"{t:g}" for t in ticks]
        if len(labels) >= 2:
            labels[-2] = f"{unit}"
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)


def plot_emg_curve(summary_df, out_path):
    x = summary_df["emg_mean"].to_numpy(dtype=float)

    y_true = summary_df["bis_mean"].to_numpy(dtype=float)
    q_true_lo = summary_df["bis_q05"].to_numpy(dtype=float)
    q_true_hi = summary_df["bis_q95"].to_numpy(dtype=float)

    y_xgb = summary_df["xgb_mean"].to_numpy(dtype=float)
    q_xgb_lo = summary_df["xgb_q05"].to_numpy(dtype=float)
    q_xgb_hi = summary_df["xgb_q95"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 4.2))

    ax.plot(x, y_true, color="black", linewidth=2.2, label="Messdaten")
    ax.fill_between(x, q_true_lo, q_true_hi, color="black", alpha=0.10)

    ax.plot(x, y_xgb, color="#E15759", linewidth=2.0, label="XGBoost Index")
    ax.fill_between(x, q_xgb_lo, q_xgb_hi, color="#E15759", alpha=0.14)

    ax.set_xlabel("EMG Leistung ($EMG$)")
    ax.set_ylabel("Bispektralindex ($BIS$)")
    ax.set_ylim(0, 100)
    ax.set_xlim(20, 45)

    ax.grid(True, linewidth=1, color="black", alpha=0.25)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    replace_second_last_tick_with_unit(ax, axis="y", unit="--")
    replace_second_last_tick_with_unit(ax, axis="x", unit="dB")

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()


# Reporting
def print_summary_header(name, raw_rows, aligned_rows, n_bins):
    print(f"\n{name}")
    print(f"  Raw rows after mask      : {raw_rows}")
    print(f"  Rows used in comparison  : {aligned_rows}")
    print(f"  EMG bins kept            : {n_bins}")


def print_bin_preview(summary_df, max_rows=10):
    print("\n  First bins:")
    cols = ["emg_mean", "n", "bis_mean", "bis_q05", "bis_q95", "xgb_mean", "xgb_q05", "xgb_q95"]
    preview = summary_df.loc[:, cols].head(max_rows).copy()
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(preview.to_string(index=False, justify="right", float_format=lambda x: f"{x:8.3f}"))


# one split
def process_split(
    split_name,
    case_info,
    xgb_model,
    xgb_features,
    label_filter=None,
    shared_emg_edges=None,
):
    df = load_testing_rows(case_info, label_filter=label_filter)
    raw_rows = len(df)

    pred_df = make_prediction_frame(df, xgb_model, xgb_features)

    if shared_emg_edges is None:
        emg_edges = choose_emg_edges(pred_df[EMG_COL].to_numpy())
    else:
        emg_edges = shared_emg_edges

    summary = summarize_by_emg_bin(pred_df, emg_edges, min_rows_per_bin=50)

    print_summary_header(split_name, raw_rows, len(pred_df), len(summary))
    print_bin_preview(summary)

    return pred_df, summary

def print_roi_sample_table(roi_df, max_rows=None):
	"""
	Print ROI samples with caseid, timestamp, BIS, and EMG.
	"""
	if roi_df is None or len(roi_df) == 0:
		print("\n[ROI TABLE] No samples in ROI.")
		return

	table = roi_df.loc[:, ["caseid", "Time", TARGET_COL, EMG_COL, "xgb_pred"]].rename(columns={"Time": "timestamp", TARGET_COL: "BIS", EMG_COL: "EMG", "xgb_pred": "BIS_pred"})
	table["timestamp"] = pd.to_numeric(table["timestamp"], errors="coerce")
	table["BIS"] = pd.to_numeric(table["BIS"], errors="coerce")
	table["EMG"] = pd.to_numeric(table["EMG"], errors="coerce")

	table = table.sort_values(["caseid", "timestamp"]).reset_index(drop=True)

	if max_rows is not None:
		table = table.head(max_rows)

	print(f"\n[ROI TABLE] Samples listed: {len(table)}")
	with pd.option_context("display.width", 200, "display.max_columns", None, "display.max_rows", None):
		print(table.to_string(index=False, justify="right", float_format=lambda x: f"{x:8.3f}"))

if __name__ == "__main__":
    print("[INFO] Loading XGBoost model...")
    xgb_model, xgb_features, xgb_meta = load_xgb_bundle()

    print(f"[INFO] XGB features: {len(xgb_features)}")
    print(f"[INFO] EMG column   : {EMG_COL}")

    print("[INFO] Loading testing case information...")
    test_case_info = collect_case_info("Testing")

    print("[INFO] Building shared EMG buckets from all testing data...")
    df_all_for_edges = load_testing_rows(test_case_info, label_filter=None)
    pred_all_for_edges = make_prediction_frame(
        df_all_for_edges,
        xgb_model,
        xgb_features
    )
    shared_edges = choose_emg_edges(
        pred_all_for_edges[EMG_COL].to_numpy(),
        max_bins=35,
        min_count_hint=1200
    )

    # ALL
    pred_all, summary_all = process_split(
        "ALL TESTING (after ane_intro)",
        test_case_info,
        xgb_model,
        xgb_features,
        label_filter=None,
        shared_emg_edges=shared_edges,
    )
    plot_emg_curve(
        summary_all,
        out_path=os.path.join(OUT_DIR, "emg_vs_bis_all_testing.png"),
    )

    # REGULAR
    pred_regular, summary_regular = process_split(
        "REGULAR TESTING (after ane_intro)",
        test_case_info,
        xgb_model,
        xgb_features,
        label_filter="regular",
        shared_emg_edges=shared_edges,
    )
    plot_emg_curve(
        summary_regular,
        out_path=os.path.join(OUT_DIR, "emg_vs_bis_regular_testing.png"),
    )

    # IRREGULAR
    pred_irregular, summary_irregular = process_split(
        "IRREGULAR TESTING (after ane_intro)",
        test_case_info,
        xgb_model,
        xgb_features,
        label_filter="irregular",
        shared_emg_edges=shared_edges,
    )
    plot_emg_curve(
        summary_irregular,
        out_path=os.path.join(OUT_DIR, "emg_vs_bis_irregular_testing.png"),
    )

    # ROI count in irregular data
    roi_irregular = print_roi_sample_count(
        pred_irregular,
        emg_edges=shared_edges,
        summary_df=summary_irregular,
        split_name="IRREGULAR TESTING (after ane_intro)",
        emg_threshold=35.0,
    )
    print_roi_sample_table(roi_irregular)

    print(f"\n[INFO] Finished. Plots saved to: {OUT_DIR}")
