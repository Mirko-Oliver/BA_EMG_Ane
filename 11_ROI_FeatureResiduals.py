import os
import csv
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import ks_2samp, wasserstein_distance

"""
Define ROI, 
in ROI Create for each Feature a very simple XGBoost Model that predicts BIS -> Feature
Calculate residual for each DP
Calculate Standardized Median resiudal for each Feature (between ROI and Non Roi) 
All EMG > 35dB
"""
TARGET_COL = "BIS/BIS"
EMG_COL = "BIS/EMG"
DATA_DIR = "data"
SUFFIX = "_rawdata.parquet"
CLINICAL_INFO = "__clinical_info.csv"

MODEL_DIR = "model"
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "bis_xgb_model.json")
XGB_META_PATH = os.path.join(MODEL_DIR, "bis_xgb_meta.json")

EMG_THRESHOLD = 35.0
MIN_ROWS_PER_BIN = 50
MIN_GROUP_SIZE = 30


# Base loading
def load_xgb_bundle():
    with open(XGB_META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    model = xgb.XGBRegressor()
    model.load_model(XGB_MODEL_PATH)

    features = meta["features"]
    return model, features, meta


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


def load_testing_rows(case_info, label_filter=None):
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
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns required by model: {missing}")

    X_df = df.loc[:, features].copy()
    for c in X_df.columns:
        X_df[c] = pd.to_numeric(X_df[c], errors="coerce")

    return X_df


def make_prediction_frame(df, xgb_model, xgb_features):
    """
    Keep all original columns and add xgb_pred.
    """
    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column missing: {TARGET_COL}")
    if EMG_COL not in df.columns:
        raise KeyError(f"EMG column missing: {EMG_COL}")

    out = df.copy()
    out[EMG_COL] = pd.to_numeric(out[EMG_COL], errors="coerce")
    out[TARGET_COL] = pd.to_numeric(out[TARGET_COL], errors="coerce")

    X_xgb_df = build_numeric_feature_matrix(out, xgb_features)
    X_xgb = X_xgb_df.to_numpy(dtype=np.float32)

    xgb_valid = np.isfinite(out[TARGET_COL].to_numpy(dtype=np.float32))
    xgb_valid &= np.isfinite(out[EMG_COL].to_numpy(dtype=np.float32))
    xgb_valid &= np.all(np.isfinite(X_xgb), axis=1)

    xgb_pred = np.full(len(out), np.nan, dtype=np.float32)
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



# EMG bins and ROI
def choose_emg_edges(emg_values, max_bins=35, min_count_hint=1000):
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
        bis_q95=(TARGET_COL, lambda s: s.quantile(0.95)),
        xgb_q95=("xgb_pred", lambda s: s.quantile(0.95)),
    ).reset_index()

    summary = summary.loc[summary["n"] >= min_rows_per_bin].copy()

    if len(summary) == 0:
        raise RuntimeError("No EMG bins survived min_rows_per_bin threshold.")

    return summary


def attach_bin_quantile_lines(pred_df, emg_edges, summary_df):
    out = pred_df.copy()
    out["emg_bin"] = pd.cut(
        out[EMG_COL],
        bins=emg_edges,
        include_lowest=True,
        right=False,
        duplicates="drop"
    )

    quantiles = summary_df[["emg_bin", "bis_q95", "xgb_q95"]].copy()
    out = out.merge(quantiles, on="emg_bin", how="left")
    return out


def build_irregular_high_emg_roi_frame(case_info, xgb_model, xgb_features):
    df = load_testing_rows(case_info, label_filter="irregular")
    pred_df = make_prediction_frame(df, xgb_model, xgb_features)

    pred_df = pred_df.loc[pred_df[EMG_COL] > EMG_THRESHOLD].copy()
    if len(pred_df) == 0:
        raise RuntimeError(f"No irregular rows with {EMG_COL} > {EMG_THRESHOLD}.")

    emg_edges = choose_emg_edges(pred_df[EMG_COL].to_numpy())
    summary_df = summarize_by_emg_bin(pred_df, emg_edges, min_rows_per_bin=MIN_ROWS_PER_BIN)
    work = attach_bin_quantile_lines(pred_df, emg_edges, summary_df)

    # ROI as corrected by you:
    # measured BIS > binwise xgb_q95 line
    # xgb_pred < binwise bis_q95 line
    roi_mask = (
        np.isfinite(pd.to_numeric(work["xgb_q95"], errors="coerce")) &
        np.isfinite(pd.to_numeric(work["bis_q95"], errors="coerce")) &
        (work[TARGET_COL] > work["xgb_q95"]) &
        (work["xgb_pred"] < work["bis_q95"])
    )

    work["is_roi"] = roi_mask.astype(int)

    print("[INFO] ROI construction on irregular rows with high EMG")
    print(f"  Samples total       : {len(work)}")
    print(f"  Samples in ROI      : {int(work['is_roi'].sum())}")
    print(f"  ROI fraction        : {work['is_roi'].mean():.4f}")

    return work



# residualized feature comparison
def residualize_feature_against_bis(df, feature_col):
    """
    Fit feature ~ measured BIS, then return residuals:
      observed feature - expected feature given BIS
    """
    work = df[[TARGET_COL, feature_col]].copy()
    work[TARGET_COL] = pd.to_numeric(work[TARGET_COL], errors="coerce")
    work[feature_col] = pd.to_numeric(work[feature_col], errors="coerce")
    work = work.dropna()

    if len(work) < 100:
        return None

    X = work[[TARGET_COL]].to_numpy(dtype=np.float32)
    y = work[feature_col].to_numpy(dtype=np.float32)

    model = xgb.XGBRegressor(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=1.0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(X, y)

    y_hat = model.predict(X).astype(np.float32)
    residual = y - y_hat

    out = work.copy()
    out[f"{feature_col}__resid"] = residual
    return out


def compare_residualized_feature(df, feature_col):
    resid_df = residualize_feature_against_bis(df, feature_col)
    if resid_df is None:
        return None

    merged = df[[TARGET_COL, "is_roi"]].copy().join(
        resid_df[[feature_col, f"{feature_col}__resid"]],
        how="inner"
    )

    roi = pd.to_numeric(
        merged.loc[merged["is_roi"] == 1, f"{feature_col}__resid"],
        errors="coerce"
    ).dropna()

    non = pd.to_numeric(
        merged.loc[merged["is_roi"] == 0, f"{feature_col}__resid"],
        errors="coerce"
    ).dropna()

    if len(roi) < MIN_GROUP_SIZE or len(non) < MIN_GROUP_SIZE:
        return None

    roi_mean = float(np.mean(roi))
    non_mean = float(np.mean(non))

    roi_std = float(np.std(roi, ddof=1)) if len(roi) > 1 else 0.0
    non_std = float(np.std(non, ddof=1)) if len(non) > 1 else 0.0
    pooled_std = np.sqrt((roi_std ** 2 + non_std ** 2) / 2.0)

    std_diff = (roi_mean - non_mean) / pooled_std if pooled_std > 0 else np.nan

    return {
        "feature": feature_col,
        "std_diff": float(std_diff),
    }


def run_analysis(df, feature_cols):
    rows = []

    for col in feature_cols:
        if col not in df.columns:
            continue

        result = compare_residualized_feature(df, col)
        if result is not None:
            rows.append(result)

    out = pd.DataFrame(rows)
    out["abs_std_diff"] = out["std_diff"].abs()

    return out.sort_values("abs_std_diff", ascending=False).reset_index(drop=True)


def print_results(result_df):
    print("\n[INFO] All features (sorted by |std_diff|):\n")

    preview = result_df.copy()
    with pd.option_context("display.width", 240, "display.max_rows", None):
        print(preview.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))

def print_sex_breakdown(df):
    work = df.copy()
    work["sex"] = pd.to_numeric(work["sex"], errors="coerce")

    roi = work[work["is_roi"] == 1]
    non = work[work["is_roi"] == 0]

    def summarize(group):
        n = len(group)
        n_male = int((group["sex"] == 1).sum())
        n_female = int((group["sex"] == 0).sum())

        return {
            "n": n,
            "male": n_male,
            "female": n_female,
            "male_frac": n_male / n if n > 0 else np.nan,
            "female_frac": n_female / n if n > 0 else np.nan,
        }

    s_roi = summarize(roi)
    s_non = summarize(non)

    print("\n[INFO] Sex breakdown:")
    print("  ROI:")
    print(f"    n = {s_roi['n']}")
    print(f"    male   = {s_roi['male']} ({s_roi['male_frac']:.3f})")
    print(f"    female = {s_roi['female']} ({s_roi['female_frac']:.3f})")

    print("  Non-ROI:")
    print(f"    n = {s_non['n']}")
    print(f"    male   = {s_non['male']} ({s_non['male_frac']:.3f})")
    print(f"    female = {s_non['female']} ({s_non['female_frac']:.3f})")


if __name__ == "__main__":
    xgb_model, xgb_features, xgb_meta = load_xgb_bundle()
    test_case_info = collect_case_info("Testing")

    print("[INFO] Building irregular high-EMG ROI frame")
    analysis_df = build_irregular_high_emg_roi_frame(
        test_case_info,
        xgb_model,
        xgb_features
    )

    print("[INFO] Running residualized feature comparison")
    result_df = run_analysis(analysis_df, xgb_features)

    print_results(result_df)
    print_sex_breakdown(analysis_df)
