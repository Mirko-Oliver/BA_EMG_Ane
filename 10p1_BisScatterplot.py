import os
import csv
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import xgboost as xgb

"""
Create Heaxa-Density "Scatter" Plot for BIS vs pred BIS. For both EMG Regular and Irregular
"""
TARGET_COL = "BIS/BIS"
DATA_DIR = "data"
SUFFIX = "_rawdata.parquet"
CLINICAL_INFO = "__clinical_info.csv"

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "bis_xgb_model.json")
META_PATH = os.path.join(MODEL_DIR, "bis_xgb_meta.json")

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 12
})


def load_xgb_bundle():
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)

    return model, meta["features"]


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
    df["age"] = pd.to_numeric(df.get("age"), errors="coerce")
    df["lbm"] = pd.to_numeric(df.get("lbm"), errors="coerce")
    df["sex"] = df.get("sex").astype(str).str.lower().map({"m": 1, "f": 0})
    return df


def load_split_arrays(case_info, features, label_filter=None):
    X_list, y_list = [], []

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

        X_list.append(df[features].to_numpy(dtype=np.float32))
        y_list.append(df[TARGET_COL].to_numpy(dtype=np.float32))

    if len(X_list) == 0:
        return np.empty((0, len(features)), dtype=np.float32), np.empty((0,), dtype=np.float32)

    return np.vstack(X_list), np.concatenate(y_list)


def filter_finite(X, y):
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    return X[mask], y[mask]


def filter_bis_range(y_true, y_pred, low=0, high=100):
    mask = (
        np.isfinite(y_true) &
        np.isfinite(y_pred) &
        (y_true >= low) & (y_true <= high) &
        (y_pred >= low) & (y_pred <= high)
    )
    return y_true[mask], y_pred[mask]


# Helpers
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


def format_eu(n, width=10):
    if isinstance(n, (int, np.integer)):
        s = f"{int(n):,}".replace(",", ".")
    else:
        s = f"{float(n):.2f}".rstrip("0").rstrip(".")
        s = s.replace(".", ",")
    return f"{s:>{width}}"



# Plot
def plot_bis_vs_prediction(y_regular, pred_regular, y_irregular, pred_irregular):
	fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

	panels = [
		(axes[0], y_regular, pred_regular, "Regulär"),
		(axes[1], y_irregular, pred_irregular, "Irregulär"),
	]

	hb_last = None

	for ax, x, y, title in panels:
		hb = ax.hexbin(
			x,
			y,
			gridsize=55,
			extent=(0, 100, 0, 100),
			mincnt=1,
			norm=LogNorm(),
			linewidths=0
		)
		hb_last = hb

		ax.plot([0, 100], [0, 100], color="black", linewidth=1.2, linestyle="--")

		ax.set_xlim(0, 100)
		ax.set_ylim(0, 100)
		ax.set_aspect("equal", adjustable="box")
		ax.set_box_aspect(1)

		ax.set_title(title)
		ax.grid(True, linewidth=1, color="black", alpha=1.0)

		ax.spines["left"].set_linewidth(1.5)
		ax.spines["bottom"].set_linewidth(1.5)
		ax.spines["right"].set_linewidth(1.0)
		ax.spines["top"].set_linewidth(1.0)

		replace_second_last_tick_with_unit(ax, axis="x", unit="--")
		replace_second_last_tick_with_unit(ax, axis="y", unit="--")

		ax.text(
			0.02, 1.02, f"N = {format_eu(len(x), width=0)}",
			transform=ax.transAxes,
			ha="left",
			va="bottom"
		)

	axes[0].set_xlabel("Bispektralindex ($BIS$)")
	axes[1].set_xlabel("Bispektralindex ($BIS$)")
	axes[0].set_ylabel("XGBoost-Index ($BIS_{pred}$)")
	axes[1].set_ylabel("XGBoost-Index ($BIS_{pred}$)")

	plt.subplots_adjust(right=0.88)

	cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
	cbar = fig.colorbar(hb_last, cax=cbar_ax)
	cbar.set_label("Anzahl pro Hexagon")

	plt.savefig("10p1_BIS_Scatterplot.png", dpi=300, bbox_inches="tight")
	plt.show()


if __name__ == "__main__":
    model, features = load_xgb_bundle()
    test_case_info = collect_case_info("Testing")

    # REGULAR
    X_regular, y_regular = load_split_arrays(test_case_info, features, label_filter="regular")
    X_regular, y_regular = filter_finite(X_regular, y_regular)
    pred_regular = model.predict(X_regular)
    y_regular, pred_regular = filter_bis_range(y_regular, pred_regular, low=0, high=100)

    # IRREGULAR
    X_irregular, y_irregular = load_split_arrays(test_case_info, features, label_filter="irregular")
    X_irregular, y_irregular = filter_finite(X_irregular, y_irregular)
    pred_irregular = model.predict(X_irregular)
    y_irregular, pred_irregular = filter_bis_range(y_irregular, pred_irregular, low=0, high=100)

    plot_bis_vs_prediction(
        y_regular, pred_regular,
        y_irregular, pred_irregular
    )
