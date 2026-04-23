import os
import csv
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
"""
Plots XGBoost Index for one Case
Several Printouts for Errorterms used in BA
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


def load_split_arrays(case_info, features, label_filter=None, return_meta=False):
    X_list, y_list = [], []
    age_list, sex_list = [], []

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

        if return_meta:
            age_list.append(df["age"].to_numpy(dtype=np.float32))
            sex_list.append(df["sex"].to_numpy(dtype=np.float32))

    if len(X_list) == 0:
        if return_meta:
            return (
                np.empty((0, len(features)), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
        return (
            np.empty((0, len(features)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    if return_meta:
        age = np.concatenate(age_list)
        sex = np.concatenate(sex_list)
        return X, y, age, sex

    return X, y


def filter_finite(X, y, *extra_arrays):
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)

    filtered = [X[mask], y[mask]]
    for arr in extra_arrays:
        filtered.append(arr[mask])

    return tuple(filtered)

# Metrics
def mae(y_true, y_pred):
    return float(np.nanmean(np.abs(y_pred - y_true)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.nanmean((y_pred - y_true) ** 2)))


def mse(y_true, y_pred):
    return float(np.nanmean((y_pred - y_true) ** 2))


def print_metric_block(title, y_true, y_pred):
    print(f"\n{title}")
    print(f"  N    : {len(y_true)}")
    print(f"  MAE  : {mae(y_true, y_pred):.3f}")
    print(f"  RMSE : {rmse(y_true, y_pred):.3f}")
    print(f"  MSE  : {mse(y_true, y_pred):.3f}")


def print_bucket_metrics(y_true, y_pred, bucket_size=20):
    print("\nPer BIS bucket:")
    for low in range(1, 101, bucket_size):
        high = min(low + bucket_size - 1, 100)
        m = (y_true >= low) & (y_true <= high)

        if np.sum(m) == 0:
            continue

        print(
            f"BIS {low:3}-{high:<3} | "
            f"N={np.sum(m):6} | "
            f"MAE={mae(y_true[m], y_pred[m]):.3f} | "
            f"RMSE={rmse(y_true[m], y_pred[m]):.3f}"
        )



# Grouped eval

def print_group_metrics(header, y_true, y_pred, mask):
    n = int(np.sum(mask))
    print(f"{header:<18} | N={n:6}", end="")

    if n == 0:
        print(" | no valid samples")
        return

    print(
        f" | MAE={mae(y_true[mask], y_pred[mask]):.3f}"
        f" | RMSE={rmse(y_true[mask], y_pred[mask]):.3f}"
        f" | MSE={mse(y_true[mask], y_pred[mask]):.3f}"
    )


def print_age_and_sex_metrics(y_true, y_pred, age, sex):
    print("\nAge-binned performance:")
    age_bins = [0, 20, 40, 60, 80, np.inf]
    age_labels = ["0-19", "20-39", "40-59", "60-79", "80+"]

    for i, label in enumerate(age_labels):
        low = age_bins[i]
        high = age_bins[i + 1]

        if np.isinf(high):
            mask = age >= low
        else:
            mask = (age >= low) & (age < high)

        print_group_metrics(f"Age {label}", y_true, y_pred, mask)

    print("\nSex-separated performance:")
    print_group_metrics("Male", y_true, y_pred, sex == 1)
    print_group_metrics("Female", y_true, y_pred, sex == 0)


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


# Plot
def plot_single_case(model, features, case_id, meta):
    df = pd.read_parquet(os.path.join(DATA_DIR, case_id + SUFFIX))
    df = df.copy()

    df = add_static_features(df, meta)
    df = coerce_static_types(df)

    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(f"Case {case_id} missing feature columns: {missing}")

    X = df[features].to_numpy(dtype=np.float32)
    y = df[TARGET_COL].to_numpy(dtype=np.float32)
    valid_y = np.isfinite(y)

    X_plot = X[valid_y]
    y_plot = y[valid_y]
    df_plot = df.loc[valid_y]

    pred = model.predict(X_plot)
    pred[~np.isfinite(y_plot) | (y_plot == 0)] = np.nan

    x = df_plot["Time"].to_numpy(dtype=float)
    x = (x - x.min()) / 60.0

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(x, y_plot, label="Messdaten", color="black")
    ax.plot(x, pred, label="XGBoost Index", color="#9FB6C4")

    ax.set_xlabel("Operationszeit ($t_{OP}$)")
    ax.set_ylabel("Bispektralindex ($BIS$)")
    ax.set_xlim(left=0, right=175)
    ax.set_ylim(bottom=0, top=100)

    ax.grid(True, linewidth=1, color='black', alpha=1.0)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    replace_second_last_tick_with_unit(ax, axis="y", unit="--")
    replace_second_last_tick_with_unit(ax, axis="x", unit="min")

    def format_eu(n, width=10):
        if isinstance(n, (int, np.integer)):
            s = f"{int(n):,}".replace(",", ".")
        else:
            s = f"{float(n):.2f}".rstrip("0").rstrip(".")
            s = s.replace(".", ",")
        return f"{s:>{width}}"

    ax.text(
        1.05, -0.1, f"Fallnummer: {format_eu(int(case_id))}",
        transform=ax.transAxes,
        ha="left",
        va="top"
    )

    ax.legend(loc='upper left', bbox_to_anchor=(1.025, 1))

    plt.tight_layout()
    plt.savefig("09p1_XGBoost_Index.png", dpi=300, bbox_inches="tight")
    plt.show()


def evaluate_dataset(model, features, dataset_name):
    case_info = collect_case_info(dataset_name)

    print(f"\n{'=' * 60}")
    print(f"XGBOOST EVALUATION - {dataset_name.upper()}")
    print(f"{'=' * 60}")

    eval_configs = [
        ("FULL", None),
        ("REGULAR", "regular"),
        ("IRREGULAR", "irregular"),
    ]

    for split_name, label_filter in eval_configs:
        X, y, age, sex = load_split_arrays(
            case_info,
            features,
            label_filter=label_filter,
            return_meta=True
        )
        X, y, age, sex = filter_finite(X, y, age, sex)

        if len(y) == 0:
            print(f"\n{split_name} {dataset_name.upper()}")
            print("  No valid samples found.")
            continue

        pred = model.predict(X)

        print_metric_block(f"{split_name} {dataset_name.upper()}", y, pred)
        print_bucket_metrics(y, pred)
        print_age_and_sex_metrics(y, pred, age, sex)


if __name__ == "__main__":
    model, features = load_xgb_bundle()

    # Training metrics
    evaluate_dataset(model, features, "Training")

    # Test metrics
    evaluate_dataset(model, features, "Testing")

    # Plot example test case
    test_case_info = collect_case_info("Testing")
    plot_single_case(model, features, "46", test_case_info["46"])
