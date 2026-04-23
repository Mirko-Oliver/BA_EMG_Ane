import os
import csv
import json
import numpy as np
import pandas as pd
import xgboost as xgb

"""
Calculate and Printout Permutation Importance XGBoost
"""
TARGET_COL = "BIS/BIS"
DATA_DIR = "data"
SUFFIX = "_rawdata.parquet"
CLINICAL_INFO = "__clinical_info.csv"

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "bis_xgb_model.json")
META_PATH = os.path.join(MODEL_DIR, "bis_xgb_meta.json")

N_REPEATS = 10
RNG_SEED = 13


def load_xgb_bundle(model_path=MODEL_PATH, meta_path=META_PATH):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    model = xgb.XGBRegressor()
    model.load_model(model_path)

    features = meta["features"]
    target = meta.get("target", TARGET_COL)
    return model, features, target


def collect_case_info(dataset):
    """
    Collect case-level metadata for the given dataset.
    Required:
      - caseid
      - ane_intro_end
      - static features age, sex, lbm
    """
    case_info = {}
    file_path = os.path.join(DATA_DIR, CLINICAL_INFO)

    with open(file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row.get("Dataset") != dataset:
                continue

            cid = row.get("caseid")
            if cid is None:
                continue

            ane_time = row.get("ane_intro_end")
            if ane_time is None or ane_time == "":
                continue

            case_info[cid] = {
                "ane_intro_end": float(ane_time),
                "age": row.get("age"),
                "sex": row.get("sex"),
                "lbm": row.get("lbm"),
            }

    return case_info


def create_training_like_mask(df, ane_intro_end):
    """
    Same logic as training:
      - remove ane_intro phase
      - keep only regular rows
    """
    return (df["Time"] > ane_intro_end) & (df["label"] == "regular")


def add_static_features(df, case_meta):
    for col in ["age", "sex", "lbm"]:
        df[col] = case_meta.get(col)
    return df


def coerce_static_types(df):
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
    if "lbm" in df.columns:
        df["lbm"] = pd.to_numeric(df["lbm"], errors="coerce")
    if "sex" in df.columns:
        df["sex"] = df["sex"].astype(str).str.lower().map({"m": 1, "f": 0})
    return df


def load_masked_regular_arrays(case_info, features):
    """
    Loads one split using the same effective mask as training:
      (Time > ane_intro_end) & (label == 'regular')
    """
    X_list, y_list = [], []

    for cid, meta in case_info.items():
        path = os.path.join(DATA_DIR, cid + SUFFIX)
        if not os.path.exists(path):
            continue

        df = pd.read_parquet(path)
        df = df.loc[create_training_like_mask(df, meta["ane_intro_end"])].copy()

        if len(df) == 0:
            continue

        df = add_static_features(df, meta)
        df = coerce_static_types(df)

        missing = [c for c in features if c not in df.columns]
        if missing:
            raise KeyError(f"Case {cid} missing feature columns: {missing}")

        X = df[features].to_numpy(dtype=np.float32)
        y = df[TARGET_COL].to_numpy(dtype=np.float32)

        X_list.append(X)
        y_list.append(y)

    if not X_list:
        raise RuntimeError("No rows loaded after ane_intro + regular masking.")

    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)
    return X_all, y_all


def filter_finite(X, y):
    """
    Match evaluation-time robustness:
      - finite target
      - finite all features
    """
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    return X[mask], y[mask]


# Metrics
def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.nanmean(np.abs(y_pred - y_true)))


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.nanmean((y_pred - y_true) ** 2)))


# Permutation importance

def permutation_importance_rmse(model, X, y, features, n_repeats=N_REPEATS, rng_seed=RNG_SEED):
    rng = np.random.default_rng(rng_seed)

    baseline_pred = model.predict(X)
    baseline_rmse = rmse(y, baseline_pred)

    results = []

    for j, feat in enumerate(features):
        deltas = []

        for _ in range(n_repeats):
            X_perm = X.copy()
            perm_idx = rng.permutation(len(X_perm))
            X_perm[:, j] = X_perm[perm_idx, j]

            pred_perm = model.predict(X_perm)
            perm_rmse = rmse(y, pred_perm)

            deltas.append(perm_rmse - baseline_rmse)

        deltas = np.asarray(deltas, dtype=float)

        results.append({
            "feature": feat,
            "importance_mean": float(np.mean(deltas)),
            "importance_std": float(np.std(deltas, ddof=0)),
            "importance_min": float(np.min(deltas)),
            "importance_max": float(np.max(deltas)),
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("importance_mean", ascending=False).reset_index(drop=True)

    return baseline_rmse, results_df


def print_ranked_importance(results_df, top_n=None):
    if top_n is None:
        top_n = len(results_df)

    print("\nRANKED PERMUTATION IMPORTANCE (higher = more important)")
    print("-" * 95)
    print(f"{'Rank':>4}  {'Feature':<35}  {'Mean ΔRMSE':>12}  {'Std':>10}  {'Min':>10}  {'Max':>10}")
    print("-" * 95)

    for i, row in results_df.head(top_n).iterrows():
        print(
            f"{i+1:>4}  "
            f"{row['feature']:<35}  "
            f"{row['importance_mean']:>12.6f}  "
            f"{row['importance_std']:>10.6f}  "
            f"{row['importance_min']:>10.6f}  "
            f"{row['importance_max']:>10.6f}"
        )

    print("-" * 95)


if __name__ == "__main__":
    model, features, target_name = load_xgb_bundle()
    dataset_name = "Testing"

    print("XGBOOST PERMUTATION IMPORTANCE")
    print(f"Model   : {MODEL_PATH}")
    print(f"Meta    : {META_PATH}")
    print(f"Dataset : {dataset_name}")
    print("Mask    : (Time > ane_intro_end) & (label == 'regular')")

    case_info = collect_case_info(dataset_name)
    X, y = load_masked_regular_arrays(case_info, features)
    X, y = filter_finite(X, y)

    print(f"Rows after mask + finite filter: {len(y):,}")
    print(f"Number of features            : {X.shape[1]}")

    baseline_rmse, results_df = permutation_importance_rmse(
        model=model,
        X=X,
        y=y,
        features=features,
        n_repeats=N_REPEATS,
        rng_seed=RNG_SEED,
    )

    baseline_pred = model.predict(X)
    baseline_mae = mae(y, baseline_pred)

    print(f"\nBaseline performance on {dataset_name} (masked regular rows):")
    print(f"  MAE  : {baseline_mae:.6f}")
    print(f"  RMSE : {baseline_rmse:.6f}")

    print_ranked_importance(results_df)

