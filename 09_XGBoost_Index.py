""" This Script creates a Index with XGBoost, it does not have Causal restrictions, and does not use time dependendcies
1) Collect Case_IDS in dataset 
2) Import Static Features
3) Create Trainingweights

"""
import os
import csv
import sys
import json
import joblib
import pandas as pd
import numpy as np

from typing import Dict, List, Tuple

import xgboost as xgb

DATA_DIR = "data"
SUFFIX = "_rawdata.parquet"
CLINICAL_INFO = "__clinical_info.csv"

Static_Features = ["age", "sex", "lbm"] 
Dynamic_Features = [
    'Solar8000/ART_DBP', 'Solar8000/ART_MBP', 'Solar8000/ART_SBP',
    'Solar8000/BT', "Solar8000/PLETH_HR",
    'Solar8000/ETCO2', 'Solar8000/FEO2', 'Solar8000/FIO2',
    'Solar8000/INCO2', 'Solar8000/PLETH_SPO2',
    'Solar8000/RR_CO2',

    'Orchestra/PPF20_CE',
    'Orchestra/RFTN20_CE',
 
    'BIS/SEF', 'BIS/SR', "BIS/TOTPOW",

    'EEG1_delta_rel_std30', 'EEG1_delta_rel_slope30',
    'EEG1_theta_rel_std30', 'EEG1_theta_rel_slope30',
    'EEG1_alpha_rel_std30', 'EEG1_alpha_rel_slope30',
    'EEG1_beta_rel_std30', 'EEG1_beta_rel_slope30',
    'EEG1_gamma_rel_std30', 'EEG1_gamma_rel_slope30',

    'BIS/SEF_std30', 'BIS/SEF_slope30',
    "BIS/TOTPOW_std30", "BIS/TOTPOW_slope30",
    "Solar8000/PLETH_HR_std60", "Solar8000/PLETH_HR_slope60",
    'Solar8000/ART_MBP_std60', 'Solar8000/ART_MBP_slope60',
    'Solar8000/ETCO2_std60', 'Solar8000/ETCO2_slope60',
    'Solar8000/BT_std60', 'Solar8000/BT_slope60',


    'EEG1_delta_rel', 'EEG1_theta_rel', 'EEG1_alpha_rel',
    'EEG1_beta_rel', 'EEG1_gamma_rel',
    'EEG1_rbr',
]

TARGET_COL = 'BIS/BIS'
MAX_ROWS_PER_BIS_BUCKET = 50_000
RNG_SEED = 13

BIS_EDGES = np.array([6,11,16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71, 76, 81, 86, 91, 96, 101], dtype=float)
N_BINS = len(BIS_EDGES) - 1

def collect_case_ids(dataset):
	# This Function collects all Case_Ids within a dataset (Training/ Testing/ Validation)
	case_ids = []
	file_path = os.path.join(DATA_DIR, CLINICAL_INFO)

	with open(file_path, newline="", encoding="utf-8") as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			if row.get("Dataset") == dataset:
				case_ids.append(row.get("caseid"))

	return case_ids

def create_mask(df, caseid, clinical_map):
	"""
	Filter for regular datapoints and remove start of case using
	case-specific ane_intro_end from clinical_map.
	"""
	case_info = clinical_map.get(caseid, {})
	cutoff = pd.to_numeric(case_info.get("ane_intro_end"), errors="coerce")

	if pd.isna(cutoff):
		raise ValueError(f"Case {caseid} has invalid or missing ane_intro_end in {CLINICAL_INFO}")

	mask = (df["label"] == "regular") & (df["Time"] > cutoff)
	return mask
	
def make_bis_bins(y):
    y_num = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)

    b = np.digitize(y_num, BIS_EDGES, right=False) - 1  
    b[np.isnan(y_num)] = -1
    b[(y_num < BIS_EDGES[0]) | (y_num >= BIS_EDGES[-1])] = -1
    return b.astype(int)

def scan_case_bucket_counts(case_ids, clinical_map):
    out = {}

    for cid in case_ids:
        path = os.path.join(DATA_DIR, cid + SUFFIX)
        if not os.path.exists(path):
            continue

        df = pd.read_parquet(path)
        df = df.loc[create_mask(df, cid, clinical_map)]
        if len(df) == 0:
            continue

        bins = make_bis_bins(df[TARGET_COL].to_numpy())
        valid = bins >= 0
        if not np.any(valid):
            continue

        counts = np.bincount(bins[valid], minlength=N_BINS).astype(int)
        out[cid] = counts

    return out


def allocate_global_bucket_quotas(case_bucket_counts, max_rows_per_bucket=MAX_ROWS_PER_BIS_BUCKET):
    """
    For each BIS bucket:
      - if total rows <= cap, keep all
      - else allocate per-case quotas proportional to each case's contribution
        using largest-remainder rounding
    Returns: dict {caseid -> np.ndarray shape (N_BINS,)}
    """
    case_ids = sorted(case_bucket_counts.keys())
    if not case_ids:
        return {}

    total_per_bucket = np.zeros(N_BINS, dtype=int)
    for cid in case_ids:
        total_per_bucket += case_bucket_counts[cid]

    quota_map = {
        cid: np.zeros(N_BINS, dtype=int) for cid in case_ids
    }

    for b in range(N_BINS):
        total_b = int(total_per_bucket[b])
        if total_b == 0:
            continue

        counts_b = {cid: int(case_bucket_counts[cid][b]) for cid in case_ids}
        active = [cid for cid in case_ids if counts_b[cid] > 0]

        if total_b <= max_rows_per_bucket:
            for cid in active:
                quota_map[cid][b] = counts_b[cid]
            continue

        raw = {
            cid: max_rows_per_bucket * (counts_b[cid] / total_b)
            for cid in active
        }

        base = {
            cid: min(counts_b[cid], int(np.floor(raw[cid])))
            for cid in active
        }

        assigned = sum(base.values())
        remainder = max_rows_per_bucket - assigned

        order = sorted(
            active,
            key=lambda cid: (raw[cid] - np.floor(raw[cid]), counts_b[cid]),
            reverse=True,
        )

        i = 0
        while remainder > 0 and order:
            cid = order[i % len(order)]
            if base[cid] < counts_b[cid]:
                base[cid] += 1
                remainder -= 1
            i += 1

            if i > 10 * len(order) and all(base[x] >= counts_b[x] for x in order):
                break

        for cid in active:
            quota_map[cid][b] = base[cid]

    return quota_map


def sample_case_by_bucket_quota(df, case_bucket_quota, target_col=TARGET_COL, rng=None):
    """
    Sample rows from one case so that for each BIS bucket b,
    up to case_bucket_quota[b] rows are retained.
    """
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)

    y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=float)
    bins = make_bis_bins(y)

    selected_idx = []

    for b in range(N_BINS):
        q = int(case_bucket_quota[b])
        if q <= 0:
            continue

        idx_b = np.flatnonzero(bins == b)
        if len(idx_b) == 0:
            continue

        if len(idx_b) <= q:
            chosen = idx_b
        else:
            chosen = rng.choice(idx_b, size=q, replace=False)

        selected_idx.append(np.sort(chosen))

    if not selected_idx:
        return df.iloc[0:0].copy()

    selected_idx = np.concatenate(selected_idx)
    selected_idx.sort()
    return df.iloc[selected_idx].copy()


def print_global_bucket_cap_summary(case_bucket_counts, case_quota_map):
    total_before = np.zeros(N_BINS, dtype=int)
    total_after = np.zeros(N_BINS, dtype=int)

    for cid, counts in case_bucket_counts.items():
        total_before += counts
        total_after += case_quota_map.get(cid, np.zeros(N_BINS, dtype=int))

    print("\n[INFO] Global BIS bucket cap summary:")
    for b in range(N_BINS):
        lo, hi = BIS_EDGES[b], BIS_EDGES[b + 1]
        print(
            f"  [{lo:>5.1f}, {hi:>5.1f}): "
            f"before={int(total_before[b]):>8d}  after={int(total_after[b]):>8d}"
        )
    
def scan_training_stats(case_ids, case_quota_map, clinical_map, rng_seed=RNG_SEED):
	case_len_map = {}
	bin_counts = np.zeros(N_BINS, dtype=int)
	rng = np.random.default_rng(rng_seed)

	for cid in case_ids:
		quota = case_quota_map.get(cid)
		if quota is None:
			continue

		path = os.path.join(DATA_DIR, cid + SUFFIX)
		if not os.path.exists(path):
			continue

		df = pd.read_parquet(path)
		df = df.loc[create_mask(df, cid, clinical_map)].copy()
		if len(df) == 0:
			continue

		df = sample_case_by_bucket_quota(df, quota, rng=rng)
		if len(df) == 0:
			continue

		case_len_map[cid] = len(df)

		bins = make_bis_bins(df[TARGET_COL].to_numpy())
		valid = bins >= 0
		counts = np.bincount(bins[valid], minlength=N_BINS)
		bin_counts += counts

	return case_len_map, bin_counts

def calculate_bin_weight(bin_counts, min_count=200, alpha=1):
	"""Takes the BIS Bin Counts and calculates the corresponding weights.
	Uses min_count so weight doesn't become too heavy for small bins,
	 and alpha to set aggressiveness in balancing"""
	freq = bin_counts.astype(float)
	freq[freq < min_count] = min_count
	return 1.0 / np.power(freq, alpha)
    
def load_training_arrays(case_ids, features, case_len_map, bin_w_lookup, clinical_map, case_quota_map, rng_seed=RNG_SEED):
	X_list, y_list, w_list = [], [], []
	rng = np.random.default_rng(rng_seed)

	for cid in case_ids:
		if cid not in case_len_map:
			continue

		quota = case_quota_map.get(cid)
		if quota is None:
			continue

		path = os.path.join(DATA_DIR, cid + SUFFIX)
		if not os.path.exists(path):
			continue

		df = pd.read_parquet(path)
		df = df.loc[create_mask(df, cid, clinical_map)].copy()
		if len(df) == 0:
			continue

		df = sample_case_by_bucket_quota(df, quota, rng=rng)
		if len(df) == 0:
			continue

		df = add_static_features(df, cid, clinical_map)
		df = coerce_static_types(df)

		missing = [c for c in features if c not in df.columns]
		if missing:
			raise KeyError(f"Case {cid} missing feature columns: {missing}")

		X = df[features].to_numpy(dtype=np.float32)
		y = df[TARGET_COL].to_numpy(dtype=np.float32)

		w_case = np.full(len(df), 1.0 / max(case_len_map[cid], 1), dtype=np.float32)

		bins = make_bis_bins(y)
		valid = bins >= 0

		if not np.all(valid):
			X = X[valid]
			y = y[valid]
			w_case = w_case[valid]
			bins = bins[valid]

		w_bin = bin_w_lookup[bins].astype(np.float32)
		w = w_case * w_bin

		X_list.append(X)
		y_list.append(y)
		w_list.append(w)

	if not X_list:
		raise RuntimeError("No training data loaded (X_list empty)")

	X_all = np.vstack(X_list)
	y_all = np.concatenate(y_list)
	w_all = np.concatenate(w_list)

	w_all = w_all * (len(w_all) / w_all.sum())
	return X_all, y_all, w_all

def load_split_arrays(case_ids, features, clinical_map):
	#basically load_training_arrays w/o weight
	X_list, y_list = [], []

	for cid in case_ids:
		path = os.path.join(DATA_DIR, cid + SUFFIX)
		if not os.path.exists(path):
			continue

		df = pd.read_parquet(path)

		mask = create_mask(df, cid, clinical_map)
		df = df.loc[mask].copy()

		# add statics
		df = add_static_features(df, cid, clinical_map)
		df = coerce_static_types(df)

		missing = [c for c in features if c not in df.columns]
		if missing:
			raise KeyError(f"Case {cid} missing feature columns: {missing}")

		X = df[features].to_numpy(dtype=np.float32)
		y = df[TARGET_COL].to_numpy(dtype=np.float32)

		X_list.append(X)
		y_list.append(y)

	return (
		np.vstack(X_list),
		np.concatenate(y_list),
	)
	
def load_clinical_info_map(static_cols=Static_Features):
    """
    Returns dict mapping caseid to:
      - static feature values
      - ane_intro_end for per-case masking
    """
    fpath = os.path.join(DATA_DIR, CLINICAL_INFO)
    info = {}

    needed_cols = list(static_cols) + ["ane_intro_end"]

    with open(fpath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row.get("caseid")
            if cid is None:
                continue
            info[cid] = {c: row.get(c) for c in needed_cols}

    return info

def add_static_features(df, caseid, clinical_map):
    """
    Adds static columns to every row of df for this caseid.
    """

    statics = clinical_map[caseid]
    for k, v in statics.items():
        df[k] = v
    return df
    
def coerce_static_types(df):
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
    if "sex" in df.columns:
        df["sex"] = df["sex"].astype(str).str.lower().map({"m": 1, "f": 0, })
    return df
    
def mae(y_true, y_pred):
	y_true = np.asarray(y_true, dtype=float)
	y_pred = np.asarray(y_pred, dtype=float)
	return float(np.nanmean(np.abs(y_pred - y_true)))

def rmse(y_true, y_pred):
	y_true = np.asarray(y_true, dtype=float)
	y_pred = np.asarray(y_pred, dtype=float)
	return float(np.sqrt(np.nanmean((y_pred - y_true) ** 2)))

def print_metrics(name, y_true, y_pred):
	print(f"[METRIC] {name} MAE:  {mae(y_true, y_pred):.4f}")
	print(f"[METRIC] {name} RMSE: {rmse(y_true, y_pred):.4f}")

def print_metrics_by_bin(name, y_true, y_pred):
	b = make_bis_bins(y_true)
	for bi in range(N_BINS):
		idx = (b == bi)
		n = int(np.sum(idx))
		if n == 0:
			continue
		print(f"[BIN] {name} bin={bi:02d} n={n:7d}  MAE={mae(y_true[idx], y_pred[idx]):.4f}  RMSE={rmse(y_true[idx], y_pred[idx]):.4f}")

def print_metrics_by_sex(name, X, y_true, y_pred, features):
	sex_i = features.index("sex")
	sex = X[:, sex_i]

	mask_m = (sex == 1)
	mask_f = (sex == 0)

	nm = int(np.sum(mask_m))
	nf = int(np.sum(mask_f))

	if nm > 0:
		print(f"[SEX] {name} sex=1 (M) n={nm:7d}  MAE={mae(y_true[mask_m], y_pred[mask_m]):.4f}  RMSE={rmse(y_true[mask_m], y_pred[mask_m]):.4f}")
	if nf > 0:
		print(f"[SEX] {name} sex=0 (F) n={nf:7d}  MAE={mae(y_true[mask_f], y_pred[mask_f]):.4f}  RMSE={rmse(y_true[mask_f], y_pred[mask_f]):.4f}")

def save_model_bundle(model, features, out_dir, bin_edges):
	os.makedirs(out_dir, exist_ok=True)

	model_path = os.path.join(out_dir, "bis_xgb_model.json")
	meta_path  = os.path.join(out_dir, "bis_xgb_meta.json")

	model.get_booster().save_model(model_path)

	meta = {
		"features": features,
		"target": TARGET_COL,
		"bis_edges": bin_edges.tolist(),
	}

	with open(meta_path, "w", encoding="utf-8") as f:
		json.dump(meta, f, indent=2)

	print(f"[SAVE] model -> {model_path}")
	print(f"[SAVE] meta  -> {meta_path}")
	    
if __name__ == "__main__":
	# 1) Create Map of Static Features
	clinical_map = load_clinical_info_map(Static_Features)
	# 2) Collect training case ids
	train_case_ids = collect_case_ids("Training")

	# 3) Scan per-case BIS bucket counts
	print("[INFO] Scanning per-case BIS bucket counts...")
	case_bucket_counts = scan_case_bucket_counts(train_case_ids, clinical_map)

	# 4) Allocate global per-bucket quotas
	print("[INFO] Allocating global per-bucket quotas...")
	case_quota_map = allocate_global_bucket_quotas(
		case_bucket_counts,
		max_rows_per_bucket=MAX_ROWS_PER_BIS_BUCKET,
	)
	print_global_bucket_cap_summary(case_bucket_counts, case_quota_map)

	# 5) find usable sampled lengths and sampled BIS bin counts
	case_len_map, bin_counts = scan_training_stats(
		train_case_ids,
		case_quota_map,
		clinical_map,
		rng_seed=RNG_SEED,
	)

	# 6) Build global bin weight lookup from sampled training data
	bin_w_lookup = calculate_bin_weight(bin_counts)

	# 7) load sampled training arrays
	FEATURES = Static_Features + Dynamic_Features
	X_train, y_train, w_train = load_training_arrays(
		train_case_ids,
		FEATURES,
		case_len_map,
		bin_w_lookup,
		clinical_map,
		case_quota_map,
		rng_seed=RNG_SEED,
	)
	print(len(X_train))
	# 8) Load Validation Arrays 
	val_case_ids = collect_case_ids("Validation")
	X_val, y_val = load_split_arrays(val_case_ids, FEATURES, clinical_map)

	# 9) Build XGBoost Model
	model = xgb.XGBRegressor(
		objective="reg:squarederror",
		eval_metric="mae",
		n_estimators=5000,
		learning_rate=0.05,
		max_depth=6,
		subsample=0.8,
		colsample_bytree=0.8,
		reg_lambda=1.0,
		random_state=42,
		n_jobs=-1,
		early_stopping_rounds=100, 
	)
	# 10) Train with Early Stopping on Validation
	model.fit(
		X_train, y_train,
		sample_weight=w_train,
		eval_set=[(X_val, y_val)],
		verbose=50,
	)
	# 11) Evaluate on Validation
	y_val_pred = model.predict(X_val)
	print_metrics("VAL", y_val, y_val_pred)
	print_metrics_by_bin("VAL", y_val, y_val_pred)
	print_metrics_by_sex("VAL", X_val, y_val, y_val_pred, FEATURES)
	
	# 12)Load Test Arrays
	test_case_ids = collect_case_ids("Testing")
	X_test, y_test = load_split_arrays(test_case_ids, FEATURES, clinical_map)

	y_test_pred = model.predict(X_test)
	print_metrics("TEST", y_test, y_test_pred)
	print_metrics_by_bin("TEST", y_test, y_test_pred)
	print_metrics_by_sex("TEST", X_test, y_test, y_test_pred, FEATURES)
	
	# 13) Save model bundle
	save_model_bundle(
		model,
		FEATURES,
		out_dir="model",
		bin_edges=BIS_EDGES
	)




