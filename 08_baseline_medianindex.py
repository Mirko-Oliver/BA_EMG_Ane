import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
"""
This SCript Calculates and plots
BIS Median Testing Dataset (Baseline 1)
Rolling Median BIS across SEF (Baseline 2)
"""

TARGET_COL = 'BIS/BIS'
FEATURES = ['BIS/SEF',]
EMG_COL = 'BIS/EMG'
DATA_DIR = "data"
SUFFIX = "_rawdata.parquet"
CLINICAL_INFO = "__clinical_info.csv"

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 12
})


def collect_case_info(dataset):
	"""Collect CaseIDs for the right dataset, and ane_intro_end time for each case"""
	case_info = {}
	file_path = os.path.join(DATA_DIR, CLINICAL_INFO)

	with open(file_path, newline="", encoding="utf-8") as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			if row.get("Dataset") == dataset:
				cid = row.get("caseid")
				ane_time = row.get("ane_intro_end")

				if ane_time is None or ane_time == "":
					continue

				case_info[cid] = float(ane_time)

	return case_info


def create_mask(df, ane_intro_end, label_filter=None):
	"""
	Always removes the ane_intro phase.
	Optional label filter:
	- None        -> keep all labels
	- "regular"   -> only regular rows
	- "irregular" -> only irregular rows
	"""
	mask = df["Time"] > ane_intro_end

	if label_filter is not None:
		mask &= (df["label"] == label_filter)

	return mask


def load_split_arrays(case_info, features, apply_mask=True, label_filter=None):
	X_list, y_list = [], []

	for cid, ane_intro_end in case_info.items():
		path = os.path.join(DATA_DIR, cid + SUFFIX)
		if not os.path.exists(path):
			continue

		df = pd.read_parquet(path)

		if apply_mask:
			mask = create_mask(df, ane_intro_end, label_filter=label_filter)
			df = df.loc[mask].copy()
		else:
			df = df.copy()

		if len(df) == 0:
			continue

		missing = [c for c in features if c not in df.columns]
		if missing:
			raise KeyError(f"Case {cid} missing feature columns: {missing}")

		X = df[features].to_numpy(dtype=np.float32)
		y = df[TARGET_COL].to_numpy(dtype=np.float32)

		X_list.append(X)
		y_list.append(y)

	if not X_list:
		raise RuntimeError("No data loaded (X_list empty).")

	return (
		np.vstack(X_list),
		np.concatenate(y_list),
	)


# Metrics
def mae(y_true, y_pred):
	y_true = np.asarray(y_true, dtype=float)
	y_pred = np.asarray(y_pred, dtype=float)
	return float(np.nanmean(np.abs(y_pred - y_true)))


def mse(y_true, y_pred):
	y_true = np.asarray(y_true, dtype=float)
	y_pred = np.asarray(y_pred, dtype=float)
	return float(np.nanmean((y_pred - y_true) ** 2))


def rmse(y_true, y_pred):
	y_true = np.asarray(y_true, dtype=float)
	y_pred = np.asarray(y_pred, dtype=float)
	return float(np.sqrt(np.nanmean((y_pred - y_true) ** 2)))


# Index 1: Median BIS
def fit_median_index(y_train):
	"""
	Calculates the global median BIS over the training set.
	"""
	return float(np.nanmedian(y_train))


def predict_median_index(median, X):
	"""
	Predicts BIS for all samples using the same global median BIS.
	"""
	return np.full(shape=(len(X),), fill_value=median, dtype=float)


# Index 2: Rolling Median BIS against SEF
def fit_rolling_sef_median_index(X_train, y_train, window_size):
	"""
	Sorts the training data by SEF and computes a rolling median of BIS.
	"""
	sef = X_train[:, 0].astype(float)
	bis = y_train.astype(float)

	valid_mask = ~np.isnan(sef) & ~np.isnan(bis)
	sef = sef[valid_mask]
	bis = bis[valid_mask]

	order = np.argsort(sef)
	sef_sorted = sef[order]
	bis_sorted = bis[order]

	bis_series = pd.Series(bis_sorted)
	rolling_median = bis_series.rolling(
		window=window_size,
		center=True,
		min_periods=1
	).median().to_numpy()

	return {
		"sef_sorted": sef_sorted,
		"rolling_bis_median": rolling_median,
		"window_size": window_size,
	}


def predict_rolling_sef_median_index(model, X):
	"""
	Predicts BIS by mapping each SEF value to the nearest SEF point
	in the training set and using its rolling BIS median.
	"""
	sef_query = X[:, 0].astype(float)
	sef_sorted = model["sef_sorted"]
	rolling_bis_median = model["rolling_bis_median"]

	preds = np.empty(len(sef_query), dtype=float)

	for i, val in enumerate(sef_query):
		if np.isnan(val):
			preds[i] = np.nan
			continue

		idx = np.searchsorted(sef_sorted, val)

		if idx == 0:
			nearest_idx = 0
		elif idx >= len(sef_sorted):
			nearest_idx = len(sef_sorted) - 1
		else:
			left_idx = idx - 1
			right_idx = idx
			if abs(val - sef_sorted[left_idx]) <= abs(val - sef_sorted[right_idx]):
				nearest_idx = left_idx
			else:
				nearest_idx = right_idx

		preds[i] = rolling_bis_median[nearest_idx]

	return preds


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


def plot_single_case(
    median,
    rolling_model,
    case_id,
    median_mae,
    median_rmse,
    baseline_mae,
    baseline_rmse,
):
	"""
	Plots predictions vs actual BIS for a single case.
	"""
	df = pd.read_parquet(os.path.join(DATA_DIR, case_id + SUFFIX))

	X_case = df[FEATURES].to_numpy(dtype=np.float32)
	y_case = df[TARGET_COL].to_numpy(dtype=np.float32)

	pred_global = predict_median_index(median, X_case)
	pred_rolling = predict_rolling_sef_median_index(rolling_model, X_case)

	x = df["Time"].to_numpy()
	x = (x - x.min()) / 60.0

	fig, ax = plt.subplots(figsize=(10, 3.5))
	ax.plot(x, y_case, label="Messdaten", color='black')
	ax.plot(x, pred_rolling, label="Baseline Index", color="#9FB6C4")
	ax.plot(x, pred_global, label="Median", color="#990000")

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

	textstr = (
    f"{'Median:':<16}\n"
    f"{'MAE:':<6}{format_eu(median_mae)}\n"
    f"{'RMSE:':<6}{format_eu(median_rmse)}\n"
    f"{'Baseline Index:':<16}\n"
    f"{'MAE:':<6}{format_eu(baseline_mae)}\n"
    f"{'RMSE:':<6}{format_eu(baseline_rmse)}\n"
	)

	ax.text(
		1.05, 0.6, textstr,
		transform=ax.transAxes,
		ha='left',
		va='top',
		fontsize=12,
		linespacing=1.5,
		family='monospace'
	)
	ax.text(
		1.05, -0.1, f"Fallnummer: {format_eu(int(case_id))}",
		transform=ax.transAxes,
		ha="left",
		va="top"
	)

	ax.legend(loc='upper left', bbox_to_anchor=(1.025, 1))
	plt.tight_layout(rect=[0, 0, 0.85, 1])
	plt.savefig("08p_Baseline_Index.png", dpi=300, bbox_inches="tight")
	plt.show()


def print_metric_block(title, y_true, y_pred):
	print(f"\n{title}")
	print(f"  N    : {np.sum(~np.isnan(y_true) & ~np.isnan(y_pred))}")
	print(f"  MAE  : {mae(y_true, y_pred):.3f}")
	print(f"  RMSE : {rmse(y_true, y_pred):.3f}")
	print(f"  MSE  : {mse(y_true, y_pred):.3f}")


def print_bucket_metrics(y_true, y_pred, bucket_size=20, max_bis=100):
	y_true = np.asarray(y_true, dtype=float)
	y_pred = np.asarray(y_pred, dtype=float)

	print("\nPer BIS bucket:")
	for low in range(1, max_bis + 1, bucket_size):
		high = min(low + bucket_size - 1, max_bis)

		mask = (y_true >= low) & (y_true <= high)
		mask &= ~np.isnan(y_true) & ~np.isnan(y_pred)

		n = np.sum(mask)
		print(f"  BIS {low:>3}-{high:<3} | N={n:>6}", end="")

		if n == 0:
			print(" | MAE=nan | RMSE=nan | MSE=nan")
			continue

		b_mae = mae(y_true[mask], y_pred[mask])
		b_rmse = rmse(y_true[mask], y_pred[mask])
		b_mse = mse(y_true[mask], y_pred[mask])

		print(f" | MAE={b_mae:>7.3f} | RMSE={b_rmse:>7.3f} | MSE={b_mse:>7.3f}")


def evaluate_sef_baseline_index(title, rolling_model, X, y):
	y_pred = predict_rolling_sef_median_index(rolling_model, X)
	print_metric_block(title, y, y_pred)
	print_bucket_metrics(y, y_pred, bucket_size=20, max_bis=100)
	return y_pred


if __name__ == "__main__":
	# Training data: without ane_intro, only regular
	train_case_info = collect_case_info("Training")
	x_train, y_train = load_split_arrays(
		train_case_info,
		FEATURES,
		apply_mask=True,
		label_filter="regular",
	)

	# Fit baselines
	median = fit_median_index(y_train)
	rolling_model = fit_rolling_sef_median_index(x_train, y_train, window_size=501)

	# Testing dataset: all with ane_intro removed
	test_case_info = collect_case_info("Testing")

	# 1) Full testing set without ane_intro, all labels
	x_test_no_ane, y_test_no_ane = load_split_arrays(
		test_case_info,
		FEATURES,
		apply_mask=True,
		label_filter=None,
	)

	# 2) Testing set without ane_intro, only regular
	x_test_regular, y_test_regular = load_split_arrays(
		test_case_info,
		FEATURES,
		apply_mask=True,
		label_filter="regular",
	)

	# 3) Testing set without ane_intro, only irregular
	x_test_irregular, y_test_irregular = load_split_arrays(
		test_case_info,
		FEATURES,
		apply_mask=True,
		label_filter="irregular",
	)

	print("SEF BASELINE INDEX EVALUATION")

	# 1) Full test set, no ane_intro
	y_pred_full = evaluate_sef_baseline_index(
		"SEF Baseline Index on FULL testing dataset (without ane_intro)",
		rolling_model,
		x_test_no_ane,
		y_test_no_ane
	)

	# 2) Test set, regular only
	y_pred_regular = evaluate_sef_baseline_index(
		"SEF Baseline Index on testing dataset (without ane_intro, label = regular)",
		rolling_model,
		x_test_regular,
		y_test_regular
	)

	# 3) Test set, irregular only
	y_pred_irregular = evaluate_sef_baseline_index(
		"SEF Baseline Index on testing dataset (without ane_intro, label = irregular)",
		rolling_model,
		x_test_irregular,
		y_test_irregular
	)

	# Reference: global median on regular masked test data
	print("\nREFERENCE: GLOBAL MEDIAN BASELINE ON TEST DATA (without ane_intro, label = regular)")

	y_pred_index1 = predict_median_index(median, x_test_regular)
	median_mae = mae(y_test_regular, y_pred_index1)
	median_rmse = rmse(y_test_regular, y_pred_index1)
	median_mse = mse(y_test_regular, y_pred_index1)

	print(f"  Median BIS (train): {median:.3f}")
	print(f"  MAE  (regular test): {median_mae:.3f}")
	print(f"  RMSE (regular test): {median_rmse:.3f}")
	print(f"  MSE  (regular test): {median_mse:.3f}")

	# Plot summary against one example case
	baseline_mae = mae(y_test_regular, y_pred_regular)
	baseline_rmse = rmse(y_test_regular, y_pred_regular)

	plot_single_case(
		median,
		rolling_model,
		'46',
		median_mae,
		median_rmse,
		baseline_mae,
		baseline_rmse,
	)
