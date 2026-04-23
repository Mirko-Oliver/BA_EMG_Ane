import numpy as np
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt

DATA_DIR = Path("data")

def collect_case_ids(data_dir=DATA_DIR):
    """
    Return a sorted list of case IDs from existing *_rawdata.parquet files.
    """
    case_ids = sorted(
        int(fname.split("_")[0])
        for fname in os.listdir(data_dir)
        if fname.endswith("_rawdata.parquet")
    )
    return case_ids


def load_case_parquet(case_id, data_dir=DATA_DIR):
    """
    Load the parquet file for a given case ID.
    """
    path = data_dir / f"{case_id}_rawdata.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    return pd.read_parquet(path), path


def rolling_slope(series, window):
    """
    Compute rolling linear slope over a fixed window.
    Assumes evenly spaced time (1 Hz).
    """
    n = window

    x = np.arange(n)
    x_mean = x.mean()
    denom = np.sum((x - x_mean) ** 2)

    def slope_fn(y_window):
        if np.any(np.isnan(y_window)):
            return np.nan
        y_mean = y_window.mean()
        return np.sum((x - x_mean) * (y_window - y_mean)) / denom

    return series.rolling(window=window, min_periods=window).apply(
        slope_fn, raw=True
    )


def build_temporal_features_for_column(
    df: pd.DataFrame,
    col: str,
    windows=(30, 60),
):
    """
    Adds rolling mean, std, and slope for given column.
    Returns list of created column names.
    """

    new_cols = []

    for w in windows:

        mean_col = f"{col}_mean{w}"
        std_col = f"{col}_std{w}"
        slope_col = f"{col}_slope{w}"

        df[mean_col] = df[col].rolling(window=w, min_periods=w).mean()
        df[std_col] = df[col].rolling(window=w, min_periods=w).std()
        df[slope_col] = rolling_slope(df[col], w)

        new_cols.extend([mean_col, std_col, slope_col])

    return df, new_cols


def build_case_temporal_features(case_df):
    """
    Build temporal rolling features for a single case dataframe.

    Returns
    -------
    df : dataframe with added features
    new_columns : list of newly created feature names
    """

    df = case_df.copy()

    if "Time" in df.columns:
        df = df.sort_values("Time")

    signals = [
        # EEG
        "EEG1_delta_rel",
        "EEG1_theta_rel",
        "EEG1_alpha_rel",
        "EEG1_beta_rel",
        "EEG1_gamma_rel",

        # BIS derived
        "BIS/SR",
        "BIS/SEF",
        "BIS/EMG",
        "BIS/TOTPOW",

        # Vitals
        "Solar8000/PLETH_HR",
        "Solar8000/ART_MBP",
        "Solar8000/ETCO2",
        "Solar8000/BT",
    ]

    created_columns = []

    for col in signals:
        if col in df.columns:
            df, new_cols = build_temporal_features_for_column(
                df,
                col,
                windows=(30, 60),
            )
            created_columns.extend(new_cols)

    created_columns = sorted(set(created_columns))

    return df, created_columns

def plot_sanity_case(df, new_cols, seconds=600):
    """
    Plot original signals and their temporal features for sanity checking.
    """

    signals = [
        "EEG1_delta_rel",
        "EEG1_theta_rel",
        "EEG1_alpha_rel",
        "EEG1_beta_rel",
        "EEG1_gamma_rel",
        "BIS/SR",
        "BIS/SEF",
        "BIS/EMG",
        "Solar8000/HR",
        "Solar8000/ART_MBP",
        "Solar8000/ETCO2",
        "Solar8000/BT",
    ]

    # Limit to first N seconds to keep plots readable
    df = df.iloc[:seconds]

    for sig in signals:

        if sig not in df.columns:
            continue

        plt.figure(figsize=(12,5))

        # plot raw signal
        plt.plot(df[sig], label=sig, linewidth=2)

        # plot derived features
        derived = [c for c in new_cols if c.startswith(sig)]

        for col in derived:
            plt.plot(df[col], alpha=0.8, label=col)

        plt.title(sig)
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

	case_ids = collect_case_ids()
	n = len(case_ids)
	i = 0
	for cid in case_ids:	
		i += 1
		df, fpath = load_case_parquet(cid)
		print(f"df size: {df.shape[0]} rows × {df.shape[1]} columns")
		df, new_cols = build_case_temporal_features(df)
		print(fpath)
		print(f"df size: {df.shape[0]} rows × {df.shape[1]} columns")
		df.to_parquet(fpath, index=False)
		print(f"case {i}/{n} processed")
	print("New temporal feature columns:")
	for c in new_cols:
		print(c)
	
"""
	df, fpath = load_case_parquet(101)
	print(f"df size: {df.shape[0]} rows × {df.shape[1]} columns")
	df, new_cols = build_case_temporal_features(df)
	print(f"df size: {df.shape[0]} rows × {df.shape[1]} columns")
	plot_sanity_case(df, new_cols)

"""
