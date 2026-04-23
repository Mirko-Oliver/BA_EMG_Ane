"""
This script imports the EEG1 track from VitalDB and calculates:
- relative bandpowers for EEG1
- relative beta ratio (RBR) for EEG1

for any cases with existing parquet files in the data folder.

It appends these features to each parquet file and finally applies a
trailing 5-sample median smoothing directly in-place for:
- all EEG1 relative bandpowers
- EEG1 RBR
- BIS/EMG
- BIS/SEF
- BIS/TOTPOW
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import time
from pathlib import Path
from multiprocessing import get_context

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import vitaldb


DATA_DIR = "data"

# Epoching: 2 s epochs, 75% overlap
EPOCH_LEN_S = 2.0
EPOCH_HOP_S = 0.5

# Smoothing window for PSD-based features, trailing 30 s
PSD_SMOOTH_S = 30.0

# Final trailing median smoothing in samples (1 Hz grid => 5 samples = 5 seconds)
FINAL_MEDIAN_WINDOW = 5

# EEG sampling frequency
fs = 128

# EEG bands for relative band powers
BANDS = {
    "delta": (0.3, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta":  (13.0, 25.0),
    "gamma": (26.0, 50.0),
}

RBR_NUM_BAND = (30.0, 47.0)
RBR_DEN_BAND = (11.0, 20.0)

# Existing parquet columns to smooth too
EXTERNAL_SMOOTH_COLS = ["BIS/EMG", "BIS/SEF", "BIS/TOTPOW"]


def collect_case_ids(data_dir=DATA_DIR):
    return sorted(
        int(fname.split("_")[0])
        for fname in os.listdir(data_dir)
        if fname.endswith("_rawdata.parquet")
    )


def load_eeg1_channel(case_id, fs=fs):
    """
    Load raw EEG1 waveform for a given case from VitalDB.
    """
    track = "BIS/EEG1_WAV"
    vf = vitaldb.VitalFile(case_id, [track])
    df = vf.to_pandas([track], 1.0 / fs)

    if track in df.columns:
        x = pd.to_numeric(df[track], errors="coerce").to_numpy()
        return x[np.isfinite(x)]

    return np.array([], dtype=float)


def detrend_savgol(x, win_s=0.5, poly=3):
    """
    Remove slow trend from EEG using a Savitzky–Golay filter.
    """
    win_len = int(round(win_s * fs))
    win_len = max(5, win_len | 1)  # ensure odd length, >= 5
    trend = savgol_filter(x, window_length=win_len, polyorder=poly, mode="interp")
    return x - trend


def make_epochs(x, epoch_s=EPOCH_LEN_S, hop_s=EPOCH_HOP_S):
    """
    Cut the continuous signal into overlapping epochs.

    Returns:
        array of shape (n_epochs, epoch_len_samples)
    """
    L = int(round(epoch_s * fs))
    H = int(round(hop_s * fs))

    if L <= 0 or H <= 0 or len(x) < L:
        return np.empty((0, L), dtype=float)

    starts = np.arange(0, len(x) - L + 1, H, dtype=int)
    return np.stack([x[s:s + L] for s in starts], axis=0)


def epoch_centers_seconds(n_epochs, epoch_s=EPOCH_LEN_S, hop_s=EPOCH_HOP_S):
    """
    Compute center time of each epoch.
    """
    return (np.arange(n_epochs) * hop_s) + (epoch_s / 2.0)


def compute_epoch_spectra(epochs):
    """
    Compute per-epoch FFT and PSD proxy.
    """
    if epochs.size == 0:
        return None, None, None

    L = epochs.shape[1]
    w = np.blackman(L)[None, :]
    X = np.fft.rfft(epochs * w, n=L, axis=1)
    freqs = np.fft.rfftfreq(L, d=1.0 / fs)
    P = (np.abs(X) ** 2) / (np.sum(np.abs(w) ** 2))

    return freqs, X, P


def band_power(freqs, Pavg, band):
    """
    Integrate spectral power over a frequency band.
    """
    lo, hi = band
    m = (freqs >= lo) & (freqs <= hi)
    if not np.any(m):
        return np.nan
    return np.trapz(Pavg[m], freqs[m])


def rolling_psd_features_trailing(
    x,
    prefix="EEG1",
    epoch_s=EPOCH_LEN_S,
    hop_s=EPOCH_HOP_S,
    smooth_s=PSD_SMOOTH_S,
):
    """
    Compute trailing PSD-based EEG1 features on an integer-second grid.

    Returns a DataFrame indexed by time_sec with columns:
    - EEG1_delta_rel to gamma
    - EEG1_rbr
    """
    epochs = make_epochs(x, epoch_s=epoch_s, hop_s=hop_s)
    if epochs.size == 0:
        return pd.DataFrame()

    freqs, _, P = compute_epoch_spectra(epochs)
    centers = epoch_centers_seconds(len(epochs), epoch_s=epoch_s, hop_s=hop_s)

    t_grid = np.arange(int(np.floor(centers[-1])) + 1, dtype=int)

    data = {
        f"{prefix}_delta_rel": np.full_like(t_grid, np.nan, dtype=float),
        f"{prefix}_theta_rel": np.full_like(t_grid, np.nan, dtype=float),
        f"{prefix}_alpha_rel": np.full_like(t_grid, np.nan, dtype=float),
        f"{prefix}_beta_rel":  np.full_like(t_grid, np.nan, dtype=float),
        f"{prefix}_gamma_rel": np.full_like(t_grid, np.nan, dtype=float),
        f"{prefix}_rbr": np.full_like(t_grid, np.nan, dtype=float),
    }

    for k, t in enumerate(t_grid):
        t_end = float(t)
        t_start = t_end - smooth_s
        use = (centers > t_start) & (centers <= t_end)

        if not np.any(use):
            continue

        Pavg = np.nanmean(P[use, :], axis=0)

        delta = band_power(freqs, Pavg, BANDS["delta"])
        theta = band_power(freqs, Pavg, BANDS["theta"])
        alpha = band_power(freqs, Pavg, BANDS["alpha"])
        beta  = band_power(freqs, Pavg, BANDS["beta"])
        gamma = band_power(freqs, Pavg, BANDS["gamma"])

        total = np.nansum([delta, theta, alpha, beta, gamma])
        if not (np.isfinite(total) and total > 0):
            continue

        p_num = band_power(freqs, Pavg, RBR_NUM_BAND)
        p_den = band_power(freqs, Pavg, RBR_DEN_BAND)

        if np.isfinite(p_num) and np.isfinite(p_den) and (p_num > 0) and (p_den > 0):
            data[f"{prefix}_rbr"][k] = np.log10(p_num / p_den)

        data[f"{prefix}_delta_rel"][k] = delta / total if np.isfinite(delta) else np.nan
        data[f"{prefix}_theta_rel"][k] = theta / total if np.isfinite(theta) else np.nan
        data[f"{prefix}_alpha_rel"][k] = alpha / total if np.isfinite(alpha) else np.nan
        data[f"{prefix}_beta_rel"][k]  = beta  / total if np.isfinite(beta)  else np.nan
        data[f"{prefix}_gamma_rel"][k] = gamma / total if np.isfinite(gamma) else np.nan

    out = pd.DataFrame(data, index=t_grid)
    out.index.name = "time_sec"
    return out


def get_smoothing_target_columns(df):
    """
    Collect all columns that should be overwritten by final trailing median smoothing.
    """
    eeg1_cols = [
        c for c in df.columns
        if c.startswith("EEG1_") and (c.endswith("_rel") or c.endswith("_rbr"))
    ]

    external_cols = [c for c in EXTERNAL_SMOOTH_COLS if c in df.columns]

    return sorted(set(eeg1_cols + external_cols))


def trailing_median(df, window=FINAL_MEDIAN_WINDOW):
    """
    Overwrite target columns using trailing median smoothing.
    """
    df = df.copy()

    target_cols = get_smoothing_target_columns(df)
    if not target_cols:
        return df

    df[target_cols] = (
        df[target_cols]
        .apply(pd.to_numeric, errors="coerce")
        .rolling(window=window, min_periods=1)
        .median()
    )

    return df


def process_case(case_id):
    """
    Process one case:
    1. Load EEG1 from VitalDB
    2. Detrend EEG1
    3. Compute EEG1 relative bandpowers + RBR
    4. Join features into existing parquet
    5. write target columns with trailing 5-sample median smoothing
    """
    t0 = time.perf_counter()

    try:
        x1_raw = load_eeg1_channel(case_id)

        if x1_raw.size == 0:
            return case_id, "No EEG1 samples found"

        x1 = detrend_savgol(x1_raw, win_s=0.5, poly=3)
        feats1 = rolling_psd_features_trailing(x1, prefix="EEG1")

        if feats1.empty:
            return case_id, "No EEG1 features computed"

        parquet_path = Path(DATA_DIR) / f"{case_id}_rawdata.parquet"
        if not parquet_path.exists():
            return case_id, f"Missing parquet file: {parquet_path.name}"

        df_existing = pd.read_parquet(parquet_path)

        time_existing = pd.to_numeric(df_existing["Time"], errors="coerce")
        if time_existing.isna().all():
            return case_id, "Existing parquet Time column could not be parsed"

        df_existing = df_existing.copy()
        df_existing["time_sec"] = np.round(time_existing).astype("int64")
        df_existing = df_existing.set_index("time_sec", drop=True)

        feats_wide = feats1.copy()
        feats_wide.index = feats_wide.index.astype("int64")
        feats_wide.index.name = "time_sec"

        overlap_cols = df_existing.columns.intersection(feats_wide.columns)
        if len(overlap_cols) > 0:
            df_existing = df_existing.drop(columns=overlap_cols)

        df_out = df_existing.join(feats_wide, how="left")
        df_out = df_out.sort_index().reset_index(drop=True)

        if "Time" in df_out.columns:
            cols = list(df_out.columns)
            cols.remove("Time")
            df_out = df_out[["Time"] + cols]
        else:
            return case_id, "Time column disappeared unexpectedly after join"

        df_out = trailing_median(
            df_out,
            window=FINAL_MEDIAN_WINDOW,
        )

        df_out.to_parquet(parquet_path)

        dt = time.perf_counter() - t0
        return case_id, f"OK ({dt:.1f}s → EEG1 features appended, target cols overwritten with trailing median)"

    except Exception as e:
        return case_id, f"ERROR: {e}"


def main():
    """
    - Discover case IDs from '{caseid}_rawdata.parquet'
    - Run process_case(case_id) in parallel
    """
    case_ids = collect_case_ids()

    if len(case_ids) == 0:
        print("[INFO] No parquet cases found.")
        return

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    n_procs = min(os.cpu_count() or 1, len(case_ids))
    print(f"[INFO] Starting pool with {n_procs} processes")

    t0 = time.perf_counter()
    with get_context("spawn").Pool(processes=n_procs) as pool:
        for case_id, msg in pool.imap_unordered(process_case, case_ids, chunksize=1):
            print(f"[CASE {case_id}] {msg}")

    dt = time.perf_counter() - t0
    print(f"[INFO] Finished all cases in {dt:.1f}s")


if __name__ == "__main__":
    main()
