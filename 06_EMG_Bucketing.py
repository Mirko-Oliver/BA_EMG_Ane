"""
This script labels regions of the data as 'EMG Irregular'.

It does this by bucketing the data into BIS levels, and for each BIS level
it labels the top X% of EMG values as irregular.

The initial EMG-based irregular selection is only performed in the middle
phase of the case, defined as:
    ane_intro_end < Time < ane_end_start

After that:
- irregular labels are extended forward in time
- short regular runs between irregular regions are relabeled as irregular

The extension and fill logic are allowed to operate across the entire case.
"""

import os
import csv
import math
import numpy as np
import pandas as pd

DATA_DIR = "data"
SUFFIX = "_rawdata.parquet"
CLINICAL_INFO = "__clinical_info.csv"

TIME_COL = "Time"
EMG_COL = "BIS/EMG"
BIS_COL = "BIS/BIS"
SQI_COL = "BIS/SQI"

LABEL_COL = "label"

LABEL_INVALID = "invalid"
LABEL_IRREGULAR = "irregular"
LABEL_REGULAR = "regular"

# Config
SQI_THRESHOLD = 50
BIS_BUCKET_SIZE = 5
TOP_PERCENT_PER_BUCKET = 15

EXTEND_SEC = 5
MIN_REGULAR_RUN_SEC = 10


def discover_case_ids(data_dir=DATA_DIR, suffix=SUFFIX):
    return sorted(
        int(f.split("_")[0])
        for f in os.listdir(data_dir)
        if f.endswith(suffix)
    )


def collect_case_phase_info():
    """
    Load ane_intro_end and ane_end_start from __clinical_info.csv.
    """
    case_info = {}
    file_path = os.path.join(DATA_DIR, CLINICAL_INFO)

    with open(file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            cid = row.get("caseid")
            intro = row.get("ane_intro_end")
            end = row.get("ane_end_start")

            if cid is None or cid == "":
                continue
            if intro is None or intro == "":
                continue
            if end is None or end == "":
                continue

            case_info[str(cid)] = {
                "ane_intro_end": float(intro),
                "ane_end_start": float(end),
            }

    return case_info


def build_bis_buckets(bucket_size):
    """
    Returns bucket definitions as list 
    (1, 5, '1-5'), (6, 10, '6-10'), ...
    """
    buckets = []
    start = 1
    while start <= 100:
        end = min(start + bucket_size - 1, 100)
        buckets.append((start, end, f"{start}-{end}"))
        start += bucket_size
    return buckets


def print_bucket_info(bucket_size):
    buckets = build_bis_buckets(bucket_size)
    print("Created BIS buckets:")
    for start, end, label in buckets:
        print(f"  {label} (edges: {start} -> {end})")
    print()


def assign_bis_buckets(bis_values, bucket_size):
    """
    Assign each BIS value to a bucket label.
    Values outside [1, 100] or NaN get None.
    """
    buckets = build_bis_buckets(bucket_size)
    out = np.full(len(bis_values), None, dtype=object)

    finite_mask = np.isfinite(bis_values)
    in_range_mask = finite_mask & (bis_values >= 1) & (bis_values <= 100)

    for start, end, label in buckets:
        mask = in_range_mask & (bis_values >= start) & (bis_values <= end)
        out[mask] = label

    return out


def extend_mask_forward(t, mask, seconds=EXTEND_SEC):
    """
    For every True at index i, set subsequent indices True while t[j] <= t[i] + seconds.
    Assumes t is sorted ascending.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return mask

    seconds = float(seconds)
    if seconds <= 0:
        return mask

    out = mask.copy()
    true_idx = np.flatnonzero(mask)
    if true_idx.size == 0:
        return out

    n = len(t)
    for i in true_idx:
        if not np.isfinite(t[i]):
            continue
        t_end = t[i] + seconds
        j = i + 1
        while j < n and np.isfinite(t[j]) and t[j] <= t_end:
            out[j] = True
            j += 1

    return out


def relabel_short_regular_runs(labels, min_len=MIN_REGULAR_RUN_SEC):
    """
    Relabel short regular runs to irregular, but only if they are sandwiched
    between irregular regions. Invalid regions are not bridged.
    """
    labels = labels.copy()
    n = len(labels)
    i = 0

    while i < n:
        if labels[i] != LABEL_REGULAR:
            i += 1
            continue

        start = i
        while i < n and labels[i] == LABEL_REGULAR:
            i += 1
        end = i  # exclusive

        run_len = end - start
        left_label = labels[start - 1] if start > 0 else None
        right_label = labels[end] if end < n else None

        if (
            run_len < min_len
            and left_label == LABEL_IRREGULAR
            and right_label == LABEL_IRREGULAR
        ):
            labels[start:end] = LABEL_IRREGULAR

    return labels


def select_top_emg_per_bucket(emg, bis_bucket, is_invalid, in_middle_phase, top_percent):
    """
    Mark the top X% of datapoints in each BIS bucket as irregular,
    based on the highest EMG values.

    Initial selection is restricted to the middle phase only.
    """
    irregular = np.zeros(len(emg), dtype=bool)

    valid_candidates = (
        (~is_invalid) &
        in_middle_phase &
        np.isfinite(emg) &
        pd.notna(bis_bucket)
    )

    if not np.any(valid_candidates) or top_percent == 0:
        return irregular

    unique_buckets = [b for b in pd.unique(bis_bucket) if b is not None]

    for bucket in unique_buckets:
        bucket_mask = valid_candidates & (bis_bucket == bucket)
        bucket_idx = np.flatnonzero(bucket_mask)

        if bucket_idx.size == 0:
            continue

        n_select = int(math.ceil(bucket_idx.size * (top_percent / 100.0)))
        if n_select <= 0:
            continue

        bucket_emg = emg[bucket_idx]
        top_local_order = np.argsort(bucket_emg)[-n_select:]
        selected_idx = bucket_idx[top_local_order]
        irregular[selected_idx] = True

    return irregular


def main():
    print_bucket_info(BIS_BUCKET_SIZE)

    case_phase_info = collect_case_phase_info()
    case_ids = discover_case_ids(DATA_DIR)
    total_cases = len(case_ids)

    print(f"Found {total_cases} case(s) in '{DATA_DIR}'.")
    print()

    for i, cid in enumerate(case_ids, start=1):
        print(f"[{i}/{total_cases}] Processing case {cid}...")

        cid_str = str(cid)

        ane_intro_end = case_phase_info[cid_str]["ane_intro_end"]
        ane_end_start = case_phase_info[cid_str]["ane_end_start"]

        path = os.path.join(DATA_DIR, f"{cid}{SUFFIX}")
        df = pd.read_parquet(path)

        t = pd.to_numeric(df[TIME_COL], errors="coerce").to_numpy(dtype=float)
        emg = pd.to_numeric(df[EMG_COL], errors="coerce").to_numpy(dtype=float)
        bis = pd.to_numeric(df[BIS_COL], errors="coerce").to_numpy(dtype=float)
        sqi = pd.to_numeric(df[SQI_COL], errors="coerce").to_numpy(dtype=float)

        bis_bucket = assign_bis_buckets(bis, BIS_BUCKET_SIZE)
        labels = np.full(len(df), LABEL_REGULAR, dtype=object)

        # Invalid means BIS == 0 or NaN, or SQI < threshold or NaN
        is_invalid = (
            np.isnan(bis) |
            np.isclose(bis, 0.0, atol=1e-9) |
            np.isnan(sqi) |
            (sqi < SQI_THRESHOLD)
        )
        labels[is_invalid] = LABEL_INVALID

        # ane_intro_end < Time < ane_end_start
        in_middle_phase = (
            np.isfinite(t) &
            (t > ane_intro_end) &
            (t < ane_end_start)
        )

        # Initial irregular selection 
        is_irregular = select_top_emg_per_bucket(
            emg=emg,
            bis_bucket=bis_bucket,
            is_invalid=is_invalid,
            in_middle_phase=in_middle_phase,
            top_percent=TOP_PERCENT_PER_BUCKET,
        )

        # Extend across the entire case never into invalid rows
        is_irregular = extend_mask_forward(t, is_irregular, EXTEND_SEC)
        is_irregular &= ~is_invalid
        labels[is_irregular] = LABEL_IRREGULAR

        # Fill short regular gaps across the entire case
        labels = relabel_short_regular_runs(labels, MIN_REGULAR_RUN_SEC)

        df[LABEL_COL] = labels
        df.to_parquet(path, index=False)

        n_invalid = int(np.sum(labels == LABEL_INVALID))
        n_irregular = int(np.sum(labels == LABEL_IRREGULAR))
        n_regular = int(np.sum(labels == LABEL_REGULAR))



if __name__ == "__main__":
    main()
