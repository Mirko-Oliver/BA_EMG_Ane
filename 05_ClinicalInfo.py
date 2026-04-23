import pandas as pd
import numpy as np
import os
#BEFORE RUNNING SCRIPT: manualy download clinical info csv from vitalbd,
"""
this script:
- filters any caseids out of the clinical info csv that are not part of the dataset
- calculates lbm
- determines amnesthisa phase borders
"""
DATA_DIR = "data"
SUFFIX = "_rawdata.parquet"
CLINICAL_INFO = "__clinical_info.csv"


clinical_path = os.path.join(DATA_DIR, CLINICAL_INFO)
df = pd.read_csv(clinical_path)


case_ids = {
    int(f.split("_")[0])
    for f in os.listdir(DATA_DIR)
    if f.endswith(SUFFIX)
}
print(len(case_ids))

def add_lbm(df, weight_col="weight", height_col="height", sex_col="sex", out_col="lbm"):
    """
    Adds a lean body mass (lbm) column to the dataframe.
        Male (M):   1.0 * weight - 128 * (weight / height)^2
        Female (F): 1.07 * weight - 148 * (weight / height)^2

    """
    df = df.copy()
    weight = pd.to_numeric(df[weight_col], errors="coerce")
    height = pd.to_numeric(df[height_col], errors="coerce")

    ratio_sq = (weight / height) ** 2

    sex = df[sex_col].astype(str).str.strip().str.upper()

    # Compute LBM
    df[out_col] = np.where(
        sex == "M",
        1.0 * weight - (128 * ratio_sq),
        np.where(
            sex == "F",
            1.07 * weight - (148 * ratio_sq),
            np.nan  
        )
    )

    return df

import numpy as np


import numpy as np


def find_anesthesia_borders(df):
	"""
	Ane Borders defined as
	Intro Min: 10min, then first time running Mean lower than 40ml/hr
	End : From the end last time running Mean under 5ml/hr
	"""
    rate_col = "Orchestra/PPF20_RATE"
    time_col = "Time"

    intro_threshold = 40.0   
    end_threshold = 5.0      
    min_intro_s = 15 * 60    
    window_s = 10 * 60       

    time_s = df[time_col].to_numpy(dtype=float)
    rate = df[rate_col].to_numpy(dtype=float)
    rate = np.nan_to_num(rate, nan=0.0, posinf=0.0, neginf=0.0)

    n = len(rate)
    if n == 0:
        return np.nan, np.nan

    # cumulative sum so sum(rate[i:j]) = csum[j] - csum[i]
    csum = np.concatenate(([0.0], np.cumsum(rate, dtype=float)))

    def forward_mean(i):
        """
        Mean over the forward-looking window [i, i+window_s).
        Clipped at the dataset end.
        """
        j = min(i + window_s, n)
        count = j - i
        if count <= 0:
            return np.nan
        return (csum[j] - csum[i]) / count

    # Compute forward-looking rolling mean for every timestamp
    fwd_mean = np.empty(n, dtype=float)
    for i in range(n):
        fwd_mean[i] = forward_mean(i)

    t0 = time_s[0]

    # intro_end:
    # first timestamp t_e >= t0 + 15min where forward mean < 40
    # return t_e + 600
    intro_end = np.nan
    intro_candidates = np.where(
        (time_s >= t0 + min_intro_s) & (fwd_mean < intro_threshold)
    )[0]
    if len(intro_candidates) > 0:
        i = intro_candidates[0]
        intro_end = time_s[i] + window_s

    # end_start:
    # first timestamp from the back where forward mean > 5
    end_start = np.nan
    end_candidates = np.where(fwd_mean > end_threshold)[0]
    if len(end_candidates) > 0:
        i = end_candidates[-1]
        end_start = time_s[i]

    return intro_end, end_start
    
def anesthesia_borders(df, case_ids):
	df["ane_intro_end"] = np.nan
	df["ane_end_start"] = np.nan
	for cid in case_ids:
		path = os.path.join(DATA_DIR, f"{cid}{SUFFIX}")
		case_df = pd.read_parquet(path)
		intro_end, end_start = find_anesthesia_borders(case_df)
		mask = df["caseid"] == cid
		df.loc[mask, "ane_intro_end"] = intro_end
		df.loc[mask, "ane_end_start"] = end_start
	return df

df["caseid"] = pd.to_numeric(df["caseid"], errors="coerce").astype("Int64")
df_filtered = df[df['caseid'].isin(case_ids)]
df_filtered = add_lbm(df_filtered) # ADD LBM Column
df_filtered = anesthesia_borders(df_filtered, case_ids)

df_filtered.to_csv(clinical_path, index=False)

print(f"Original rows: {len(df)}")
print(f"Remaining rows: {len(df_filtered)}")


