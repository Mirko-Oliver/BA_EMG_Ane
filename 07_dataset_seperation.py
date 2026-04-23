"""
This Script Divides the Dataset into 3 Subsets (Training, Validation and Testing)
It labels each case in the clinical info csv 
each Subset has an equal M/F Split
"""
import os
import pandas as pd


DATA_DIR = "data"
CLINICAL_INFO = "__clinical_info.csv"

# Set split sizes 
SPLIT_RATIOS = {
    "Training": 0.70,
    "Validation": 0.15,
    "Testing": 0.15,
}

RANDOM_SEED = 42  

clinical_path = os.path.join(DATA_DIR, CLINICAL_INFO)
df = pd.read_csv(clinical_path)

df["sex"] = (df["sex"].astype(str))



def split_counts(n: int, ratios: dict) -> dict:
    """Return integer counts per split that sum exactly to n."""
    raw = {k: n * r for k, r in ratios.items()}
    base = {k: int(v) for k, v in raw.items()}  # floor
    remainder = n - sum(base.values())

    # Distribute remainder to splits with largest fractional parts
    frac_order = sorted(raw.keys(), key=lambda k: (raw[k] - base[k]), reverse=True)
    for i in range(remainder):
        base[frac_order[i % len(frac_order)]] += 1
    return base

rat_sum = sum(SPLIT_RATIOS.values())
if abs(rat_sum - 1.0) > 1e-9:
    raise ValueError(f"SPLIT_RATIOS must sum to 1.0; got {rat_sum}")

counts_by_sex = df["sex"].value_counts()
n_m = int(counts_by_sex.get("M", 0))
n_f = int(counts_by_sex.get("F", 0))

m_counts = split_counts(n_m, SPLIT_RATIOS)
f_counts = split_counts(n_f, SPLIT_RATIOS)

# Assign Dataset labels with stratification
df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
df["Dataset"] = pd.NA

def assign_for_sex(sex_value: str, per_split_counts: dict):
    idx = df.index[df["sex"] == sex_value].tolist()
    start = 0
    for split_name in SPLIT_RATIOS.keys():
        take = per_split_counts[split_name]
        chosen = idx[start:start + take]
        df.loc[chosen, "Dataset"] = split_name
        start += take

assign_for_sex("M", m_counts)
assign_for_sex("F", f_counts)





df.to_csv(clinical_path, index=False)


print("\nOverall counts by Dataset:")
print(df["Dataset"].value_counts(dropna=False).to_string())

print("\nSex distribution within each Dataset (M/F only):")
report = (
    df[df["sex"].isin(["M", "F"])]
    .groupby(["Dataset", "sex"])
    .size()
    .unstack(fill_value=0)
)
report["Total"] = report.sum(axis=1)
print(report.to_string())

print("\nOverall sex counts:")
print(df[df["sex"].isin(["M", "F"])]["sex"].value_counts().to_string())
