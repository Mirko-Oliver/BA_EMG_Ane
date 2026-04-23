"""
Sanity check script:

- Loads relative bandpowers for EEG1 and BIS
- Plots BIS on the left axis
- Plots stacked relative bandpowers on the right axis
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_DIR = Path("data")
CASE_ID = 5934

BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
EEG_PREFIX = "EEG1"
BIS_COL = "BIS/BIS"

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 12
})

BAND_COLORS = {
    "delta": "#9FB6C4",
    "theta": "#005e9b",
    "alpha": "#bea6a0",
    "beta":  "#7D6666",
    "gamma": "#990000",
}

def load_case_parquet(case_id, data_dir=DATA_DIR):
    """
    Load the parquet file for a given case ID.
    """
    path = data_dir / f"{case_id}_rawdata.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    return pd.read_parquet(path)


def extract_bandpowers_bis(df):
    """
    Extract EEG1 relative bandpowers and BIS from the case file.
    """
    t = pd.to_numeric(df["Time"], errors="coerce")
    out = df.copy()
    out.index = t.astype(float).to_numpy()
    out.index.name = "Time"

    band_cols = [f"{EEG_PREFIX}_{b}_rel" for b in BANDS]
    existing_band_cols = [c for c in band_cols if c in out.columns]

    band_wide = out[existing_band_cols].copy() if existing_band_cols else pd.DataFrame(index=out.index)
    band_wide = band_wide.reindex(columns=band_cols)

    bis = pd.to_numeric(out[BIS_COL], errors="coerce")
    bis.name = "BIS"

    return band_wide, bis

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

def plot_eeg1_with_bis(case_id, band_wide, bis):
    fig, ax_left = plt.subplots(figsize=(12, 4.5))

    # LEFT axis: BIS
    if not bis.empty:
        t_bis = bis.index.to_numpy() / 60.0 
        y_bis = bis.to_numpy()
        ax_left.plot(
            t_bis,
            y_bis,
            color="black",
            linewidth=1.0,
            label="BIS",
        )

    ax_left.set_ylim(0, 100)
    ax_left.set_xlim(0, 200)  

    ax_left.set_ylabel("BIS")
    ax_left.set_xlabel("Time (min)")
    ax_left.grid(True, alpha=0.3)

    #RIGHT axis: bandpowers
    ax_right = ax_left.twinx()

    t_band = band_wide.index.to_numpy() / 60.0  
    y_stacks = []
    labels = []
    colors = []

    for band in BANDS:
        col = f"{EEG_PREFIX}_{band}_rel"
        if col not in band_wide.columns:
            y = np.zeros_like(t_band, dtype=float)
        else:
            y = band_wide[col].fillna(0.0).to_numpy()

        y_stacks.append(y)
        labels.append(band)
        colors.append(BAND_COLORS.get(band, None))

    ax_right.stackplot(
        t_band,
        *y_stacks,
        labels=labels,
        colors=colors,    
        baseline="zero",
    )

    ax_right.set_ylabel("Relative bandpower")
    ax_right.set_ylim(0.0, 1)

    # ---- Legend ----
    left_handles, left_labels = ax_left.get_legend_handles_labels()
    right_handles, right_labels = ax_right.get_legend_handles_labels()

    ax_left.legend(
        left_handles + right_handles,
        left_labels + right_labels,
        loc="upper left",
        bbox_to_anchor=(1.025, 1),
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print(f"Using configured case: {CASE_ID}")
    df = load_case_parquet(CASE_ID, DATA_DIR)
    band_wide, bis = extract_bandpowers_bis(df)
    plot_eeg1_with_bis(CASE_ID, band_wide, bis)
