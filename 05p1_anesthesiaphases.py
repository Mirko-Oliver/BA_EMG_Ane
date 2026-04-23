import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BIS_COL = "BIS/BIS"
PPF_COL = "Orchestra/PPF20_RATE"
DATA_DIR = "data"
SUFFIX = "_rawdata.parquet"
CLINICAL_INFO = "__clinical_info.csv"

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 14
})


def get_phase_timestamps(case_id):
    """
	Lookup  (ane_intro_end, ane_end_start)
    """
    file_path = os.path.join(DATA_DIR, CLINICAL_INFO)

    with open(file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            if row.get("caseid") == str(case_id):
                intro_end = row.get("ane_intro_end")
                end_start = row.get("ane_end_start")

                return int(float(intro_end)), int(float(end_start))

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


def format_eu(n, width=10):
    if isinstance(n, (int, np.integer)):
        s = f"{int(n):,}".replace(",", ".")
    else:
        s = f"{float(n):.2f}".rstrip("0").rstrip(".")
        s = s.replace(".", ",")
    return f"{s:>{width}}"

    
def plot_single_case_bis_ppf(case_id):
	"""
	Plots BIS and Orchestra/PPF20_RATE against time for one case.
	Background is shaded for the three anesthesia phases:
	1) start -> ane_intro_end
	2) ane_intro_end -> ane_end_start
	3) ane_end_start -> end
	"""
	# Load clinical phase borders
	ane_intro_end, ane_end_start = get_phase_timestamps(case_id)

	# Load case data
	path = os.path.join(DATA_DIR, case_id + SUFFIX)
	if not os.path.exists(path):
		raise FileNotFoundError(f"Parquet file not found for case {case_id}: {path}")

	df = pd.read_parquet(path)
	required_cols = ["Time", BIS_COL, PPF_COL]
	missing = [c for c in required_cols if c not in df.columns]
	if missing:
		raise KeyError(f"Case {case_id} missing required columns: {missing}")

	# Data
	x_sec = df["Time"].to_numpy(dtype=np.float32)
	x_min = (x_sec - np.nanmin(x_sec)) / 60.0

	y_bis = df[BIS_COL].to_numpy(dtype=np.float32)
	y_ppf = df[PPF_COL].to_numpy(dtype=np.float32)

	rate = df[PPF_COL].to_numpy(dtype=float)
	rate = np.nan_to_num(rate, nan=0.0)

	nz_idx = np.where(rate > 0)[0]
	last_idx = nz_idx[-1]
	print("Last nonzero PPF:")
	print(f"index={last_idx}, time={x_sec[last_idx]:.0f}s, rate={rate[last_idx]}")

	# Convert phase borders to minutes relative to case start
	x0_sec = np.nanmin(x_sec)
	intro_end_min = (ane_intro_end - x0_sec) / 60.0
	end_start_min = (ane_end_start - x0_sec) / 60.0

	x_min_max = np.nanmax(x_min)

	fig, ax = plt.subplots(figsize=(12, 3.5))
	ax2 = ax.twinx()

	# Background phase shading
	ax.axvspan(0, intro_end_min, color="#DCEED1", alpha=0.7, zorder=0)
	ax.axvspan(intro_end_min, end_start_min, color="#D9C2B0", alpha=0.7, zorder=0)
	ax.axvspan(end_start_min, 140, color="#DCEAF7", alpha=0.7, zorder=0)

	# Lines
	ax.plot(x_min, y_bis, label="BIS", color="black", zorder=3)
	ax2.plot(x_min, y_ppf, label="Propofol", color="#990000", zorder=4)

	# Labels
	ax.set_xlabel("Operationszeit ($t_{OP}$)")
	ax.set_ylabel("Bispektralindex ($BIS$)")
	ax2.set_ylabel("Infusionsrate  ($R$)", color="#990000")

	# Limits
	ax.set_xlim(left=0)
	ax.set_xlim(right=140)
	ax.set_ylim(bottom=0, top=100)
	ax2.set_ylim(bottom=0, top=100)

	# Axis styling
	ax.grid(True, linewidth=1, color='black', alpha=1.0)
	ax.spines['left'].set_linewidth(1.5)
	ax.spines['bottom'].set_linewidth(1.5)
	ax2.spines['right'].set_linewidth(1.5)
	ax2.spines['right'].set_color("#990000")
	ax2.tick_params(axis='y', colors="#990000")

	replace_second_last_tick_with_unit(ax, axis="y", unit="--")
	replace_second_last_tick_with_unit(ax, axis="x", unit="min")
	replace_second_last_tick_with_unit(ax2, axis="y", unit="ml/h")

	# Text box
	def fmt_min(v):
		return f"{int(round(v))}min"

	label_w = 14
	range_w = 18

	textstr = (
		r"$\bf{Anästhesiephasen:}$" + "\n" +
		f"{'Einleitung:':<{label_w}}{f'{fmt_min(0)} - {fmt_min(intro_end_min)}':>{range_w}}\n"
		f"{'Narkose:':<{label_w}}{f'{fmt_min(intro_end_min)} - {fmt_min(end_start_min)}':>{range_w}}\n"
		f"{'Ausleitung:':<{label_w}}{f'{fmt_min(end_start_min)} - {fmt_min(x_min_max)}':>{range_w}}\n"
	)
	ax.text(
		1.25, 0.7, textstr,
		transform=ax.transAxes,
		ha='left',
		va='top',
		fontsize=12,
		linespacing=1.5,
		family='monospace'
	)

	case_str = f"{'Fallnummer:':<{label_w}}{format_eu(int(case_id)):>{range_w}}"
	ax.text(
		1.25, -0.1, case_str,
		transform=ax.transAxes,
		ha="left",          
		va="top",
		fontsize=12,      
		family='monospace'       
	)

	# Combined legend
	lines_1, labels_1 = ax.get_legend_handles_labels()
	lines_2, labels_2 = ax2.get_legend_handles_labels()
	ax.legend(
    lines_1 + lines_2,
    labels_1 + labels_2,
    loc='upper left',
    bbox_to_anchor=(1.25, 1),
    borderaxespad=0.0
)

	plt.tight_layout(rect=[0, 0, 0.85, 1])
	plt.savefig("06p1_anesthesiaphases.png", dpi=300, bbox_inches="tight")
	plt.show()


if __name__ == "__main__":
	
    plot_single_case_bis_ppf(case_id="4941")
	#4941
