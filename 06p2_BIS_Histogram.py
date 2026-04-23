"""
BIS/BIS distribution across ALL cases plotted in a stacked histogram by label.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = "data"
SUFFIX = "_rawdata.parquet"

BIS_COL = "BIS/BIS"
LABEL_COL = "label"

LABEL_REGULAR = "regular"
LABEL_IRREGULAR = "irregular"

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 14
})

def discover_case_ids(data_dir=DATA_DIR, suffix=SUFFIX):
    return sorted(
        int(f.split("_")[0])
        for f in os.listdir(data_dir)
        if f.endswith(suffix)
    )
    
def load_bis_and_label(data_dir=DATA_DIR):
	"""Return DF with BIS and LABEL Column for all cases"""
	case_ids = discover_case_ids()
	dfs = []
	for case_id in case_ids:
		file_path = os.path.join(data_dir, f"{case_id}{SUFFIX}")
		df = pd.read_parquet(file_path, columns=[BIS_COL, LABEL_COL])   
		dfs.append(df)
	combined_df = pd.concat(dfs, ignore_index=True)    
	return combined_df

def compute_hist(df):
	# counts[label, bis_value]
	counts_regular = np.zeros(101, dtype=np.int64)
	counts_irregular = np.zeros(101, dtype=np.int64)
	valid = df[BIS_COL].notna()
	bis_values = df.loc[valid, BIS_COL].astype(int)
	# Masks
	mask_regular = df[LABEL_COL] == LABEL_REGULAR
	mask_irregular = df[LABEL_COL] == LABEL_IRREGULAR

	counts_regular += np.bincount(
		bis_values[mask_regular], minlength=101
	)
	counts_irregular += np.bincount(
		bis_values[mask_irregular], minlength=101
	)

	return counts_regular, counts_irregular
   
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
		 
def main():
	df= load_bis_and_label()
	counts_regular, counts_irregular = compute_hist(df)
	
	total_regular = int(counts_regular.sum())
	total_irregular = int(counts_irregular.sum())
	total_all = total_regular + total_irregular
	
	counts_regular = counts_regular / total_all
	counts_irregular = counts_irregular / total_all
	
	# Plot stacked bars per BIS value
	x = np.arange(101, dtype=int)

	fig, ax = plt.subplots(figsize=(10, 3.5))
	
	COLOR_REGULAR = "black"   
	COLOR_IRREGULAR = "#9FB6C4" 

	ax.bar(x, counts_regular, width=1.0, label="EMG Regulär", color=COLOR_REGULAR)
	ax.bar(x, counts_irregular, width=1.0, bottom=counts_regular,
		   label="EMG Irregulär", color=COLOR_IRREGULAR)
	ax.set_xlabel("Bispektralindex ($BIS$)")
	ax.set_ylabel("Häufigkeit ($h$)")
	ax.set_xlim(0, 100)
	ax.set_ylim(bottom=0)
	
	ax.grid(True, linewidth=1, color='black', alpha=1.0)
	ax.spines['left'].set_linewidth(1.5)
	ax.spines['bottom'].set_linewidth(1.5)
	ax.set_xticks(np.arange(0, 101, 20))


	replace_second_last_tick_with_unit(ax, axis="x", unit="--")
	replace_second_last_tick_with_unit(ax, axis="y", unit="--")
	
	def format_eu(n, width=10):
		return f"{n:>{width},}".replace(",", ".")
    
	textstr = (
		r"$\bf{Anzahl:}$" "\n"
		f"Regulär:   {format_eu(total_regular)}\n"
		f"Irregulär: {format_eu(total_irregular)}\n"
		f"Alle:      {format_eu(total_all)}"
	)

	ax.text(
		1.05, 0.65, textstr,   
		transform=ax.transAxes,
		ha='left',
		va='top',
		fontsize=14,
		linespacing=1.5,
		family='monospace'
	)
	
	ax.legend(loc='upper left', bbox_to_anchor=(1.025, 1))
	plt.tight_layout(rect=[0, 0, 0.85, 1])
	plt.savefig("06p1_bis_histogram.png", dpi=300, bbox_inches="tight")
	

	# Print totals

	print("Totals:")
	print(f"  regular:   {total_regular:,}")
	print(f"  irregular: {total_irregular:,}")
	print(f"  all:       {total_all:,}")
	
	plt.show()


if __name__ == "__main__":
    main()
