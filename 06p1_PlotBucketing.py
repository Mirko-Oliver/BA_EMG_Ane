import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch

DATA_DIR = Path("data")
SUFFIX = "_rawdata.parquet"

CASE_ID = 5934

TIME_COL = "Time"
EMG_COL = "BIS/EMG"
DERIV_COL = "emg_derivative"
LABEL_COL = "label"

LABEL_IRREGULAR = "irregular"
LABEL_INVALID = "invalid"

SHADE_INVALID = True

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 14
})



def load_case_for_plot(case_id, data_dir=DATA_DIR):
    path = data_dir / f"{case_id}{SUFFIX}"
    df = pd.read_parquet(path)

    required = [TIME_COL, EMG_COL, DERIV_COL, LABEL_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Case {case_id}: missing columns {missing}")

    t = pd.to_numeric(df[TIME_COL], errors="coerce").to_numpy(dtype=float)
    emg = pd.to_numeric(df[EMG_COL], errors="coerce").to_numpy(dtype=float)
    deriv = pd.to_numeric(df[DERIV_COL], errors="coerce").to_numpy(dtype=float)
    labels = df[LABEL_COL].astype(str).to_numpy()

    m = np.isfinite(t)
    return t[m], emg[m], deriv[m], labels[m], path


def contiguous_true_segments(mask):
    mask = np.asarray(mask, dtype=bool)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []

    splits = np.where(np.diff(idx) > 1)[0] + 1
    segments = np.split(idx, splits)
    return [(int(seg[0]), int(seg[-1])) for seg in segments]


def shade_segments(ax, t, segments, color, alpha=0.2):
    for i0, i1 in segments:
        ax.axvspan(t[i0], t[i1], color=color, alpha=alpha)


def replace_second_last_tick_with_unit(ax, axis, unit):
    ax.tick_params(axis='both', which='both', length=0)

    if axis == "y":
        ticks = ax.get_yticks()
        labels = [f"{t:g}" for t in ticks]
        if len(labels) >= 2:
            labels[-2] = unit
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)

    elif axis == "x":
        ticks = ax.get_xticks()
        labels = [f"{t:g}" for t in ticks]
        if len(labels) >= 2:
            labels[-2] = unit
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)


def main():
	t, emg, deriv, labels, path = load_case_for_plot(CASE_ID, DATA_DIR)
	t = (t - t.min()) / 60.0

	irregular_mask = labels == LABEL_IRREGULAR
	invalid_mask = labels == LABEL_INVALID

	irr_segments = contiguous_true_segments(irregular_mask)
	inv_segments = contiguous_true_segments(invalid_mask)

	fig, ax = plt.subplots(figsize=(10, 3.5))

	# Shading
	COLOR_IRREGULAR = "#7d6666"
	COLOR_INVALID = "#990000"

	shade_segments(ax, t, irr_segments, color=COLOR_IRREGULAR, alpha=0.35)
	if SHADE_INVALID:
		shade_segments(ax, t, inv_segments, color=COLOR_INVALID, alpha=0.35)

	# Lines
	line_emg, = ax.plot(t, emg, linewidth=1.2, color="black", label=EMG_COL)

	ax.set_xlabel("Operationszeit ($t_{OP}$)")
	ax.set_ylabel("EMG Leistung ($EMG$)")
	ax.set_xlim(0, 200)
	ax.set_ylim(0, 80)

	ax.grid(True, linewidth=1, color='black', alpha=1.0)
	ax.spines['left'].set_linewidth(1.5)
	ax.spines['bottom'].set_linewidth(1.5)
	ax.spines['top'].set_linewidth(1.0)
	ax.spines['right'].set_linewidth(1.0)

	replace_second_last_tick_with_unit(ax, axis="x", unit="min")
	replace_second_last_tick_with_unit(ax, axis="y", unit="dB")

	n_irr = int(np.sum(irregular_mask))
	n_inv = int(np.sum(invalid_mask))
	n_all = len(labels)

	def format_eu(n, width=10):
		if isinstance(n, (int, np.integer)):
			s = f"{int(n):,}".replace(",", ".")
		else:
			s = f"{float(n):.2f}".rstrip("0").rstrip(".")
			s = s.replace(".", ",")
		return f"{s:>{width}}"

	textstr = (
		r"$\bf{Anzahl:}$" "\n"
		f"Irregulär: {format_eu(n_irr)}\n"
		f"Invalid:   {format_eu(n_inv)}\n"
		f"Alle:      {format_eu(n_all)}"
	)

	ax.text(
		1.05, 0.5,
		textstr,
		transform=ax.transAxes,
		ha='left',
		va='top',
		fontsize=14,
		linespacing=1.5,
		family='monospace'
	)

	ax.text(
		1.05, -0.1125, f"{'Fallnummer:':<11}{format_eu(int(CASE_ID))}",
		transform=ax.transAxes,
		ha="left",          
		va="top",      
		family='monospace'       
	)
	
	patch_irregular = Patch(facecolor=COLOR_IRREGULAR, alpha=0.35, label="Irregulär")
	patch_invalid = Patch(facecolor=COLOR_INVALID, alpha=0.35, label="Invalid")
	
	handles = [line_emg, patch_irregular]
	if SHADE_INVALID:
		handles.append(patch_invalid)

	ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.025, 1))
	plt.tight_layout(rect=[0, 0, 0.85, 1])
	plt.savefig("06p1_PlotBucketing.png", dpi=300, bbox_inches="tight")
	plt.show()


if __name__ == "__main__":
    main()
