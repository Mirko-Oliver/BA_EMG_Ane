"""
This Script Imports all Cases that have the required Tracks, don't have the excluded Tracks.
For those Cases all included Tracks are imported
The Data is imported Aligned at 1 second, during import a forwardfill of 1 takes place 
Each Row represents 1sec, each column 1 Track
"""

import os
import math
import concurrent.futures
from typing import Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd
import vitaldb

DATA_DIR = "data"

def get_caseids(required_tracks, excluded_tracks):
    """
    Returns CaseIds for cases that contain all required tracks and none of the excluded tracks.
    """
    caseids = set(vitaldb.find_cases(required_tracks))
    for track in excluded_tracks:
        caseids -= set(vitaldb.find_cases([track]))
    return sorted(caseids)


def load_case(caseid, included_tracks):
    """
    Loads a case and returns wide-format data indexed in 1 second intervalls.
    forwardfills single wide gaps
    """
    vf = vitaldb.VitalFile(caseid, included_tracks, interval=1)
    df = vf.to_pandas(included_tracks, 1, return_timestamp=True)
    df = df.ffill(limit = 1)
    return df


def _process_single_case(caseid, included_tracks, outdir):
    """
    Worker: load one case and saves it parquet.
    """
    try:
        df = load_case(caseid, included_tracks)
    except Exception as e:
        return caseid, False, f"Failed to load case {caseid}: {e}"

    opath = os.path.join(outdir, f"{caseid}_rawdata.parquet")

    try:
        df.to_parquet(opath, index=True)
    except Exception as e:
        return caseid, False, f"Failed to save case {caseid}: {e}"

    return caseid, True, f"Saved → {opath} (shape={df.shape})"


def save_cases(caselist, included_tracks, outdir = DATA_DIR, max_workers = 8):
	"""
	Parallel import
	"""
	os.makedirs(outdir, exist_ok=True)
	caselist = list(caselist)
	n = len(caselist)
	i = 0
	print(f"Starting parallel import of {n} cases...")
	tasks = []
	with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
		for caseid in caselist:
			tasks.append(
				executor.submit(
					_process_single_case,
					caseid,
					included_tracks,
					outdir
				)
			)

		for future in concurrent.futures.as_completed(tasks):
			caseid, ok, msg = future.result()
			i += 1
			if ok:
				print(f"Success in Case {caseid}: {msg}, Completion: {i}/{n} ")
			else:
				print(f"Error in Case {caseid}: {msg}, Completion: {i}/{n} ")


if __name__ == "__main__":
    included_tracks = [
        "Solar8000/ART_DBP",
        "Solar8000/ART_MBP",
        "Solar8000/ART_SBP",
        "Solar8000/BT",
        "Solar8000/CVP",
        "Solar8000/ETCO2",
        "Solar8000/FEM_DBP",
        "Solar8000/FEM_MBP",
        "Solar8000/FEM_SBP",
        "Solar8000/FEO2",
        "Solar8000/FIO2",
        "Solar8000/GAS2_EXPIRED",
        "Solar8000/GAS2_INSPIRED",
        "Solar8000/HR",
        "Solar8000/INCO2",
        "Solar8000/NIBP_DBP",
        "Solar8000/NIBP_MBP",
        "Solar8000/NIBP_SBP",
        "Solar8000/PA_DBP",
        "Solar8000/PA_MBP",
        "Solar8000/PA_SBP",
        "Solar8000/PLETH_HR",
        "Solar8000/PLETH_SPO2",
        "Solar8000/RR",
        "Solar8000/RR_CO2",
        "Solar8000/ST_AVF",
        "Solar8000/ST_AVL",
        "Solar8000/ST_AVR",
        "Solar8000/ST_I",
        "Solar8000/ST_II",
        "Solar8000/ST_III",
        "Solar8000/ST_V5",
        "Solar8000/VENT_COMPL",
        "Solar8000/VENT_INSP_TM",
        "Solar8000/VENT_MAWP",
        "Solar8000/VENT_MEAS_PEEP",
        "Solar8000/VENT_MV",
        "Solar8000/VENT_PIP",
        "Solar8000/VENT_PPLAT",
        "Solar8000/VENT_RR",
        "Solar8000/VENT_SET_FIO2",
        "Solar8000/VENT_SET_PCP",
        "Solar8000/VENT_SET_TV",
        "Solar8000/VENT_TV",
        "Primus/COMPLIANCE",
        "Primus/ETCO2",
        "Primus/FEN2O",
        "Primus/FEO2",
        "Primus/FIN2O",
        "Primus/FIO2",
        "Primus/FLOW_AIR",
        "Primus/FLOW_N2O",
        "Primus/FLOW_O2",
        "Primus/INCO2",
        "Primus/INSP_DES",
        "Primus/INSP_SEVO",
        "Primus/MAC",
        "Primus/MAWP_MBAR",
        "Primus/MV",
        "Primus/PAMB_MBAR",
        "Primus/PEEP_MBAR",
        "Primus/PIP_MBAR",
        "Primus/PPLAT_MBAR",
        "Primus/RR_CO2",
        "Primus/SET_AGE",
        "Primus/SET_FIO2",
        "Primus/SET_FLOW_TRIG",
        "Primus/SET_FRESH_FLOW",
        "Primus/SET_INSP_PAUSE",
        "Primus/SET_INSP_PRES",
        "Primus/SET_INSP_TM",
        "Primus/SET_INTER_PEEP",
        "Primus/SET_PIP",
        "Primus/SET_RR_IPPV",
        "Primus/SET_TV_L",
        "Primus/TV",
        "Primus/VENT_LEAK",
        "Orchestra/PPF20_CE",
        "Orchestra/PPF20_CP",
        "Orchestra/PPF20_CT",
        "Orchestra/PPF20_RATE",
        "Orchestra/PPF20_VOL",
        "Orchestra/RFTN20_CE",
        "Orchestra/RFTN20_CP",
        "Orchestra/RFTN20_CT",
        "Orchestra/RFTN20_RATE",
        "Orchestra/RFTN20_VOL",
        "BIS/BIS",
        "BIS/EMG",
        "BIS/SEF",
        "BIS/SQI",
        "BIS/SR",
        "BIS/TOTPOW",
        "Invos/SCO2_L",
        "Invos/SCO2_R",
        "CardioQ/CI",
        "CardioQ/CO",
        "CardioQ/FTc",
        "CardioQ/FTp",
        "CardioQ/HR",
        "CardioQ/MA",
        "CardioQ/MD",
        "CardioQ/PV",
        "CardioQ/SD",
        "CardioQ/SV",
        "CardioQ/SVI",
    ]

    required_tracks = [
        "BIS/BIS",
        "Orchestra/PPF20_RATE",
        "Orchestra/RFTN20_RATE",
        "Solar8000/ART_MBP",
    ]

    excluded_tracks = [
        "Primus/EXP_DES",
        "Primus/EXP_SEVO",
    ]

    case_ids = get_caseids(required_tracks, excluded_tracks)
    print(f"Eligible cases: {len(case_ids)}")

    save_cases(case_ids, included_tracks)