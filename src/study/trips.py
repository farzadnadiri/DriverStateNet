"""Trip index, exclusion audit, and analysis windows.

Window boundaries come from TRI's own DBM chunking code so that the `pre90` window is
exactly the interval TRI used in `analysis/feature_test.py` (90 s ending 30 s before the
scenario end). DBM's removal filters are switched off here so every trip is listed and the
reason it would be excluded is recorded instead.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from ..data_bootstrap import _import_dbm

SCENARIO_CODES = ["1a", "2", "2b", "3c", "5", "6e", "7a", "8a"]
LABEL_ORDER = ["baseline", "cognitive_distraction", "alcohol_intoxication", "combined"]

# (end offset s, duration s) relative to scenario end; see docs/HYPOTHESES.md
WINDOWS = {
    "pre90": (30.0, 90.0),
    "pre30": (30.0, 30.0),
    "haz30": (0.0, 30.0),
}

# idd_annotation.csv column blocks (8 scenario columns each, after the ID column)
_ANNOT_BLOCKS = {
    "anomalies": 1,
    "hazard_outcome": 9,
    "crash_times": 17,
    "hazard_valid": 25,
    "off_course": 106,
}


def label_for(cd: bool, intoxicated: bool) -> str:
    if intoxicated:
        return "combined" if cd else "alcohol_intoxication"
    return "cognitive_distraction" if cd else "baseline"


def _parse_ranges(cell: str) -> list[tuple[float, float]]:
    ranges = []
    for part in cell.split(","):
        bounds = [x.strip() for x in part.strip().split("-") if x.strip() != ""]
        if len(bounds) == 2:
            ranges.append((float(bounds[0]), float(bounds[1])))
    return ranges


def _parse_times(cell: str) -> list[float]:
    return [float(x) for x in cell.replace(";", ",").split(",") if x.strip() != ""]


def read_annotations(processed_root: str | Path) -> pd.DataFrame:
    """One row per (participant, scenario_code) from idd_annotation.csv."""
    rows = []
    with open(Path(processed_root) / "idd_annotation.csv") as handle:
        reader = csv.reader(handle)
        next(reader)
        next(reader)
        for row in reader:
            participant = row[0]
            if not participant or participant[0] not in ("P", "7"):
                continue
            for i, code in enumerate(SCENARIO_CODES):
                valid = row[_ANNOT_BLOCKS["hazard_valid"] + i].strip()
                rows.append(
                    {
                        "participant_name": participant,
                        "scenario_code": code,
                        "anomaly_ranges_s": _parse_ranges(row[_ANNOT_BLOCKS["anomalies"] + i]),
                        "hazard_outcome": row[_ANNOT_BLOCKS["hazard_outcome"] + i].strip().lower(),
                        "crash_times_s": _parse_times(row[_ANNOT_BLOCKS["crash_times"] + i]),
                        "hazard_valid": int(valid) if valid in ("0", "1") else np.nan,
                        "off_course_ranges_s": _parse_ranges(row[_ANNOT_BLOCKS["off_course"] + i]),
                    }
                )
    return pd.DataFrame(rows)


def _dbm_dataset(dbm_repo, processed_root, end_offset_s: float, duration_s: float):
    DBM_Dataset, DBM_Dataset_Config = _import_dbm(dbm_repo)
    config = DBM_Dataset_Config()
    config.base_path = str(Path(processed_root).resolve())
    config.index_relative_path = "Resampled_previous_10"
    config.scenario_name_filter = r"driving/[0-9].*"
    config.remove_participants = []
    config.scenario_remove_anomalies = False
    config.chunk_remove_anomalies = None
    config.chunk_strategy = "end"
    config.chunk_end_offset = float(end_offset_s)
    config.chunk_duration = float(duration_s)
    config.chunks_per_scenario = 1
    config.chunk_fps = 10
    config.cache_chunks = True
    config.features = {}
    return DBM_Dataset(config)


def build_trip_table(dbm_repo: str | Path, processed_root: str | Path) -> pd.DataFrame:
    """All trips with labels, cohort, fold, exclusion flags, and every window's bounds."""
    _, DBM_Dataset_Config = _import_dbm(dbm_repo)
    default_removed = set(DBM_Dataset_Config().remove_participants)

    trips = None
    for window, (end_offset_s, duration_s) in WINDOWS.items():
        dataset = _dbm_dataset(dbm_repo, processed_root, end_offset_s, duration_s)
        # Scenario.chunk_index is the scenario's row in dataset.scenario_index; scenarios
        # too short for the window get no chunk and keep NaN bounds.
        bounds = {}
        for idx in range(len(dataset)):
            chunk_index = dataset.chunk_factory.get_index(idx)
            scenario = dataset.scenarios[chunk_index.scenario_number]
            chunk = dataset.chunk_factory.get_chunk(chunk_index, scenario)
            bounds[scenario.chunk_index] = (int(chunk.start_epoch_ns), int(chunk.end_epoch_ns))
        if trips is None:
            index = dataset.scenario_index.reset_index(drop=True).copy()
            trips = index[
                [
                    "participant_name",
                    "scenario_name",
                    "scenario_path",
                    "round",
                    "cognitive_task",
                    "intoxicated",
                    "experiment_version",
                    "start_time",
                    "end_time",
                    "duration_s",
                    "fixed_fold",
                ]
            ].copy()
        trips[f"{window}_start_ns"] = [bounds[i][0] if i in bounds else np.nan for i in range(len(trips))]
        trips[f"{window}_end_ns"] = [bounds[i][1] if i in bounds else np.nan for i in range(len(trips))]

    trips["scenario_code"] = trips["scenario_name"].str.split("/").str[1].str.split("-").str[0]
    trips["cd"] = trips["cognitive_task"] != "no_task"
    trips["intoxicated"] = trips["intoxicated"].astype(bool)
    trips["label"] = [label_for(cd, tox) for cd, tox in zip(trips["cd"], trips["intoxicated"])]
    trips["cohort_v1"] = trips["experiment_version"] == 1
    trips["round2"] = trips["round"] == "R2"

    annotations = read_annotations(processed_root)
    trips = trips.merge(annotations, on=["participant_name", "scenario_code"], how="left")

    trips["removed_participant"] = trips["participant_name"].isin(default_removed)
    trips["has_anomaly"] = trips["anomaly_ranges_s"].apply(lambda r: isinstance(r, list) and len(r) > 0)
    trips["included_primary"] = ~trips["removed_participant"] & ~trips["has_anomaly"]
    trips["exclusion_reason"] = np.select(
        [trips["removed_participant"], trips["has_anomaly"]],
        ["removed_participant (DBM default)", "simulator anomaly (DBM scenario_remove_anomalies)"],
        default="",
    )
    for window in WINDOWS:
        trips[f"{window}_anomaly_overlap"] = [
            _overlaps(ranges, start, end)
            for ranges, start, end in zip(trips["anomaly_ranges_s"], trips[f"{window}_start_ns"], trips[f"{window}_end_ns"])
        ]
    return trips.sort_values(["participant_name", "round", "scenario_code"]).reset_index(drop=True)


def _overlaps(ranges, start_ns: int, end_ns: int) -> bool:
    if not isinstance(ranges, list):
        return False
    return any(lo * 1e9 <= end_ns and hi * 1e9 >= start_ns for lo, hi in ranges)


def audit_tables(trips: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Cross-tabs for the exclusion audit."""
    tables = {}
    tables["by_reason_and_label"] = pd.crosstab(
        trips["exclusion_reason"].replace("", "included"), trips["label"]
    ).reindex(columns=LABEL_ORDER, fill_value=0)
    kept = trips[~trips["removed_participant"]]
    tables["study_participants_by_reason_and_label"] = pd.crosstab(
        kept["exclusion_reason"].replace("", "included"), kept["label"]
    ).reindex(columns=LABEL_ORDER, fill_value=0)
    included = trips[trips["included_primary"]]
    tables["included_by_cohort_round_task"] = pd.crosstab(
        [included["experiment_version"], included["round"]], included["cognitive_task"]
    )
    tables["scenario_by_label_included"] = pd.crosstab(included["scenario_code"], included["label"]).reindex(
        columns=LABEL_ORDER, fill_value=0
    )
    tables["hazard_outcome_by_label_included"] = pd.crosstab(included["hazard_outcome"], included["label"]).reindex(
        columns=LABEL_ORDER, fill_value=0
    )
    tables["participants_by_cohort"] = (
        included.groupby("experiment_version")["participant_name"].nunique().rename("participants").to_frame()
    )
    return tables
