from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .classical_baseline import METADATA_COLUMNS
from .data_bootstrap import LABEL_TO_ID, _import_dbm, make_impairment_label

MERGE_KEYS = [
    "participant_name",
    "scenario_name",
    "round",
    "cognitive_task",
    "intoxicated",
]


def _normalize_merge_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in ["participant_name", "scenario_name", "round", "cognitive_task"]:
        if column in df.columns:
            df[column] = df[column].astype(str)
    if "intoxicated" in df.columns:
        df["intoxicated"] = df["intoxicated"].astype(bool)
    return df


def get_handcrafted_feature_config() -> dict[str, dict[str, Any]]:
    return {
        "ego_tracks": {
            "relative_path": "Resampled_previous_10_derived",
            "type": "tabular",
            "fps": 10,
            "aggregate": [
                {"column": "longitudinal_speed", "function": "pos_mean"},
                {"column": "longitudinal_acceleration", "function": "pos_mean"},
                {"column": "longitudinal_acceleration", "function": "neg_mean"},
            ],
        },
        "steering_reversals": {
            "relative_path": "Resampled_previous_10_derived",
            "type": "tabular",
            "fps": 10,
            "aggregate": [
                {"column": "reversals_0_5", "function": "sum_rate"},
                {"column": "reversals_2_5", "function": "sum_rate"},
            ],
        },
        "experiment_tobii_frame": {
            "relative_path": "Resampled_previous_60",
            "type": "tabular",
            "fps": 60,
            "aggregate": [
                {
                    "name": "tobii_eye_pupil_diameter",
                    "columns": ["tobii_left_eye_pupil_diameter", "tobii_right_eye_pupil_diameter"],
                    "function": "mean",
                }
            ],
        },
        "gaze_pitch_yaw": {
            "relative_path": "Resampled_previous_60_derived",
            "type": "tabular",
            "fps": 60,
            "aggregate": [
                {"name": "gaze_pitch", "columns": ["gaze_left_pitch", "gaze_right_pitch"], "function": "std"},
                {"name": "gaze_yaw", "columns": ["gaze_left_yaw", "gaze_right_yaw"], "function": "std"},
            ],
        },
        "gaze_fixation_saccade": {
            "relative_path": "Resampled_previous_60_derived",
            "type": "tabular",
            "fps": 60,
            "aggregate": [
                {
                    "name": "gaze_saccade",
                    "columns": ["gaze_left_saccade", "gaze_right_saccade"],
                    "function": "change_rate",
                },
                {
                    "name": "gaze_fixation",
                    "columns": ["gaze_left_fixation", "gaze_right_fixation"],
                    "function": "change_rate",
                },
            ],
        },
    }


def build_handcrafted_dataset(
    dbm_repo: str | Path,
    processed_root: str | Path,
    scenario_name_filter: str = r"driving/[0-9].*",
    chunk_strategy: str = "end",
    chunk_duration_s: float = 30.0,
    chunks_per_scenario: int | None = 1,
    chunk_end_offset_s: float = 0.0,
):
    DBM_Dataset, DBM_Dataset_Config = _import_dbm(dbm_repo)

    config = DBM_Dataset_Config()
    config.base_path = str(Path(processed_root).resolve())
    config.index_relative_path = "Resampled_previous_10"
    config.scenario_name_filter = scenario_name_filter
    config.chunk_strategy = chunk_strategy
    config.chunk_duration = float(chunk_duration_s)
    config.chunks_per_scenario = chunks_per_scenario
    config.chunk_end_offset = float(chunk_end_offset_s)
    config.cache_chunks = chunk_strategy != "random"
    config.features = get_handcrafted_feature_config()

    return DBM_Dataset(config)


def collect_handcrafted_feature_vector(sample) -> dict[str, float]:
    output: dict[str, float] = {}
    for feature_name, feature in sample.features.items():
        if not hasattr(feature, "aggregate"):
            continue
        aggregate = feature.aggregate
        if aggregate is None or len(aggregate) == 0:
            continue
        for column in aggregate.columns:
            output[f"feat__{column}"] = float(aggregate.iloc[0][column])
    return output


def handcrafted_feature_columns_from_table(feature_table: pd.DataFrame) -> list[str]:
    return sorted(column for column in feature_table.columns if column.startswith("feat__"))


def build_handcrafted_feature_table(
    dataset,
    manifest: pd.DataFrame | None = None,
    progress_every: int = 25,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = len(dataset)

    for sample_idx in range(total):
        sample = dataset[sample_idx]
        try:
            row = {
                "sample_idx": sample_idx,
                "participant_name": sample.participant.name,
                "scenario_name": sample.scenario.scenario_name,
                "round": sample.scenario.round,
                "cognitive_task": sample.scenario.cognitive_task,
                "intoxicated": bool(sample.scenario.intoxicated),
                "impairment_label": make_impairment_label(sample.scenario.cognitive_task, sample.scenario.intoxicated),
                "label_id": LABEL_TO_ID[
                    make_impairment_label(sample.scenario.cognitive_task, sample.scenario.intoxicated)
                ],
                "num_frames": int(len(sample.chunk.resample_times)),
                "chunk_start_ns": int(sample.chunk.start_epoch_ns),
                "chunk_end_ns": int(sample.chunk.end_epoch_ns),
                **collect_handcrafted_feature_vector(sample),
            }
            rows.append(row)
        finally:
            if hasattr(sample, "close"):
                sample.close()

        if progress_every and ((sample_idx + 1) % progress_every == 0 or sample_idx + 1 == total):
            print(f"Built handcrafted features for {sample_idx + 1}/{total} scenario windows")

    feature_table = pd.DataFrame(rows)

    if manifest is not None and len(feature_table) > 0:
        manifest = _normalize_merge_keys(manifest)
        feature_table = _normalize_merge_keys(feature_table)

        extra_columns = [
            column
            for column in ["fixed_fold", "scenario_code", "scenario_path", "duration_s"]
            if column in manifest.columns
        ]
        lookup = manifest[MERGE_KEYS + extra_columns].drop_duplicates(subset=MERGE_KEYS)
        feature_table = feature_table.merge(lookup, on=MERGE_KEYS, how="left")

    feature_columns = handcrafted_feature_columns_from_table(feature_table)
    ordered_columns = [column for column in METADATA_COLUMNS if column in feature_table.columns] + feature_columns
    other_columns = [column for column in feature_table.columns if column not in ordered_columns]
    return feature_table[ordered_columns + other_columns]
