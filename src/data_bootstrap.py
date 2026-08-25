from __future__ import annotations

import csv
import fnmatch
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import boto3
import botocore
import cv2
import numpy as np
import pandas as pd
import yaml

LABEL_ORDER = [
    "baseline",
    "cognitive_distraction",
    "alcohol_intoxication",
    "combined",
]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABEL_ORDER)}

_DERIVED_RECIPES = {
    "ego_tracks": {
        "relative_dir": "Resampled_previous_10_derived",
        "probe_file": "ego_tracks.csv",
        "script": "features/generate_derived_track_features.py",
    },
    "steering_reversals": {
        "relative_dir": "Resampled_previous_10_derived",
        "probe_file": "steering_reversals.csv",
        "script": "features/generate_derived_steering_features.py",
    },
    "gaze_pitch_yaw": {
        "relative_dir": "Resampled_previous_60_derived",
        "probe_file": "gaze_pitch_yaw.csv",
        "script": "features/generate_derived_gaze_pitch_yaw_features.py",
    },
    "gaze_fixation_saccade": {
        "relative_dir": "Resampled_previous_60_derived",
        "probe_file": "gaze_fixation_saccade.csv",
        "script": "features/generate_derived_gaze_fixation_features.py",
    },
}


def _import_dbm(dbm_repo: str | Path):
    dbm_repo = Path(dbm_repo).resolve()
    dbm_repo_str = str(dbm_repo)
    if dbm_repo_str not in sys.path:
        sys.path.insert(0, dbm_repo_str)
    from data.dbm_dataset import DBM_Dataset, DBM_Dataset_Config

    return DBM_Dataset, DBM_Dataset_Config


def _read_public_dataset_config(dbm_repo: str | Path) -> dict[str, Any]:
    cfg_path = Path(dbm_repo) / "data" / "configs" / "public-dataset.yaml"
    with cfg_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _matches_filters(
    key: str,
    include_patterns: list[str],
    exclude_patterns: list[str],
    s3_prefix: str,
) -> bool:
    include_hit = any(
        fnmatch.fnmatch(key, pattern) or fnmatch.fnmatch(key, f"{s3_prefix}/{pattern}")
        for pattern in include_patterns
    )
    exclude_hit = any(
        fnmatch.fnmatch(key, pattern) or fnmatch.fnmatch(key, f"{s3_prefix}/{pattern}")
        for pattern in exclude_patterns
    )
    return include_hit and not exclude_hit


def _list_s3_objects(bucket: str, prefix: str, client) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    continuation_token = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if continuation_token is not None:
            kwargs["ContinuationToken"] = continuation_token
        response = client.list_objects_v2(**kwargs)
        objects.extend(response.get("Contents", []))
        if not response.get("IsTruncated"):
            break
        continuation_token = response.get("NextContinuationToken")
    return objects


def _is_processed_dataset_present(processed_root: str | Path) -> bool:
    processed_root = Path(processed_root)
    return (
        (processed_root / "idd_annotation.csv").exists()
        and (processed_root / "Resampled_previous_10" / "Participants").exists()
    )


def _has_scenario_index(processed_root: str | Path) -> bool:
    participants_dir = Path(processed_root) / "Resampled_previous_10" / "Participants"
    if not participants_dir.exists():
        return False
    return any(participants_dir.rglob("scenario_index.csv"))


def _has_video_modality(processed_root: str | Path, video_name: str) -> bool:
    for relative_dir in ["Resampled_previous_10", "Resampled_previous_10_derived"]:
        participants_dir = Path(processed_root) / relative_dir / "Participants"
        if not participants_dir.exists():
            continue
        for video_dir in participants_dir.rglob(video_name):
            if video_dir.is_dir() and any(video_dir.glob("video.*")):
                return True
    return False


def _dataset_matches_version(processed_root: str | Path, version: str) -> bool:
    processed_root = Path(processed_root)
    if not _is_processed_dataset_present(processed_root):
        return False
    if version == "sample":
        return any((processed_root / "Resampled_previous_10" / "Participants").rglob("sim_bag"))
    if version in {"no-video", "no-video-only-raw"}:
        return _has_scenario_index(processed_root)
    if version == "only-rgb-video":
        return _has_scenario_index(processed_root) and _has_video_modality(processed_root, "cam_front")
    if version == "only-scaled-rgb-video":
        return _has_scenario_index(processed_root) and _has_video_modality(processed_root, "cam_front_scale4")
    if version == "all":
        return _has_scenario_index(processed_root) and _has_video_modality(processed_root, "cam_front")
    return _is_processed_dataset_present(processed_root)


def _choose_front_video_file_name(processed_root: str | Path) -> str | None:
    if _has_video_modality(processed_root, "cam_front_scale4"):
        return "cam_front_scale4"
    if _has_video_modality(processed_root, "cam_front"):
        return "cam_front"
    return None


def _choose_front_video_relative_path(processed_root: str | Path, video_name: str) -> str:
    if video_name == "cam_front_scale4":
        return "Resampled_previous_10_derived"
    return "Resampled_previous_10"


def _sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.replace(" ", "_")
    return df


def _load_csv_indexed(csv_path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = _sanitize_columns(df)
    if "resampled_epoch_ns" in df.columns:
        df = df.set_index("resampled_epoch_ns")
    if columns is not None:
        keep_columns = [column for column in columns if column in df.columns]
        df = df[keep_columns].copy()
    return df


def _read_fixed_fold_map(dbm_repo: str | Path) -> dict[str, int]:
    fold_map: dict[str, int] = {}
    folds_path = Path(dbm_repo) / "hail-datasets" / "hail_datasets" / "datasets" / "ddd_2024" / "folds.csv"
    with folds_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter=",")
        for fold_index, row in enumerate(reader):
            for participant_name in row:
                fold_map[participant_name] = fold_index
    return fold_map


def _estimate_scenario_timing(task_dir: Path) -> tuple[int, int, float, int]:
    candidate_files = [
        task_dir / "sim_bag" / "telemetry_roadwheel_angle.csv",
        task_dir / "sim_bag" / "telemetry_velocity.csv",
        task_dir / "sim_bag" / "experiment_tobii_frame.csv",
        task_dir / "cam_front_scale4" / "frame_timing.csv",
        task_dir / "cam_front" / "frame_timing.csv",
    ]
    for csv_path in candidate_files:
        if not csv_path.exists():
            continue
        df = _load_csv_indexed(csv_path)
        if len(df) == 0 or not isinstance(df.index, pd.Index):
            continue
        start_time = int(df.index.min())
        end_time = int(df.index.max())
        duration_s = max(0.0, (end_time - start_time) / 1e9)
        return start_time, end_time, duration_s, int(len(df))
    return 0, 0, 0.0, 0


def _scan_simple_manifest(dbm_repo: str | Path, processed_root: str | Path) -> pd.DataFrame:
    processed_root = Path(processed_root)
    participants_dir = processed_root / "Resampled_previous_10" / "Participants"
    fixed_fold_map = _read_fixed_fold_map(dbm_repo)
    rows: list[dict[str, Any]] = []

    if not participants_dir.exists():
        return pd.DataFrame()

    for participant_dir in sorted(participants_dir.iterdir()):
        if not participant_dir.is_dir():
            continue
        for round_dir in sorted(participant_dir.iterdir()):
            drive_dir = round_dir / "driving"
            if not drive_dir.exists():
                continue
            for scenario_dir in sorted(drive_dir.iterdir()):
                for task_dir in sorted(scenario_dir.iterdir()):
                    if not task_dir.is_dir():
                        continue
                    if not (task_dir / "sim_bag").exists():
                        continue

                    participant_name = participant_dir.name
                    round_name = round_dir.name
                    cognitive_task = task_dir.name
                    start_time, end_time, duration_s, duration_frames = _estimate_scenario_timing(task_dir)
                    intoxicated = participant_name.startswith("P") and round_name == "R2"

                    rows.append(
                        {
                            "participant_name": participant_name,
                            "scenario_name": f"driving/{scenario_dir.name}",
                            "scenario_code": scenario_dir.name.split("-")[0],
                            "scenario_path": f"{round_name}/driving/{scenario_dir.name}/{task_dir.name}",
                            "round": round_name,
                            "cognitive_task": cognitive_task,
                            "is_cd": cognitive_task != "no_task",
                            "intoxicated": intoxicated,
                            "impairment_label": make_impairment_label(cognitive_task, intoxicated),
                            "label_id": LABEL_TO_ID[make_impairment_label(cognitive_task, intoxicated)],
                            "fixed_fold": fixed_fold_map.get(participant_name),
                            "duration_s": duration_s,
                            "duration_frames": duration_frames,
                            "start_time": start_time,
                            "end_time": end_time,
                            "experiment_version": 1 if participant_name.startswith("P") else 2,
                            "number_all_crashes": np.nan,
                            "number_hazard_crashes": np.nan,
                            "has_any_crash": False,
                            "has_hazard_crash": False,
                            "anomalies": [],
                        }
                    )

    manifest = pd.DataFrame(rows)
    if len(manifest) == 0:
        return manifest
    manifest["fixed_fold"] = manifest["fixed_fold"].astype("Int64")
    return manifest


class _SimpleChunk:
    def __init__(self, start_epoch_ns: int, end_epoch_ns: int):
        self.start_epoch_ns = int(start_epoch_ns)
        self.end_epoch_ns = int(end_epoch_ns)

    def filter_resample_times(self, resample_times: np.ndarray) -> np.ndarray:
        return resample_times[(resample_times >= self.start_epoch_ns) & (resample_times <= self.end_epoch_ns)]

    def filter_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[self.start_epoch_ns : self.end_epoch_ns]

    def filter_intersections(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


class _SimpleScenario:
    def __init__(self, row: pd.Series):
        self.scenario_name = row["scenario_name"]
        self.scenario_path = row["scenario_path"]
        self.round = row["round"]
        self.cognitive_task = row["cognitive_task"]
        self.start_time = int(row["start_time"])
        self.end_time = int(row["end_time"])
        self.duration_s = float(row["duration_s"])
        self.duration_frames = int(row["duration_frames"])
        self.participant_name = row["participant_name"]
        self.intoxicated = bool(row["intoxicated"])
        self.experiment_version = int(row["experiment_version"])
        self.number_all_crashes = row["number_all_crashes"]
        self.number_hazard_crashes = row["number_hazard_crashes"]
        self.anomalies = row["anomalies"]

    def get_resample_times(self, fps: int) -> np.ndarray:
        if fps <= 0 or self.end_time <= self.start_time:
            return np.array([self.start_time, self.end_time], dtype=np.int64)
        step_ns = int(round(1e9 / fps))
        return np.arange(self.start_time, self.end_time + step_ns, step_ns, dtype=np.int64)


class _SimpleParticipant:
    def __init__(self, name: str):
        self.name = name


class _SimpleTabularFeature:
    def __init__(self, table: pd.DataFrame):
        self.table = table

    def close(self):
        pass


class _SimpleVideoFrame:
    def __init__(self, image_data: np.ndarray):
        self.image_data = image_data


class _SimpleVideoFrameReader:
    def __init__(self, video_path: Path, timing_data: pd.DataFrame):
        self.video_path = str(video_path)
        self.timing_data = timing_data
        self.cap = cv2.VideoCapture(self.video_path)

    def read_at_time(self, time_ns: int):
        if len(self.timing_data) == 0:
            return None
        idx = int(np.argmin(np.abs(self.timing_data.index.to_numpy(dtype=np.int64) - int(time_ns))))
        row = self.timing_data.iloc[idx]
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(row["resampled_frame"]))
        ok, frame = self.cap.read()
        if not ok:
            return None
        return _SimpleVideoFrame(frame)

    def close(self):
        self.cap.release()


class _SimpleVideoFeature:
    def __init__(self, timing_data: pd.DataFrame, video_path: Path):
        self.timing_data = timing_data
        self.video_path = str(video_path)

    def get_frame_reader(self):
        return _SimpleVideoFrameReader(Path(self.video_path), self.timing_data)

    def close(self):
        pass


class _SimpleSample:
    def __init__(self, processed_root: Path, row: pd.Series, features_config: dict[str, dict[str, Any]], chunk_strategy: str, chunk_duration_s: float):
        self.participant = _SimpleParticipant(row["participant_name"])
        self.scenario = _SimpleScenario(row)
        self.chunk = self._make_chunk(chunk_strategy, chunk_duration_s)
        self.features: dict[str, Any] = {}

        for feature_name, feature_attributes in features_config.items():
            loaded_feature = self._load_feature(processed_root, feature_name, feature_attributes)
            if loaded_feature is not None:
                self.features[feature_name] = loaded_feature

    def _make_chunk(self, chunk_strategy: str, chunk_duration_s: float) -> _SimpleChunk:
        if chunk_strategy == "full" or self.scenario.duration_s <= chunk_duration_s:
            return _SimpleChunk(self.scenario.start_time, self.scenario.end_time)

        chunk_ns = int(round(chunk_duration_s * 1e9))
        if chunk_strategy == "start":
            return _SimpleChunk(self.scenario.start_time, min(self.scenario.end_time, self.scenario.start_time + chunk_ns))
        if chunk_strategy == "random":
            max_start = max(self.scenario.start_time, self.scenario.end_time - chunk_ns)
            return _SimpleChunk(max_start, min(self.scenario.end_time, max_start + chunk_ns))
        return _SimpleChunk(max(self.scenario.start_time, self.scenario.end_time - chunk_ns), self.scenario.end_time)

    def _get_scenario_dir(self, processed_root: Path, relative_path: str) -> Path:
        scenario_parts = Path(self.scenario.scenario_path).parts
        return processed_root / relative_path / "Participants" / self.participant.name / Path(*scenario_parts)

    def _load_feature(self, processed_root: Path, feature_name: str, feature_attributes: dict[str, Any]):
        file_name = feature_attributes.get("file_name", feature_name)
        scenario_dir = self._get_scenario_dir(processed_root, feature_attributes["relative_path"])

        if feature_attributes["type"] == "tabular":
            csv_path = scenario_dir / f"{file_name}.csv"
            if not csv_path.exists():
                csv_path = scenario_dir / "sim_bag" / f"{file_name}.csv"
            if not csv_path.exists():
                return None
            df = _load_csv_indexed(csv_path, columns=feature_attributes.get("columns"))
            if "resampled_epoch_ns" in df.columns:
                df = df.set_index("resampled_epoch_ns")
            df = self.chunk.filter_dataframe(df)
            return _SimpleTabularFeature(df)

        if feature_attributes["type"] == "video":
            video_dir = scenario_dir / file_name
            if not video_dir.exists():
                return None
            timing_path = video_dir / "frame_timing.csv"
            if not timing_path.exists():
                return None
            timing_data = _load_csv_indexed(timing_path, columns=["resampled_frame"])
            if "resampled_frame" not in timing_data.columns:
                return None
            timing_data = self.chunk.filter_dataframe(timing_data)
            video_files = list(video_dir.glob("video.*"))
            if len(video_files) == 0:
                return None
            return _SimpleVideoFeature(timing_data, video_files[0])

        return None

    def close(self):
        for feature in self.features.values():
            if hasattr(feature, "close"):
                feature.close()


class _SimpleInspectionDataset:
    def __init__(self, processed_root: Path, manifest: pd.DataFrame, features_config: dict[str, dict[str, Any]], chunk_strategy: str, chunk_duration_s: float):
        self.processed_root = processed_root
        self.manifest = manifest.reset_index(drop=True)
        self.features_config = features_config
        self.chunk_strategy = chunk_strategy
        self.chunk_duration_s = chunk_duration_s

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index: int):
        row = self.manifest.iloc[index]
        return _SimpleSample(self.processed_root, row, self.features_config, self.chunk_strategy, self.chunk_duration_s)


def _ensure_license_acceptance(prompt_for_license: bool) -> None:
    license_text = (
        "The IDD public dataset is licensed under CC BY-NC-SA 4.0. "
        "You may share and adapt it for non-commercial use with attribution, "
        "and derived material must use the same license. "
        "Full terms: https://creativecommons.org/licenses/by-nc-sa/4.0/"
    )
    if prompt_for_license:
        response = input(f"{license_text}\nType 'y' to confirm and continue with the download: ").strip().lower()
        if response != "y":
            raise RuntimeError("Download cancelled because the dataset license was not accepted.")
        return
    raise RuntimeError(
        "The dataset is missing. Set prompt_for_license=True and rerun the notebook to confirm the dataset license."
    )


def download_public_dataset(
    dbm_repo: str | Path,
    output_dir: str | Path,
    version: str = "sample",
    prompt_for_license: bool = True,
    profile_name: str | None = None,
    use_credentials: bool = False,
) -> Path:
    cfg = _read_public_dataset_config(dbm_repo)
    if version not in cfg:
        raise KeyError(f'Unknown public dataset version "{version}".')

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    session_kwargs: dict[str, Any] = {}
    if profile_name is not None:
        session_kwargs["profile_name"] = profile_name
    session = boto3.Session(**session_kwargs)

    if use_credentials:
        client = session.client("s3")
    else:
        client = session.client("s3", config=botocore.config.Config(signature_version=botocore.UNSIGNED))

    bucket = cfg["s3_bucket_name"]
    prefix = cfg["s3_project_folder"].rstrip("/")
    include_patterns = cfg[version].get("include_patterns", ["*"])
    exclude_patterns = cfg[version].get("exclude_patterns", [])
    local_dataset_path = cfg[version].get("local_dataset_path", "")

    objects = _list_s3_objects(bucket=bucket, prefix=prefix, client=client)
    selected_objects = [
        obj
        for obj in objects
        if not obj["Key"].endswith("/")
        and _matches_filters(obj["Key"], include_patterns, exclude_patterns, prefix)
    ]
    required_bytes = sum(obj["Size"] for obj in selected_objects)
    available_bytes = shutil.disk_usage(output_dir).free

    print(f"Preparing to download IDD version '{version}' to {output_dir}.")
    print(f"Selected files: {len(selected_objects)}")
    print(f"Required space: {required_bytes / 2**30:.2f} GB")
    print(f"Available space: {available_bytes / 2**30:.2f} GB")

    if required_bytes > available_bytes:
        raise RuntimeError("Insufficient disk space for the requested IDD download.")

    _ensure_license_acceptance(prompt_for_license=prompt_for_license)

    local_root = output_dir / local_dataset_path
    local_root.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    for obj in selected_objects:
        key = obj["Key"]
        rel_path = key[len(prefix) :].lstrip("/\\")
        destination = local_root / rel_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.stat().st_size == obj["Size"]:
            skipped += 1
            continue
        client.download_file(bucket, key, str(destination))
        downloaded += 1

    print(f"Downloaded {downloaded} file(s); skipped {skipped} existing file(s).")
    return local_root / "Processed" if (local_root / "Processed").exists() else local_root


def ensure_processed_dataset(
    dbm_repo: str | Path,
    processed_root: str | Path,
    version: str = "sample",
    download_if_missing: bool = True,
    prompt_for_license: bool = True,
) -> Path:
    processed_root = Path(processed_root).expanduser().resolve()
    download_root = processed_root.parent if processed_root.name == "Processed" else processed_root

    if _dataset_matches_version(processed_root, version):
        return processed_root
    if _dataset_matches_version(processed_root.parent, version):
        return processed_root.parent
    if not download_if_missing:
        raise FileNotFoundError(
            f"IDD processed dataset was not found at {processed_root}. "
            "Enable download_if_missing or point PROCESSED_ROOT at an existing dataset."
        )
    downloaded_root = download_public_dataset(
        dbm_repo=dbm_repo,
        output_dir=download_root,
        version=version,
        prompt_for_license=prompt_for_license,
    )
    if _dataset_matches_version(downloaded_root, version):
        return downloaded_root
    if _dataset_matches_version(downloaded_root / "Processed", version):
        return downloaded_root / "Processed"
    if _dataset_matches_version(processed_root, version):
        return processed_root
    if _dataset_matches_version(processed_root.parent, version):
        return processed_root.parent
    raise FileNotFoundError(
        f"Download completed, but version '{version}' was not found in either "
        f"{downloaded_root} or {downloaded_root / 'Processed'}."
    )


def _feature_probe_exists(processed_root: str | Path, relative_dir: str, probe_file: str) -> bool:
    feature_dir = Path(processed_root) / relative_dir / "Participants"
    if not feature_dir.exists():
        return False
    return any(feature_dir.rglob(probe_file))


def ensure_dbm_derived_features(
    dbm_repo: str | Path,
    processed_root: str | Path,
    run_missing: bool = False,
    num_processes: int = 0,
) -> dict[str, dict[str, Any]]:
    dbm_repo = Path(dbm_repo).resolve()
    processed_root = Path(processed_root).resolve()

    status: dict[str, dict[str, Any]] = {}
    for feature_name, recipe in _DERIVED_RECIPES.items():
        exists = _feature_probe_exists(processed_root, recipe["relative_dir"], recipe["probe_file"])
        status[feature_name] = {
            "relative_dir": recipe["relative_dir"],
            "probe_file": recipe["probe_file"],
            "present": exists,
            "generated": False,
        }
        if exists or not run_missing:
            continue

        command = [
            sys.executable,
            recipe["script"],
            "--in_dir",
            str(processed_root),
            "--out_dir",
            str(processed_root / recipe["relative_dir"]),
            "--num_processes",
            str(int(num_processes)),
        ]
        subprocess.run(command, cwd=dbm_repo, check=True)

        exists = _feature_probe_exists(processed_root, recipe["relative_dir"], recipe["probe_file"])
        status[feature_name]["present"] = exists
        status[feature_name]["generated"] = exists
        if not exists:
            raise RuntimeError(f"Derived feature generation failed for {feature_name}.")

    return status


def make_impairment_label(cognitive_task: str, intoxicated: bool) -> str:
    is_cd = cognitive_task != "no_task"
    if not intoxicated and not is_cd:
        return "baseline"
    if not intoxicated and is_cd:
        return "cognitive_distraction"
    if intoxicated and not is_cd:
        return "alcohol_intoxication"
    return "combined"


def build_scenario_manifest(
    dbm_repo: str | Path,
    processed_root: str | Path,
    scenario_name_filter: str = r"driving/[0-9].*",
) -> pd.DataFrame:
    if not _has_scenario_index(processed_root):
        manifest = _scan_simple_manifest(dbm_repo, processed_root)
        if len(manifest) == 0:
            raise FileNotFoundError("No scenarios were found in the sample dataset layout.")
        return manifest[manifest["scenario_name"].str.match(scenario_name_filter)].reset_index(drop=True)

    DBM_Dataset, DBM_Dataset_Config = _import_dbm(dbm_repo)

    config = DBM_Dataset_Config()
    config.base_path = str(Path(processed_root).resolve())
    config.index_relative_path = "Resampled_previous_10"
    config.scenario_name_filter = scenario_name_filter
    config.chunk_strategy = "full"
    config.features = {}

    dataset = DBM_Dataset(config)
    manifest = dataset.scenario_index.copy().reset_index(drop=True)
    manifest["is_cd"] = manifest["cognitive_task"] != "no_task"
    manifest["impairment_label"] = [
        make_impairment_label(cd_task, intoxicated)
        for cd_task, intoxicated in zip(manifest["cognitive_task"], manifest["intoxicated"])
    ]
    manifest["label_id"] = manifest["impairment_label"].map(LABEL_TO_ID).astype(int)
    manifest["scenario_code"] = manifest["scenario_name"].str.split("/").str[-1].str.split("-").str[0]
    manifest["has_hazard_crash"] = manifest["number_hazard_crashes"].fillna(0) > 0
    manifest["has_any_crash"] = manifest["number_all_crashes"].fillna(0) > 0
    manifest["fixed_fold"] = manifest["fixed_fold"].astype("Int64")

    ordered_columns = [
        "participant_name",
        "scenario_name",
        "scenario_code",
        "round",
        "cognitive_task",
        "is_cd",
        "intoxicated",
        "impairment_label",
        "label_id",
        "fixed_fold",
        "duration_s",
        "experiment_version",
        "number_all_crashes",
        "number_hazard_crashes",
        "has_any_crash",
        "has_hazard_crash",
    ]
    other_columns = [column for column in manifest.columns if column not in ordered_columns]
    return manifest[ordered_columns + other_columns]


def _build_notebook_feature_config(processed_root: str | Path, include_video: bool = True) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {
        "experiment_tobii_frame": {
            "relative_path": "Resampled_previous_60",
            "type": "tabular",
            "fps": 60,
            "columns": [
                "tobii_left_eye_gaze_pt_validity",
                "tobii_right_eye_gaze_pt_validity",
                "tobii_left_eye_gaze_pt_in_display_x",
                "tobii_right_eye_gaze_pt_in_display_x",
                "tobii_left_eye_gaze_pt_in_display_y",
                "tobii_right_eye_gaze_pt_in_display_y",
                "tobii_left_eye_pupil_diameter",
                "tobii_right_eye_pupil_diameter",
            ],
        },
        "gaze_pitch_yaw": {
            "relative_path": "Resampled_previous_60_derived",
            "type": "tabular",
            "fps": 60,
            "columns": [
                "gaze_left_pitch",
                "gaze_right_pitch",
                "gaze_left_yaw",
                "gaze_right_yaw",
            ],
        },
        "gaze_fixation_saccade": {
            "relative_path": "Resampled_previous_60_derived",
            "type": "tabular",
            "fps": 60,
            "columns": [
                "gaze_left_fixation",
                "gaze_right_fixation",
                "gaze_left_saccade",
                "gaze_right_saccade",
                "gaze_left_fixation_saccade_ratio",
                "gaze_right_fixation_saccade_ratio",
            ],
        },
        "ego_tracks": {
            "relative_path": "Resampled_previous_10_derived",
            "type": "tabular",
            "fps": 10,
            "columns": [
                "carla_objects_pose_x",
                "carla_objects_pose_y",
                "yaw",
                "lateral_speed",
                "longitudinal_speed",
                "lateral_acceleration",
                "longitudinal_acceleration",
                "lateral_jerk",
                "longitudinal_jerk",
            ],
        },
        "telemetry_roadwheel_angle": {
            "relative_path": "Resampled_previous_10",
            "type": "tabular",
            "fps": 10,
            "columns": [
                "telemetry_roadwheel_angle_data",
            ],
        },
    }

    if include_video:
        front_video_name = _choose_front_video_file_name(processed_root)
        if front_video_name is not None:
            features["cam_front"] = {
                "relative_path": _choose_front_video_relative_path(processed_root, front_video_name),
                "file_name": front_video_name,
                "type": "video",
                "fps": 10,
                "columns": ["resampled_frame"],
                "skip_frames": True,
            }
    return features


def build_inspection_dataset(
    dbm_repo: str | Path,
    processed_root: str | Path,
    scenario_name_filter: str = r"driving/[0-9].*",
    chunk_strategy: str = "end",
    chunk_duration_s: float = 30.0,
    chunks_per_scenario: int | None = 1,
    chunk_end_offset_s: float = 0.0,
    include_video: bool = True,
):
    if not _has_scenario_index(processed_root):
        manifest = build_scenario_manifest(dbm_repo, processed_root, scenario_name_filter)
        return _SimpleInspectionDataset(
            processed_root=Path(processed_root).resolve(),
            manifest=manifest,
            features_config=_build_notebook_feature_config(processed_root, include_video=include_video),
            chunk_strategy=chunk_strategy,
            chunk_duration_s=chunk_duration_s,
        )

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
    config.features = _build_notebook_feature_config(processed_root, include_video=include_video)

    return DBM_Dataset(config)


def select_example_sample(
    dataset,
    preferred_label: str | None = None,
    require_video: bool = False,
):
    for sample_idx in range(len(dataset)):
        sample = dataset[sample_idx]
        label = make_impairment_label(sample.scenario.cognitive_task, sample.scenario.intoxicated)
        has_video = "cam_front" in sample.features
        if preferred_label is not None and label != preferred_label:
            sample.close()
            continue
        if require_video and not has_video:
            sample.close()
            continue
        return sample_idx, sample
    raise LookupError("No sample matched the requested label/video constraints.")


def extract_video_frame(
    sample,
    feature_name: str = "cam_front",
    frame_index: int | None = None,
) -> tuple[np.ndarray | None, int | None]:
    video = sample.features.get(feature_name)
    if video is None or not hasattr(video, "timing_data") or video.timing_data.empty:
        return None, None

    if frame_index is None:
        frame_index = len(video.timing_data) // 2
    frame_index = int(np.clip(frame_index, 0, len(video.timing_data) - 1))
    time_ns = int(video.timing_data.index[frame_index])

    reader = video.get_frame_reader()
    frame_data = reader.read_at_time(time_ns)
    reader.close()
    if frame_data is None:
        return None, None

    image = frame_data.image_data
    if image.ndim == 2:
        image = image[..., None]
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, time_ns


def overlay_gaze_on_frame(
    sample,
    rgb_frame: np.ndarray,
    time_ns: int,
    eye_preference: str = "left",
) -> np.ndarray:
    tobii = sample.features.get("experiment_tobii_frame")
    if tobii is None or tobii.table.empty:
        return rgb_frame

    table = tobii.table.reset_index()
    time_column = table.columns[0]
    closest_idx = (table[time_column] - time_ns).abs().idxmin()
    row = table.loc[closest_idx]

    out = rgb_frame.copy()
    for eye in [eye_preference, "right" if eye_preference == "left" else "left"]:
        x_col = f"tobii_{eye}_eye_gaze_pt_in_display_x"
        y_col = f"tobii_{eye}_eye_gaze_pt_in_display_y"
        valid_col = f"tobii_{eye}_eye_gaze_pt_validity"
        if x_col not in row or y_col not in row:
            continue
        if valid_col in row and float(row[valid_col]) <= 0:
            continue
        x = float(row[x_col])
        y = float(row[y_col])
        if np.isnan(x) or np.isnan(y):
            continue
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            continue
        px = int(round(x * (out.shape[1] - 1)))
        py = int(round(y * (out.shape[0] - 1)))
        cv2.circle(out, (px, py), 12, (255, 204, 0), 2)
        cv2.circle(out, (px, py), 4, (255, 80, 0), -1)
        return out
    return out


def _align_feature_table(
    table: pd.DataFrame,
    timeline_ns: np.ndarray,
    tolerance_ns: int,
) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame(index=pd.Index(timeline_ns, name="resampled_epoch_ns"))

    source = table.sort_index().reset_index()
    time_column = source.columns[0]
    source = source.rename(columns={time_column: "source_time_ns"})
    target = pd.DataFrame({"resampled_epoch_ns": timeline_ns})

    aligned = pd.merge_asof(
        target,
        source,
        left_on="resampled_epoch_ns",
        right_on="source_time_ns",
        direction="nearest",
        tolerance=tolerance_ns,
    )
    aligned = aligned.set_index("resampled_epoch_ns")
    return aligned.drop(columns=["source_time_ns"], errors="ignore")


def _compute_time_since_last_valid_seconds(df: pd.DataFrame) -> pd.DataFrame:
    timeline_ns = df.index.to_numpy(dtype=np.int64)
    elapsed_s = (timeline_ns - timeline_ns[0]) / 1e9
    output = pd.DataFrame(index=df.index)
    for column in df.columns:
        valid = df[column].notna().to_numpy()
        delta_s = np.zeros(len(df), dtype=np.float32)
        last_valid_time = None
        for idx, is_valid in enumerate(valid):
            if is_valid:
                delta_s[idx] = 0.0
                last_valid_time = timeline_ns[idx]
            elif last_valid_time is None:
                delta_s[idx] = float(elapsed_s[idx])
            else:
                delta_s[idx] = float((timeline_ns[idx] - last_valid_time) / 1e9)
        output[column] = delta_s
    return output


def engineer_multimodal_sample(
    sample,
    target_fps: int = 10,
    modality_names: list[str] | None = None,
) -> dict[str, Any]:
    if modality_names is None:
        modality_names = [
            "experiment_tobii_frame",
            "gaze_pitch_yaw",
            "gaze_fixation_saccade",
            "ego_tracks",
            "telemetry_roadwheel_angle",
        ]

    timeline_ns = sample.scenario.get_resample_times(target_fps)
    timeline_ns = sample.chunk.filter_resample_times(timeline_ns).astype(np.int64)
    tolerance_ns = int(round((1e9 / target_fps) / 2.0))

    values_parts = []
    masks_parts = []
    delta_parts = []
    for modality_name in modality_names:
        feature = sample.features.get(modality_name)
        if feature is None or not hasattr(feature, "table"):
            continue
        aligned = _align_feature_table(feature.table, timeline_ns=timeline_ns, tolerance_ns=tolerance_ns)
        aligned = aligned.add_prefix(f"{modality_name}__")
        values_parts.append(aligned)
        masks_parts.append(aligned.notna().astype(np.float32))
        delta_parts.append(_compute_time_since_last_valid_seconds(aligned))

    if not values_parts:
        raise ValueError("No tabular modalities were available for engineering.")

    values = pd.concat(values_parts, axis=1).sort_index(axis=1)
    masks = pd.concat(masks_parts, axis=1).sort_index(axis=1)
    delta_s = pd.concat(delta_parts, axis=1).sort_index(axis=1)

    filled_values = values.ffill().fillna(0.0)
    model_input = pd.concat(
        [
            filled_values.add_suffix("__value"),
            masks.add_suffix("__mask"),
            delta_s.add_suffix("__delta_s"),
        ],
        axis=1,
    ).sort_index(axis=1)

    metadata = {
        "participant_name": sample.participant.name,
        "scenario_name": sample.scenario.scenario_name,
        "round": sample.scenario.round,
        "cognitive_task": sample.scenario.cognitive_task,
        "intoxicated": bool(sample.scenario.intoxicated),
        "impairment_label": make_impairment_label(sample.scenario.cognitive_task, sample.scenario.intoxicated),
        "label_id": LABEL_TO_ID[
            make_impairment_label(sample.scenario.cognitive_task, sample.scenario.intoxicated)
        ],
        "target_fps": int(target_fps),
        "num_frames": int(len(timeline_ns)),
        "chunk_start_ns": int(sample.chunk.start_epoch_ns),
        "chunk_end_ns": int(sample.chunk.end_epoch_ns),
    }

    return {
        "timeline_ns": pd.Index(timeline_ns, name="resampled_epoch_ns"),
        "timeline_s": pd.Index((timeline_ns - timeline_ns[0]) / 1e9, name="time_s"),
        "values": values,
        "masks": masks,
        "delta_s": delta_s,
        "filled_values": filled_values,
        "model_input": model_input,
        "metadata": metadata,
    }
