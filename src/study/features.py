"""Per-trip window features.

Two groups:
- `tri__*`: TRI's 10 DBM features (dataset/features/feature_sets.py "main"), computed with
  DBM's own `calculate_aggregate` and the same per-column valid-count weighting as
  `ddd_2024_features/tabular_feature.py`, so they reproduce DBM's values for the same chunk.
- `beh__*`: standard driving-performance and gaze measures used for the behavioural models.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .trips import WINDOWS

_DDD_DIR = "hail-datasets/hail_datasets/datasets/ddd_2024"

# TRI "main" feature set: (file key, output name, columns, aggregate function, fps)
TRI_FEATURES = [
    ("ego", "longitudinal_speed__pos_mean", ["longitudinal_speed"], "pos_mean", 10),
    ("ego", "longitudinal_acceleration__pos_mean", ["longitudinal_acceleration"], "pos_mean", 10),
    ("ego", "longitudinal_acceleration__neg_mean", ["longitudinal_acceleration"], "neg_mean", 10),
    ("reversals", "reversals_0_5__sum_rate", ["reversals_0_5"], "sum_rate", 10),
    ("reversals", "reversals_2_5__sum_rate", ["reversals_2_5"], "sum_rate", 10),
    (
        "tobii",
        "tobii_eye_pupil_diameter__mean",
        ["tobii_left_eye_pupil_diameter", "tobii_right_eye_pupil_diameter"],
        "mean",
        60,
    ),
    ("pitch_yaw", "gaze_pitch__std", ["gaze_left_pitch", "gaze_right_pitch"], "std", 60),
    ("pitch_yaw", "gaze_yaw__std", ["gaze_left_yaw", "gaze_right_yaw"], "std", 60),
    ("fix_sac", "gaze_saccade__change_rate", ["gaze_left_saccade", "gaze_right_saccade"], "change_rate", 60),
    ("fix_sac", "gaze_fixation__change_rate", ["gaze_left_fixation", "gaze_right_fixation"], "change_rate", 60),
]
TRI_COLUMNS = [f"tri__{name}" for _, name, _, _, _ in TRI_FEATURES]

PRC_RADIUS_DEG = 8.0

# file key -> (relative root, file inside the trip directory)
_FILES = {
    "ego": ("Resampled_previous_10_derived", "ego_tracks.csv"),
    "reversals": ("Resampled_previous_10_derived", "steering_reversals.csv"),
    "tobii": ("Resampled_previous_60", "sim_bag/experiment_tobii_frame.csv"),
    "pitch_yaw": ("Resampled_previous_60_derived", "gaze_pitch_yaw.csv"),
    "fix_sac": ("Resampled_previous_60_derived", "gaze_fixation_saccade.csv"),
    "steer": ("Resampled_previous_10", "sim_bag/usercontrols_status_steeringwheel_angle.csv"),
    "hero": ("Resampled_previous_10", "sim_bag/carla_hero_vehicle_status.csv"),
    "hazard": ("Resampled_previous_10_derived", "hazard_tracks.csv"),
}

# Straight-road lateral control (lane position is not in IDD; these are proxies)
STRAIGHT_YAW_RATE_DEG_S = 3.0  # |yaw rate| below this, smoothed over 1 s, counts as straight
STEER_HIGHPASS_S = 3.0  # steering minus its centred moving average over this span
PATH_SMOOTH_S = 5.0  # lateral deviation from the centred moving-average path over this span
MIN_STRAIGHT_S = 10.0  # need at least this much straight driving in a window

# Hazard response. The door-open hazard (2b) has no actor motion in hazard_tracks, so its
# onset cannot be estimated and it is excluded from response measures.
HAZARD_MOTION_M_S = 0.5
HAZARD_SEARCH_S = 40.0  # look for actor motion onset in the last 40 s of the scenario
RESPONSE_LIMIT_S = 6.0
BRAKE_ON = 0.02
THROTTLE_OFF = 0.02
NO_ONSET_SCENARIOS = {"2b"}


def _calculate_aggregate(dbm_repo: str | Path):
    path = str((Path(dbm_repo) / _DDD_DIR).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)
    from ddd_2024_features.tabular_aggregate_functions import calculate_aggregate

    return calculate_aggregate


def _read(processed_root: Path, key: str, participant: str, scenario_path: str) -> pd.DataFrame | None:
    rel_root, file_name = _FILES[key]
    path = processed_root / rel_root / "Participants" / participant / scenario_path / file_name
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = [c.replace(" ", "_") for c in df.columns]
    return df.set_index("resampled_epoch_ns").sort_index()


def _tri_aggregate(calculate_aggregate, data: np.ndarray, func: str, fps: int) -> float:
    total, weight = 0.0, 0
    for i in range(data.shape[1]):
        valid = int(np.sum(~np.isnan(data[:, i])))
        total += valid * calculate_aggregate(data[:, i], func, fps)
        weight += valid
    return total / weight if weight else np.nan


def _prc(pitch_yaw: pd.DataFrame) -> tuple[float, float]:
    """Percent road centre and gaze validity from binocular-mean pitch/yaw (degrees)."""
    yaw = pitch_yaw[["gaze_left_yaw", "gaze_right_yaw"]].mean(axis=1, skipna=True).to_numpy()
    pitch = pitch_yaw[["gaze_left_pitch", "gaze_right_pitch"]].mean(axis=1, skipna=True).to_numpy()
    valid = ~np.isnan(yaw) & ~np.isnan(pitch)
    valid_frac = float(valid.mean()) if len(valid) else np.nan
    if valid.sum() < 60:
        return np.nan, valid_frac
    cy, cp = np.median(yaw[valid]), np.median(pitch[valid])
    dist = np.hypot(yaw[valid] - cy, pitch[valid] - cp)
    return float(np.mean(dist <= PRC_RADIUS_DEG)), valid_frac


def _lateral_series(frames: dict) -> pd.DataFrame | None:
    """Full-trip 10 Hz series for straight-road lateral control, computed before windowing so
    the moving averages are not truncated at window edges."""
    ego, steer = frames.get("ego"), frames.get("steer")
    if ego is None or steer is None or len(ego) < 100:
        return None
    yaw = np.unwrap(ego["yaw"].to_numpy(float), period=360.0)
    yaw_rate = pd.Series(np.gradient(yaw) * 10.0, index=ego.index).rolling(10, center=True, min_periods=1).median()
    window = lambda s: int(round(s * 10))  # noqa: E731
    steer_angle = steer["usercontrols_status_steeringwheel_angle_data"].reindex(ego.index)
    steer_hf = steer_angle - steer_angle.rolling(window(STEER_HIGHPASS_S), center=True, min_periods=1).mean()

    x, y = ego["carla_objects_pose_x"], ego["carla_objects_pose_y"]
    dx = x - x.rolling(window(PATH_SMOOTH_S), center=True, min_periods=1).mean()
    dy = y - y.rolling(window(PATH_SMOOTH_S), center=True, min_periods=1).mean()
    heading = np.deg2rad(yaw)
    lateral_dev = -dx * np.sin(heading) + dy * np.cos(heading)  # component normal to heading
    return pd.DataFrame(
        {"straight": yaw_rate.abs() < STRAIGHT_YAW_RATE_DEG_S, "steer_hf": steer_hf, "lateral_dev": lateral_dev},
        index=ego.index,
    )


def hazard_response(frames: dict, scenario_code: str) -> dict[str, float]:
    """Hazard onset = first time the hazard actor exceeds 0.5 m/s in the last 40 s; response
    measures are taken in the following 6 s."""
    out = {key: np.nan for key in [
        "haz__onset_before_end_s", "haz__brake_rt_s", "haz__throttle_release_rt_s", "haz__peak_decel",
        "haz__min_distance_m", "haz__speed_at_onset", "haz__braking_at_onset",
    ]}
    hazard, ego, hero = frames.get("hazard"), frames.get("ego"), frames.get("hero")
    if scenario_code in NO_ONSET_SCENARIOS or hazard is None or ego is None or hero is None:
        return out
    t = hazard.index.to_numpy() / 1e9
    end = t[-1]
    speed = np.hypot(hazard["carla_objects_twist_linear_vx"], hazard["carla_objects_twist_linear_vy"]).to_numpy()
    moving = np.where((speed > HAZARD_MOTION_M_S) & (t >= end - HAZARD_SEARCH_S))[0]
    if len(moving) == 0:
        return out
    onset = t[moving[0]]
    out["haz__onset_before_end_s"] = end - onset

    hero_t = hero.index.to_numpy() / 1e9
    after = (hero_t >= onset) & (hero_t <= onset + RESPONSE_LIMIT_S)
    at_onset = int(np.searchsorted(hero_t, onset))
    if at_onset >= len(hero):
        return out
    brake, throttle = hero["brake"].to_numpy(float), hero["throttle"].to_numpy(float)
    out["haz__braking_at_onset"] = float(brake[at_onset] > BRAKE_ON)
    if brake[at_onset] <= BRAKE_ON:
        pressed = np.where(after & (brake > BRAKE_ON))[0]
        out["haz__brake_rt_s"] = hero_t[pressed[0]] - onset if len(pressed) else np.nan
    if throttle[at_onset] >= THROTTLE_OFF:
        released = np.where(after & (throttle < THROTTLE_OFF))[0]
        out["haz__throttle_release_rt_s"] = hero_t[released[0]] - onset if len(released) else np.nan

    ego_t = ego.index.to_numpy() / 1e9
    ego_after = (ego_t >= onset) & (ego_t <= onset + RESPONSE_LIMIT_S)
    out["haz__peak_decel"] = float(np.nanmax(-ego["longitudinal_acceleration"].to_numpy()[ego_after])) if ego_after.any() else np.nan
    out["haz__speed_at_onset"] = float(ego["longitudinal_speed"].to_numpy()[min(int(np.searchsorted(ego_t, onset)), len(ego) - 1)])
    joined = hazard[["carla_objects_pose_x", "carla_objects_pose_y"]].join(
        ego[["carla_objects_pose_x", "carla_objects_pose_y"]], rsuffix="_ego", how="inner"
    )
    joined = joined[joined.index.to_numpy() / 1e9 >= onset]
    if len(joined):
        out["haz__min_distance_m"] = float(np.nanmin(np.hypot(
            joined["carla_objects_pose_x"] - joined["carla_objects_pose_x_ego"],
            joined["carla_objects_pose_y"] - joined["carla_objects_pose_y_ego"],
        )))
    return out


def _window_features(calculate_aggregate, frames: dict, start_ns: int, end_ns: int) -> dict[str, float]:
    sliced = {k: (v.loc[start_ns:end_ns] if v is not None else None) for k, v in frames.items()}
    out: dict[str, float] = {}
    for key, name, columns, func, fps in TRI_FEATURES:
        df = sliced[key]
        if df is None or not set(columns).issubset(df.columns) or len(df) == 0:
            out[f"tri__{name}"] = np.nan
            continue
        out[f"tri__{name}"] = _tri_aggregate(calculate_aggregate, df[columns].to_numpy(dtype=float), func, fps)

    ego = sliced["ego"]
    if ego is not None and len(ego):
        out["beh__speed_mean"] = float(ego["longitudinal_speed"].mean())
        out["beh__speed_sd"] = float(ego["longitudinal_speed"].std())
        out["beh__lat_accel_sd"] = float(ego["lateral_acceleration"].std())
    steer = sliced["steer"]
    if steer is not None and len(steer):
        out["beh__steer_angle_sd"] = float(steer["usercontrols_status_steeringwheel_angle_data"].std())
    lateral = sliced.get("lateral")
    if lateral is not None and len(lateral):
        straight = lateral[lateral["straight"]]
        out["beh__straight_share"] = float(lateral["straight"].mean())
        if len(straight) >= MIN_STRAIGHT_S * 10:
            out["beh__steer_hf_sd_straight"] = float(straight["steer_hf"].std())
            out["beh__lateral_dev_sd_straight"] = float(straight["lateral_dev"].std())
    pitch_yaw = sliced["pitch_yaw"]
    if pitch_yaw is not None and len(pitch_yaw):
        out["beh__prc"], out["beh__gaze_valid_frac"] = _prc(pitch_yaw)
    out["beh__gaze_yaw_sd"] = out.get("tri__gaze_yaw__std", np.nan)
    out["beh__gaze_pitch_sd"] = out.get("tri__gaze_pitch__std", np.nan)
    out["beh__fixation_rate"] = out.get("tri__gaze_fixation__change_rate", np.nan)
    out["beh__saccade_rate"] = out.get("tri__gaze_saccade__change_rate", np.nan)
    out["beh__pupil_mean"] = out.get("tri__tobii_eye_pupil_diameter__mean", np.nan)
    out["beh__reversals_0_5"] = out.get("tri__reversals_0_5__sum_rate", np.nan)
    out["beh__reversals_2_5"] = out.get("tri__reversals_2_5__sum_rate", np.nan)
    return out


def _trip_rows(dbm_repo, processed_root: Path, trip: dict) -> list[dict]:
    calculate_aggregate = _calculate_aggregate(dbm_repo)
    frames = {key: _read(processed_root, key, trip["participant_name"], trip["scenario_path"]) for key in _FILES}
    hazard = hazard_response(frames, trip["scenario_code"])
    frames["lateral"] = _lateral_series(frames)
    rows = []
    for window in WINDOWS:
        start_ns, end_ns = trip[f"{window}_start_ns"], trip[f"{window}_end_ns"]
        if pd.isna(start_ns) or pd.isna(end_ns):
            continue
        row = {"participant_name": trip["participant_name"], "scenario_path": trip["scenario_path"], "window": window}
        row.update(_window_features(calculate_aggregate, frames, int(start_ns), int(end_ns)))
        row.update(hazard)  # trip-level; repeated on each window row for convenience
        rows.append(row)
    return rows


def _binocular_gaze(processed_root: Path, trip: dict) -> dict[str, np.ndarray]:
    """Valid binocular (yaw, pitch) samples per window for one trip."""
    pitch_yaw = _read(processed_root, "pitch_yaw", trip["participant_name"], trip["scenario_path"])
    out = {}
    for window in WINDOWS:
        start_ns, end_ns = trip[f"{window}_start_ns"], trip[f"{window}_end_ns"]
        if pitch_yaw is None or pd.isna(start_ns):
            continue
        sliced = pitch_yaw.loc[int(start_ns):int(end_ns)]
        yaw = sliced[["gaze_left_yaw", "gaze_right_yaw"]].mean(axis=1, skipna=True).to_numpy()
        pitch = sliced[["gaze_left_pitch", "gaze_right_pitch"]].mean(axis=1, skipna=True).to_numpy()
        valid = ~np.isnan(yaw) & ~np.isnan(pitch)
        out[window] = np.column_stack([yaw[valid], pitch[valid]])
    return out


def driver_centre_prc(trips: pd.DataFrame, processed_root: str | Path, n_jobs: int = 8) -> pd.DataFrame:
    """PRC around each driver's own road centre: the median gaze direction over that driver's
    sober R1 no-task trips (pre90 window), closer to Victor et al.'s (2005) definition than the
    per-window median used for `beh__prc`."""
    processed_root = Path(processed_root)
    records = trips.to_dict("records")
    samples = Parallel(n_jobs=n_jobs)(delayed(_binocular_gaze)(processed_root, trip) for trip in records)
    centres = {}
    for trip, gaze in zip(records, samples):
        reference = trip["included_primary"] and trip["round"] == "R1" and trip["cognitive_task"] == "no_task"
        if reference and "pre90" in gaze and len(gaze["pre90"]):
            centres.setdefault(trip["participant_name"], []).append(gaze["pre90"])
    centres = {p: np.median(np.vstack(arrays), axis=0) for p, arrays in centres.items()}
    rows = []
    for trip, gaze in zip(records, samples):
        centre = centres.get(trip["participant_name"])
        for window, xy in gaze.items():
            value = np.nan
            if centre is not None and len(xy) >= 60:
                value = float(np.mean(np.hypot(xy[:, 0] - centre[0], xy[:, 1] - centre[1]) <= PRC_RADIUS_DEG))
            rows.append({"participant_name": trip["participant_name"], "scenario_path": trip["scenario_path"],
                         "window": window, "beh__prc_driver": value})
    return pd.DataFrame(rows)


def build_window_features(
    trips: pd.DataFrame, dbm_repo: str | Path, processed_root: str | Path, n_jobs: int = 8
) -> pd.DataFrame:
    """Long table: one row per (trip, window)."""
    processed_root = Path(processed_root)
    records = trips.to_dict("records")
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_trip_rows)(str(dbm_repo), processed_root, trip) for trip in records
    )
    return pd.DataFrame([row for rows in results for row in rows])
