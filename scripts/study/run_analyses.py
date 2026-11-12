"""Stage 2: behavioural models (H1-H3), order/cohort diagnostics (H5), classification by
window and with hazard-response features (H4), and robustness checks.

    python -m scripts.study.run_analyses

Needs stage 1 (scripts.study.build_features). Aggregate results go to
artifacts/study/results/; row-level predictions to artifacts/study/private/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score

from src.evaluation import bootstrap_uar_ci, paired_bootstrap_delta
from src.study.features import TRI_COLUMNS
from src.study.models import fit_and_apply, mixed_effects, nested_cv
from src.study.trips import LABEL_ORDER, WINDOWS

ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ROOT / "artifacts/study/private"
RESULTS = ROOT / "artifacts/study/results"

BEHAVIOUR_MEASURES = [
    "beh__gaze_yaw_sd",
    "beh__gaze_pitch_sd",
    "beh__prc",
    "beh__fixation_rate",
    "beh__saccade_rate",
    "beh__pupil_mean",
    "beh__steer_hf_sd_straight",
    "beh__lateral_dev_sd_straight",
    "beh__steer_angle_sd",
    "beh__reversals_0_5",
    "beh__reversals_2_5",
    "beh__lat_accel_sd",
    "beh__speed_mean",
    "beh__speed_sd",
    "beh__gaze_valid_frac",
]
HAZARD_MEASURES = [
    "haz__log_brake_rt",
    "haz__log_throttle_release_rt",
    "haz__braked",
    "haz__peak_decel",
    "haz__min_distance_m",
    "haz__crash",
]
HAZARD_FEATURES = ["haz__brake_rt_s", "haz__throttle_release_rt_s", "haz__braked", "haz__peak_decel", "haz__min_distance_m"]

LMM_RHS = "cohort_v1 + round2 + alcohol + cd + alcohol:cd + C(scenario_code)"
LMM_TERMS = ["round2", "alcohol", "cd", "alcohol:cd", "cohort_v1"]
KEY_TERMS = ["cd", "alcohol", "alcohol:cd", "round2"]

TASKS = {
    "four_class": ("label", LABEL_ORDER),
    "cd_binary": ("cd_label", ["no_cd", "cd"]),
    "intox_binary": ("intox_label", ["sober", "intoxicated"]),
}


def load_window(window: str, inclusion: str = "primary") -> pd.DataFrame:
    """inclusion: 'primary' = DBM rule (drop any trip with an annotated anomaly);
    'window_overlap' = drop only trips whose anomaly overlaps this window."""
    trips = pd.read_pickle(PRIVATE / "trips.pkl")
    features = pd.read_pickle(PRIVATE / "window_features.pkl")
    if inclusion == "primary":
        keep = trips["included_primary"]
    elif inclusion == "all":
        keep = ~trips["removed_participant"]
    elif inclusion == "window_overlap":
        keep = ~trips["removed_participant"] & ~trips[f"{window}_anomaly_overlap"]
    else:
        raise ValueError(inclusion)
    data = trips[keep].merge(
        features[features["window"] == window], on=["participant_name", "scenario_path"], how="inner"
    )
    data["cd_label"] = data["cd"].map({True: "cd", False: "no_cd"})
    data["intox_label"] = data["intoxicated"].map({True: "intoxicated", False: "sober"})
    data["cohort_label"] = data["cohort_v1"].map({True: "v1", False: "v2"})
    data["alcohol"] = data["intoxicated"].astype(int)
    for column in ["cohort_v1", "round2", "cd"]:
        data[column] = data[column].astype(int)
    data["fixed_fold"] = data["fixed_fold"].astype(int)

    # Hazard response: defined only where an onset was found and the driver was not already braking
    onset = data["haz__onset_before_end_s"].notna()
    data["haz__braked"] = np.where(onset & (data["haz__braking_at_onset"] == 0), data["haz__brake_rt_s"].notna(), np.nan)
    data["haz__log_brake_rt"] = np.log(data["haz__brake_rt_s"])
    data["haz__log_throttle_release_rt"] = np.log(data["haz__throttle_release_rt_s"])
    outcome = data["hazard_outcome"].fillna("")
    data["haz__crash"] = np.where(outcome.str.startswith("fail"), np.nan, outcome.str.contains("crash").astype(float))
    return data.reset_index(drop=True)


def _ci_row(name: str, predictions: pd.DataFrame, labels: list[str], **extra) -> dict:
    ci = bootstrap_uar_ci(predictions, labels=labels)
    recalls = recall_score(predictions["true_label"], predictions["pred_label"], labels=labels, average=None, zero_division=0)
    return {"analysis": name, "uar": ci["uar"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
            "n_trips": len(predictions), "n_participants": ci["n_groups"], **extra,
            **{f"recall__{k}": v for k, v in zip(labels, recalls)}}


def _paired(pa: pd.DataFrame, pb: pd.DataFrame, a: str, b: str, labels: list[str]) -> dict:
    keys = ["participant_name", "scenario_path"]
    common = pa[keys].merge(pb[keys], on=keys)
    pa = pa.merge(common, on=keys).sort_values(keys).reset_index(drop=True)
    pb = pb.merge(common, on=keys).sort_values(keys).reset_index(drop=True)
    return paired_bootstrap_delta(pa, pb, model_a_name=a, model_b_name=b, labels=labels)


# ---------------------------------------------------------------- behaviour (H1-H3, H5)

def behaviour_models() -> pd.DataFrame:
    frames = []
    for window in ["pre90", "haz30"]:
        data = load_window(window)
        reference = (data["round2"] == 0) & (data["cd"] == 0)
        result = mixed_effects(data, BEHAVIOUR_MEASURES, LMM_RHS, LMM_TERMS, reference)
        result.insert(0, "window", window)
        frames.append(result)
    return pd.concat(frames)


def behaviour_robustness() -> pd.DataFrame:
    """Key LMM terms under alternative analysis choices (primary = pre90, DBM inclusion)."""
    variants = {
        "primary (pre90)": ("pre90", "primary", ""),
        "+ gaze validity covariate": ("pre90", "primary", " + beh__gaze_valid_frac"),
        "pre30 window": ("pre30", "primary", ""),
        "keep trips whose anomaly is outside the window": ("pre90", "window_overlap", ""),
    }
    frames = []
    for name, (window, inclusion, extra) in variants.items():
        data = load_window(window, inclusion)
        reference = (data["round2"] == 0) & (data["cd"] == 0)
        measures = [m for m in BEHAVIOUR_MEASURES if not (extra and m == "beh__gaze_valid_frac")]
        result = mixed_effects(data, measures, LMM_RHS + extra, KEY_TERMS, reference)
        result.insert(0, "variant", name)
        frames.append(result)

    # Reviewer checks (added 27 Oct 2026, see docs/HYPOTHESES.md): a per-participant random
    # slope for CD, and PRC around each driver's own baseline road centre.
    data = load_window("pre90")
    reference = (data["round2"] == 0) & (data["cd"] == 0)
    result = mixed_effects(data, BEHAVIOUR_MEASURES, LMM_RHS, KEY_TERMS, reference, re_formula="~cd")
    result.insert(0, "variant", "+ random slope for CD")
    frames.append(result)
    result = mixed_effects(data, ["beh__prc_driver"], LMM_RHS, KEY_TERMS, reference)
    result.insert(0, "variant", "PRC around per-driver centre")
    frames.append(result)
    return pd.concat(frames)


def hazard_models() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hazard response, valid hazards with an estimable onset (door-open 2b excluded)."""
    data = load_window("haz30")
    data = data[(data["hazard_valid"] == 1) & data["haz__onset_before_end_s"].notna()].reset_index(drop=True)
    rhs = LMM_RHS + " + haz__speed_at_onset"
    result = mixed_effects(data, HAZARD_MEASURES, rhs, LMM_TERMS, standardise_on=None)

    descriptive = []
    for label in LABEL_ORDER:
        subset = data[data["label"] == label]
        row = {"label": label, "n_trips": len(subset), "n_participants": subset["participant_name"].nunique()}
        for measure in ["haz__brake_rt_s", "haz__throttle_release_rt_s", "haz__braked", "haz__crash", "haz__peak_decel"]:
            row[f"{measure}__mean"] = float(subset[measure].mean())
        descriptive.append(row)
    return result, pd.DataFrame(descriptive)


def order_and_cohort_diagnostics(window: str = "pre90") -> pd.DataFrame:
    data = load_window(window)
    rows = []

    v2 = data[data["cohort_v1"] == 0].reset_index(drop=True)
    preds, _ = nested_cv(v2, TRI_COLUMNS, "round")
    rows.append(_ci_row("order_only: R2 vs R1 within control arm (v2)", preds, ["R1", "R2"], window=window))

    r1 = data[data["round2"] == 0].reset_index(drop=True)
    preds, _ = nested_cv(r1, TRI_COLUMNS, "cohort_label")
    rows.append(_ci_row("cohort_only: v1 vs v2 within R1 (no alcohol)", preds, ["v1", "v2"], window=window))

    v1 = data[data["cohort_v1"] == 1].reset_index(drop=True)
    preds, _ = nested_cv(v1, TRI_COLUMNS, "round")
    rows.append(_ci_row("alcohol arm: R2 (drunk) vs R1 (sober) within v1, TRI-style", preds, ["R1", "R2"], window=window))

    transfer = fit_and_apply(v2, v1, TRI_COLUMNS, "round")
    rows.append(_ci_row("transfer: order model trained on v2, applied to v1 R2 vs R1", transfer, ["R1", "R2"],
                        window=window))
    return pd.DataFrame(rows)


def tri_wilcoxon_replication() -> pd.DataFrame:
    """TRI's validation tests (dataset/analysis/feature_test.py) on the same pre90 window and
    trips, plus the same test applied to the control arm across rounds (order effect).

    TRI pairs participant means and skips a feature when a participant lacks a condition; here
    pairs are formed on participants present in both conditions (noted in the deviations log).
    Run on all study trips (matches the IDD paper, which did not drop anomaly trips) and on the
    primary trip set.
    """
    from scipy import stats

    rows = []
    for trip_set in ["all", "primary"]:
        data = load_window("pre90", trip_set)
        v1, v2 = data["cohort_v1"] == 1, data["cohort_v1"] == 0
        cd, r2, tox = data["cd"] == 1, data["round2"] == 1, data["intoxicated"]
        tests = {
            "TRI intoxicated: v1 no-CD, R2 (drunk) - R1 (sober)": (data[v1 & ~cd & ~r2], data[v1 & ~cd & r2]),
            "TRI cd: sober trips, CD - no-CD": (data[~tox & ~cd], data[~tox & cd]),
            "TRI joint: v1, both - neither": (data[v1 & ~cd & ~r2], data[v1 & cd & r2]),
            "order control: v2 no-CD, R2 - R1 (no alcohol)": (data[v2 & ~cd & ~r2], data[v2 & ~cd & r2]),
            "order control: v2 CD, R2 - R1 (no alcohol)": (data[v2 & cd & ~r2], data[v2 & cd & r2]),
        }
        for test, (base, cond) in tests.items():
            for column in TRI_COLUMNS:
                a = base.groupby("participant_name")[column].mean()
                b = cond.groupby("participant_name")[column].mean()
                paired = pd.concat([a, b], axis=1, keys=["base", "cond"]).dropna()
                delta = (paired["cond"] - paired["base"]).to_numpy()
                result = stats.wilcoxon(delta, method="approx")
                rows.append({
                    "trip_set": trip_set,
                    "test": test,
                    "feature": column.removeprefix("tri__"),
                    "n_participants": len(delta),
                    "mean_diff": float(delta.mean()),
                    "effect_r": float(abs(result.zstatistic) / np.sqrt(len(delta))),
                    "p_value": float(result.pvalue),
                })
    return pd.DataFrame(rows)


def session_timing() -> pd.DataFrame:
    """Break between the last R1 drive and the first R2 drive, per participant, by cohort."""
    trips = pd.read_pickle(PRIVATE / "trips.pkl")
    trips = trips[~trips["removed_participant"]]
    rows = []
    for participant, group in trips.groupby("participant_name"):
        r1, r2 = group[group["round"] == "R1"], group[group["round"] == "R2"]
        if r1.empty or r2.empty:
            continue
        rows.append({
            "participant_name": participant,
            "cohort": "v1 (alcohol arm)" if group["cohort_v1"].iloc[0] else "v2 (control arm)",
            "break_min": (r2["start_time"].min() - r1["end_time"].max()) / 60e9,
            "r1_driving_span_min": (r1["end_time"].max() - r1["start_time"].min()) / 60e9,
            "r2_driving_span_min": (r2["end_time"].max() - r2["start_time"].min()) / 60e9,
        })
    per_participant = pd.DataFrame(rows)
    summary = per_participant.groupby("cohort")[["break_min", "r1_driving_span_min", "r2_driving_span_min"]].describe(
        percentiles=[0.25, 0.5, 0.75]
    )
    summary.columns = [f"{a}__{b}" for a, b in summary.columns]
    return summary.reset_index()


# ---------------------------------------------------------------- classification (H4)

def classification_by_window() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, logs, deltas = [], [], []
    predictions = {}
    for window in WINDOWS:
        data = load_window(window)
        for task, (target, labels) in TASKS.items():
            preds, log = nested_cv(data, TRI_COLUMNS, target)
            preds.to_csv(PRIVATE / f"pred_{task}_{window}.csv", index=False)
            predictions[(task, window)] = preds
            log.insert(0, "window", window)
            log.insert(0, "task", task)
            logs.append(log)
            rows.append(_ci_row(task, preds, labels, task=task, window=window))
    for task, (_, labels) in TASKS.items():
        for a, b in [("haz30", "pre30"), ("pre90", "pre30"), ("haz30", "pre90")]:
            deltas.append({"task": task, **_paired(predictions[(task, a)], predictions[(task, b)], a, b, labels)})
    return pd.DataFrame(rows), pd.concat(logs), pd.DataFrame(deltas)


def classification_with_hazard_response() -> tuple[pd.DataFrame, pd.DataFrame]:
    """H4 with hazard-specific features, on trips whose hazard executed validly."""
    pre = load_window("pre30")
    haz = load_window("haz30")
    valid = haz.loc[haz["hazard_valid"] == 1, ["participant_name", "scenario_path"]]
    pre = pre.merge(valid, on=["participant_name", "scenario_path"]).reset_index(drop=True)
    haz = haz.merge(valid, on=["participant_name", "scenario_path"]).reset_index(drop=True)
    # carry the trip-level hazard features onto the pre30 rows for the combined model
    haz_cols = haz[["participant_name", "scenario_path"] + HAZARD_FEATURES]
    pre = pre.drop(columns=[c for c in HAZARD_FEATURES if c in pre.columns]).merge(
        haz_cols, on=["participant_name", "scenario_path"]
    )

    configs = {
        "pre30: TRI": (pre, TRI_COLUMNS),
        "haz30: TRI": (haz, TRI_COLUMNS),
        "haz30: hazard response only": (haz, HAZARD_FEATURES),
        "haz30: TRI + hazard response": (haz, TRI_COLUMNS + HAZARD_FEATURES),
        "pre30 TRI + hazard response": (pre, TRI_COLUMNS + HAZARD_FEATURES),
    }
    rows, deltas = [], []
    for task, (target, labels) in TASKS.items():
        preds = {}
        for name, (data, columns) in configs.items():
            preds[name], _ = nested_cv(data, columns, target)
            rows.append(_ci_row(name, preds[name], labels, task=task))
        for name in configs:
            if name != "pre30: TRI":
                deltas.append({"task": task, **_paired(preds[name], preds["pre30: TRI"], name, "pre30: TRI", labels)})
    return pd.DataFrame(rows), pd.DataFrame(deltas)


STAGES = ["behaviour", "robustness", "hazard", "tri", "timing", "diagnostics", "windows", "hazard_clf"]


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    stages = set(parser.parse_args().stages)
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)

    if "behaviour" in stages:
        behaviour_models().to_csv(RESULTS / "lmm_behaviour.csv", index=False)
    if "robustness" in stages:
        behaviour_robustness().to_csv(RESULTS / "lmm_behaviour_robustness.csv", index=False)
    if "hazard" in stages:
        hazard_lmm, hazard_desc = hazard_models()
        hazard_lmm.to_csv(RESULTS / "lmm_hazard_response.csv", index=False)
        hazard_desc.to_csv(RESULTS / "hazard_response_by_label.csv", index=False)
    if "tri" in stages:
        tri = tri_wilcoxon_replication()
        tri.to_csv(RESULTS / "tri_wilcoxon_replication.csv", index=False)
    if "timing" in stages:
        timing = session_timing()
        timing.to_csv(RESULTS / "session_timing_by_cohort.csv", index=False)
        print("== Session timing by cohort (minutes)")
        print(timing.round(1).to_string(index=False))
    if "diagnostics" in stages:
        order_and_cohort_diagnostics("pre90").to_csv(RESULTS / "order_cohort_diagnostics.csv", index=False)
    if "windows" in stages:
        summary, logs, deltas = classification_by_window()
        summary.to_csv(RESULTS / "classification_by_window.csv", index=False)
        logs.to_csv(RESULTS / "nested_cv_selection_log.csv", index=False)
        deltas.to_csv(RESULTS / "window_comparison_paired.csv", index=False)
    if "hazard_clf" in stages:
        hazard_clf, hazard_deltas = classification_with_hazard_response()
        hazard_clf.to_csv(RESULTS / "classification_hazard_response.csv", index=False)
        hazard_deltas.to_csv(RESULTS / "classification_hazard_response_paired.csv", index=False)


if __name__ == "__main__":
    main()
