"""Stage 2d: revision analyses (29 Oct 2026; see the deviations log in docs/HYPOTHESES.md).

    python -m scripts.study.run_revision [--boot 500] [--jobs 8]

Outputs go to artifacts/study/results/revision/. All analyses here were specified after the
earlier results were known and are reported as post hoc.

WP2  Exclusion audit: 416-trip flow, anomaly timing, behaviour-related events (not excluded).
WP3  Full factorial model (main): z ~ cohort * R2 * CD + route + (1 | participant), with named
     estimands, raw and SD units, Holm per estimand, diagnostics, inclusion / gaze-quality
     sensitivity, participant-cluster bootstrap, constrained-model comparison, task model,
     and naive-vs-full differences with paired participant bootstrap.
WP4  Cohort false alarms: denominators, trip- and driver-weighted rates, composition-standardised
     gaps, within-cohort bootstrap, Holm across models.
WP5  Hazard audit: sample flow, onset availability, observable post-onset time.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from joblib import Parallel, delayed
from patsy import dmatrices

from scripts.study.run_analyses import BEHAVIOUR_MEASURES, PRIVATE, RESULTS, load_window
from src.study.models import holm

OUT = RESULTS / "revision"
SEED = 20261029
GAZE_MEASURES = ["beh__gaze_yaw_sd", "beh__gaze_pitch_sd", "beh__prc", "beh__fixation_rate", "beh__saccade_rate",
                 "beh__pupil_mean", "beh__gaze_valid_frac"]

LOG_MEASURES = ["beh__gaze_yaw_sd", "beh__gaze_pitch_sd", "beh__steer_hf_sd_straight", "beh__lateral_dev_sd_straight",
                "beh__steer_angle_sd", "beh__lat_accel_sd", "beh__speed_sd"]

FULL_FORMULA = "z ~ cohort_v1 * round2 * cd + C(scenario_code)"
CONSTRAINED_FORMULA = "z ~ cohort_v1 + round2 + alcohol + cd + alcohol:cd + C(scenario_code)"
TASK_FORMULA = "z ~ cohort_v1 * round2 * (nback + sentence) + C(scenario_code)"
NAIVE_FORMULA = "z ~ round2 * cd + C(scenario_code)"

# Estimands as {fixed-effect name: weight}. CD effects among sober drivers average the three sober
# cells (control R1, control R2, alcohol cohort R1) with equal weight.
FULL_ESTIMANDS = {
    "alcohol_noCD": {"cohort_v1:round2": 1.0},
    "alcohol_CD": {"cohort_v1:round2": 1.0, "cohort_v1:round2:cd": 1.0},
    "alcohol_x_CD": {"cohort_v1:round2:cd": 1.0},
    "CD_sober": {"cd": 1.0, "cohort_v1:cd": 1 / 3, "round2:cd": 1 / 3},
    "order_controls_noCD": {"round2": 1.0},
    "cohort_R1_noCD": {"cohort_v1": 1.0},
}
CONSTRAINED_ESTIMANDS = {"alcohol_noCD": {"alcohol": 1.0}, "alcohol_x_CD": {"alcohol:cd": 1.0},
                         "CD_sober": {"cd": 1.0}, "order_controls_noCD": {"round2": 1.0}}
TASK_ESTIMANDS = {
    "nback_sober": {"nback": 1.0, "cohort_v1:nback": 1 / 3, "round2:nback": 1 / 3},
    "sentence_sober": {"sentence": 1.0, "cohort_v1:sentence": 1 / 3, "round2:sentence": 1 / 3},
    "nback_minus_sentence_sober": {"nback": 1.0, "cohort_v1:nback": 1 / 3, "round2:nback": 1 / 3,
                                   "sentence": -1.0, "cohort_v1:sentence": -1 / 3, "round2:sentence": -1 / 3},
    "alcohol_nback": {"cohort_v1:round2": 1.0, "cohort_v1:round2:nback": 1.0},
    "alcohol_sentence": {"cohort_v1:round2": 1.0, "cohort_v1:round2:sentence": 1.0},
}


# ---------------------------------------------------------------- data
def analysis_data(inclusion: str = "primary") -> pd.DataFrame:
    data = load_window("pre90", "primary" if inclusion == "complete_case" else inclusion)
    if inclusion == "complete_case":
        trips = pd.read_pickle(PRIVATE / "trips.pkl")
        study = trips[~trips["removed_participant"]]
        keep = study.groupby("participant_name")["included_primary"].all()
        data = data[data["participant_name"].isin(keep[keep].index)].reset_index(drop=True)
    data["nback"] = (data["cognitive_task"] == "nback_task").astype(int)
    data["sentence"] = (data["cognitive_task"] == "statement_task").astype(int)
    return data


def reference_stats(data: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Mean and SD of sober R1 no-task trips in the primary sample; reused by every variant so
    that all estimates share units."""
    ref = data[(data["round2"] == 0) & (data["cd"] == 0)]
    return {m: (float(ref[m].mean()), float(ref[m].std())) for m in BEHAVIOUR_MEASURES}


def with_z(data: pd.DataFrame, measure: str, ref: dict) -> pd.DataFrame:
    mu, sd = ref[measure]
    return data.assign(z=(data[measure] - mu) / sd).dropna(subset=["z"])


# ---------------------------------------------------------------- model helpers
def fit_lmm(formula: str, frame: pd.DataFrame, re_formula: str | None = None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm(formula, frame, groups=frame["participant_name"], re_formula=re_formula)
        fit = model.fit(reml=True)
        optimiser = "lbfgs"
        if not fit.converged:
            fit = model.fit(reml=True, method="powell")
            optimiser = "powell"
    return fit, optimiser


def contrast(fit, weights: dict[str, float]):
    names = list(fit.fe_params.index)
    missing = [k for k in weights if k not in names]
    if missing:
        raise KeyError(f"terms not in model: {missing}")
    vec = np.zeros(len(names))
    for k, w in weights.items():
        vec[names.index(k)] = w
    est = float(vec @ fit.fe_params.to_numpy())
    cov = fit.cov_params().iloc[: len(names), : len(names)].to_numpy()
    se = float(np.sqrt(vec @ cov @ vec))
    z = est / se
    from scipy.stats import norm

    return est, se, est - 1.959964 * se, est + 1.959964 * se, float(2 * norm.sf(abs(z)))


def estimate_table(data, ref, formula, estimands, measures=BEHAVIOUR_MEASURES, label="", re_formula=None):
    rows, diag = [], []
    for m in measures:
        frame = with_z(data, m, ref)
        fit, optimiser = fit_lmm(formula, frame, re_formula)
        for name, weights in estimands.items():
            est, se, lo, hi, p = contrast(fit, weights)
            sd = ref[m][1]
            rows.append({"analysis": label, "measure": m, "estimand": name, "estimate_sd": est, "se_sd": se,
                         "ci_low_sd": lo, "ci_high_sd": hi, "p_value": p, "estimate_raw": est * sd,
                         "ci_low_raw": lo * sd, "ci_high_raw": hi * sd, "n_trips": int(fit.nobs),
                         "n_drivers": int(frame["participant_name"].nunique())})
        resid = np.asarray(fit.resid)
        diag.append({"analysis": label, "measure": m, "optimiser": optimiser, "converged": bool(fit.converged),
                     "var_participant": float(fit.cov_re.iloc[0, 0]), "var_residual": float(fit.scale),
                     "boundary_variance": bool(fit.cov_re.iloc[0, 0] < 1e-4),
                     "resid_skew": float(pd.Series(resid).skew()), "resid_excess_kurtosis": float(pd.Series(resid).kurt()),
                     "max_abs_std_resid": float(np.max(np.abs(resid / np.sqrt(fit.scale)))),
                     "n_trips": int(fit.nobs), "n_drivers": int(frame["participant_name"].nunique())})
    out = pd.DataFrame(rows)
    out["p_holm"] = out.groupby("estimand")["p_value"].transform(holm)
    return out, pd.DataFrame(diag)


# ---------------------------------------------------------------- WP2 exclusion audit
def exclusion_audit() -> dict[str, pd.DataFrame]:
    trips = pd.read_pickle(PRIVATE / "trips.pkl")
    feats = pd.read_pickle(PRIVATE / "window_features.pkl")
    s = trips[~trips["removed_participant"]].copy()
    s["cohort"] = np.where(s["cohort_v1"], "Alcohol cohort", "Control cohort")
    s["task"] = s["cognitive_task"].map({"no_task": "No task", "nback_task": "1-back", "statement_task": "Sentence"})
    s["reason"] = np.where(s["included_primary"], "Included",
                           np.where(s["pre90_anomaly_overlap"], "Anomaly inside pre90",
                                    "Anomaly outside pre90 (whole trip excluded by default rule)"))

    def _in_window(ranges, start, end):
        return isinstance(ranges, list) and any(lo * 1e9 <= end and hi * 1e9 >= start for lo, hi in ranges)

    def _times_in(times, start, end):
        return isinstance(times, list) and any(start <= t * 1e9 <= end for t in times)

    s["off_course_in_pre90"] = [_in_window(r, a, b) for r, a, b in zip(s["off_course_ranges_s"], s["pre90_start_ns"], s["pre90_end_ns"])]
    s["crash_in_pre90"] = [_times_in(c, a, b) for c, a, b in zip(s["crash_times_s"], s["pre90_start_ns"], s["pre90_end_ns"])]
    g = feats[feats["window"] == "pre90"][["participant_name", "scenario_path", "beh__gaze_valid_frac"]]
    s = s.merge(g, on=["participant_name", "scenario_path"], how="left")
    manifest_cols = ["participant_name", "cohort", "round", "task", "label", "scenario_code", "duration_s", "fixed_fold",
                     "anomaly_ranges_s", "pre90_anomaly_overlap", "haz30_anomaly_overlap", "hazard_outcome", "hazard_valid",
                     "off_course_in_pre90", "crash_in_pre90", "beh__gaze_valid_frac", "included_primary", "reason"]
    s[manifest_cols].to_csv(PRIVATE / "trip_manifest_416.csv", index=False)

    tables = {
        "flow_by_cohort_part_reason": pd.crosstab([s["cohort"], s["round"]], s["reason"], margins=True),
        "flow_by_cohort_part_task_included": pd.crosstab([s["cohort"], s["round"], s["task"]], s["included_primary"]),
        "exclusions_by_route": pd.crosstab(s.loc[~s["included_primary"], "scenario_code"],
                                           [s.loc[~s["included_primary"], "cohort"], s.loc[~s["included_primary"], "round"]]),
        "behaviour_events_included_trips": s[s["included_primary"]].groupby(["cohort", "round", "label"])[
            ["off_course_in_pre90", "crash_in_pre90"]].sum().astype(int),
    }
    per = s.groupby(["participant_name", "cohort"])["included_primary"].agg(["sum", "count"]).reset_index()
    per["excluded"] = per["count"] - per["sum"]
    tables["drivers_by_number_excluded"] = pd.crosstab(per["cohort"], per["excluded"])
    cells = s[s["included_primary"]].groupby(["cohort", "round", "cognitive_task"])["participant_name"].nunique()
    tables["drivers_contributing_per_cell"] = cells.unstack()
    route_cells = s[s["included_primary"]]
    tables["route_by_condition_included"] = pd.crosstab(route_cells["scenario_code"],
                                                        [route_cells["cohort"], route_cells["round"], route_cells["task"]])
    return tables


# ---------------------------------------------------------------- WP3 bootstrap
def _boot_rep(data, ref, measures, estimands, formula, rep_seed):
    rng = np.random.default_rng(rep_seed)
    parts = []
    for cohort in [0, 1]:
        ids = data.loc[data["cohort_v1"] == cohort, "participant_name"].unique()
        for k, pid in enumerate(rng.choice(ids, size=len(ids), replace=True)):
            parts.append(data[data["participant_name"] == pid].assign(participant_name=f"{pid}#{cohort}_{k}"))
    sample = pd.concat(parts, ignore_index=True)
    out = {}
    for m in measures:
        frame = with_z(sample, m, ref)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fit = smf.mixedlm(formula, frame, groups=frame["participant_name"]).fit(reml=True)
            except Exception:  # noqa: BLE001 - a singular resample is skipped, not fatal
                continue
        for name, weights in estimands.items():
            out[(m, name)] = float(sum(w * fit.fe_params[k] for k, w in weights.items()))
    return out


def cluster_bootstrap(data, ref, formula, estimands, n_boot, jobs, label):
    reps = Parallel(n_jobs=jobs)(delayed(_boot_rep)(data, ref, BEHAVIOUR_MEASURES, estimands, formula, SEED + b)
                                 for b in range(n_boot))
    rows = []
    for m in BEHAVIOUR_MEASURES:
        for name in estimands:
            vals = np.array([r[(m, name)] for r in reps if (m, name) in r])
            rows.append({"analysis": label, "measure": m, "estimand": name, "boot_n": len(vals),
                         "boot_ci_low_sd": np.quantile(vals, 0.025), "boot_ci_high_sd": np.quantile(vals, 0.975),
                         "boot_p_two_sided": min(1.0, 2 * min((vals <= 0).mean(), (vals >= 0).mean()))})
    return pd.DataFrame(rows)


def _naive_vs_full_rep(data, ref, rep_seed):
    rng = np.random.default_rng(rep_seed)
    parts = []
    for cohort in [0, 1]:
        ids = data.loc[data["cohort_v1"] == cohort, "participant_name"].unique()
        for k, pid in enumerate(rng.choice(ids, size=len(ids), replace=True)):
            parts.append(data[data["participant_name"] == pid].assign(participant_name=f"{pid}#{cohort}_{k}"))
    sample = pd.concat(parts, ignore_index=True)
    out = {}
    for m in BEHAVIOUR_MEASURES:
        frame = with_z(sample, m, ref)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                full = smf.mixedlm(FULL_FORMULA, frame, groups=frame["participant_name"]).fit(reml=True)
                v1 = frame[frame["cohort_v1"] == 1]
                naive = smf.mixedlm(NAIVE_FORMULA, v1, groups=v1["participant_name"]).fit(reml=True)
            except Exception:  # noqa: BLE001
                continue
        out[m] = (float(naive.fe_params["round2"]), float(full.fe_params["cohort_v1:round2"]))
    return out


def naive_vs_full(data, ref, n_boot, jobs) -> pd.DataFrame:
    point = {}
    for m in BEHAVIOUR_MEASURES:
        frame = with_z(data, m, ref)
        full, _ = fit_lmm(FULL_FORMULA, frame)
        v1 = frame[frame["cohort_v1"] == 1]
        naive, _ = fit_lmm(NAIVE_FORMULA, v1)
        nci = naive.conf_int().loc["round2"]
        fci = full.conf_int().loc["cohort_v1:round2"]
        point[m] = dict(naive=naive.fe_params["round2"], naive_lo=nci[0], naive_hi=nci[1], naive_p=naive.pvalues["round2"],
                        full=full.fe_params["cohort_v1:round2"], full_lo=fci[0], full_hi=fci[1],
                        full_p=full.pvalues["cohort_v1:round2"])
    reps = Parallel(n_jobs=jobs)(delayed(_naive_vs_full_rep)(data, ref, SEED + 10_000 + b) for b in range(n_boot))
    rows = []
    for m in BEHAVIOUR_MEASURES:
        diffs = np.array([r[m][0] - r[m][1] for r in reps if m in r])
        p = point[m]
        rows.append({"measure": m, "naive_sd": p["naive"], "naive_ci_low": p["naive_lo"], "naive_ci_high": p["naive_hi"],
                     "naive_p": p["naive_p"], "full_sd": p["full"], "full_ci_low": p["full_lo"], "full_ci_high": p["full_hi"],
                     "full_p": p["full_p"], "difference_sd": p["naive"] - p["full"],
                     "diff_boot_ci_low": np.quantile(diffs, 0.025), "diff_boot_ci_high": np.quantile(diffs, 0.975),
                     "diff_boot_p": min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())), "boot_n": len(diffs)})
    out = pd.DataFrame(rows)
    out["naive_p_holm"] = holm(out["naive_p"])
    out["full_p_holm"] = holm(out["full_p"])
    out["diff_p_holm"] = holm(out["diff_boot_p"])
    return out


def design_checks(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, formula in [("full", FULL_FORMULA), ("task", TASK_FORMULA), ("constrained", CONSTRAINED_FORMULA)]:
        frame = with_z(data, "beh__gaze_yaw_sd", reference_stats(data))
        _, X = dmatrices(formula, frame, return_type="dataframe")
        rows.append({"model": name, "n_columns": X.shape[1], "rank": int(np.linalg.matrix_rank(X.to_numpy())),
                     "condition_number": float(np.linalg.cond(X.to_numpy()))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- WP4 cohort false alarms
MODEL_FILES = {"Logistic regression": "logreg", "Linear SVM": "linear_svm", "Gradient boosting": "gbm",
               "Temporal transformer": "transformer"}
ALCOHOL = ["alcohol_intoxication", "combined"]


def _sober_predictions(model_key: str) -> pd.DataFrame:
    trips = pd.read_pickle(PRIVATE / "trips.pkl")[["participant_name", "scenario_path", "cohort_v1", "round",
                                                    "cognitive_task", "scenario_code"]]
    preds = pd.read_csv(PRIVATE / f"pred_model_{model_key}_pre90.csv").merge(trips, on=["participant_name", "scenario_path"])
    preds["flag"] = preds["pred_label"].isin(ALCOHOL).astype(float)
    return preds[~preds["true_label"].isin(ALCOHOL)].copy()


def _rates(frame: pd.DataFrame, weights_by_task: dict | None = None):
    """Trip-weighted, driver-weighted, and (optionally) task-standardised false-alarm rates."""
    trip = frame["flag"].mean()
    driver = frame.groupby("participant_name")["flag"].mean().mean()
    std = np.nan
    if weights_by_task is not None:
        by_task = frame.groupby("cognitive_task")["flag"].mean()
        if set(weights_by_task) <= set(by_task.index):
            std = float(sum(w * by_task[t] for t, w in weights_by_task.items()))
    return trip, driver, std


def cohort_false_alarms(n_boot: int) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    summary, counts = [], []
    for model, key in MODEL_FILES.items():
        sober = _sober_predictions(key)
        alc_r1 = sober[(sober["cohort_v1"]) & (sober["round"] == "R1")]
        ctl_r1 = sober[(~sober["cohort_v1"]) & (sober["round"] == "R1")]
        ctl_r2 = sober[(~sober["cohort_v1"]) & (sober["round"] == "R2")]
        task_w = ctl_r1["cognitive_task"].value_counts(normalize=True).to_dict()
        route_w = ctl_r1["scenario_code"].value_counts(normalize=True).to_dict()
        for cell, frame in [("Alcohol cohort R1", alc_r1), ("Control cohort R1", ctl_r1), ("Control cohort R2", ctl_r2)]:
            for (task, route), g in frame.groupby(["cognitive_task", "scenario_code"]):
                counts.append({"model": model, "cell": cell, "task": task, "route": route, "n_trips": len(g),
                               "n_drivers": g["participant_name"].nunique(), "n_flagged": int(g["flag"].sum())})

        def stats(a, c):
            ta, da, sa = _rates(a, task_w)
            tc, dc, _ = _rates(c)
            by_route = a.groupby("scenario_code")["flag"].mean()
            ra = float(sum(w * by_route[r] for r, w in route_w.items())) if set(route_w) <= set(by_route.index) else np.nan
            return {"trip": ta - tc, "driver": da - dc, "task_std": sa - tc, "route_std": ra - tc}

        point = stats(alc_r1, ctl_r1)
        ids_a, ids_c = alc_r1["participant_name"].unique(), ctl_r1["participant_name"].unique()
        ga, gc = dict(tuple(alc_r1.groupby("participant_name"))), dict(tuple(ctl_r1.groupby("participant_name")))
        boots = {k: [] for k in point}
        for _ in range(n_boot):
            a = pd.concat([ga[p] for p in rng.choice(ids_a, len(ids_a))])
            c = pd.concat([gc[p] for p in rng.choice(ids_c, len(ids_c))])
            for k, v in stats(a, c).items():
                boots[k].append(v)
        for weighting, est in point.items():
            vals = np.array(boots[weighting], dtype=float)
            vals = vals[~np.isnan(vals)]
            summary.append({"model": model, "quantity": f"cohort gap R1 ({weighting})", "estimate": est,
                            "ci_low": np.quantile(vals, .025), "ci_high": np.quantile(vals, .975),
                            "p_value": min(1.0, 2 * min((vals <= 0).mean(), (vals >= 0).mean())), "boot_n": len(vals)})
        for cell, frame in [("Alcohol cohort R1", alc_r1), ("Control cohort R1", ctl_r1), ("Control cohort R2", ctl_r2)]:
            t, d, _ = _rates(frame)
            per = frame.groupby("participant_name")["flag"].agg(["sum", "count"]).to_numpy()
            rb = [per[idx, 0].sum() / per[idx, 1].sum() for idx in (rng.integers(0, len(per), len(per)) for _ in range(n_boot))]
            summary.append({"model": model, "quantity": f"rate {cell} (trip-weighted)", "estimate": t,
                            "ci_low": np.quantile(rb, .025), "ci_high": np.quantile(rb, .975),
                            "n_trips": len(frame), "n_drivers": frame["participant_name"].nunique()})
            summary.append({"model": model, "quantity": f"rate {cell} (driver-weighted)", "estimate": d,
                            "n_trips": len(frame), "n_drivers": frame["participant_name"].nunique()})
        ctl = sober[~sober["cohort_v1"]].groupby(["participant_name", "round"])["flag"].mean().unstack().dropna()
        delta = (ctl["R2"] - ctl["R1"]).to_numpy()
        bd = np.array([delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(n_boot)])
        summary.append({"model": model, "quantity": "control R2 - R1 (paired drivers)", "estimate": float(delta.mean()),
                        "ci_low": np.quantile(bd, .025), "ci_high": np.quantile(bd, .975),
                        "p_value": min(1.0, 2 * min((bd <= 0).mean(), (bd >= 0).mean())), "n_drivers": len(delta)})
    out = pd.DataFrame(summary)
    for weighting in ["trip", "driver", "task_std", "route_std"]:
        mask = out["quantity"] == f"cohort gap R1 ({weighting})"
        out.loc[mask, "p_holm_across_models"] = holm(out.loc[mask, "p_value"]).to_numpy()
    mask = out["quantity"] == "control R2 - R1 (paired drivers)"
    out.loc[mask, "p_holm_across_models"] = holm(out.loc[mask, "p_value"]).to_numpy()
    return {"cohort_false_alarms_v3": out, "cohort_false_alarm_counts": pd.DataFrame(counts)}


# ---------------------------------------------------------------- WP5 hazard audit
def hazard_audit() -> dict[str, pd.DataFrame]:
    h = load_window("haz30")
    h["valid"] = h["hazard_valid"] == 1
    h["onset"] = h["haz__onset_before_end_s"].notna()
    flow = pd.DataFrame([
        {"step": "Analysed trips", "n_trips": len(h), "n_drivers": h["participant_name"].nunique()},
        {"step": "Hazard executed as scripted (annotation)", "n_trips": int(h["valid"].sum()),
         "n_drivers": h.loc[h["valid"], "participant_name"].nunique()},
        {"step": "Hazard onset estimable (actor motion; door-open route excluded)", "n_trips": int((h["valid"] & h["onset"]).sum()),
         "n_drivers": h.loc[h["valid"] & h["onset"], "participant_name"].nunique()},
    ])
    timed = h[h["valid"] & h["onset"]]
    availability = timed.groupby("scenario_code").agg(
        n_trips=("onset", "size"), observable_after_onset_min_s=("haz__onset_before_end_s", "min"),
        observable_after_onset_median_s=("haz__onset_before_end_s", "median"),
        braking_at_onset=("haz__braking_at_onset", "sum"),
        brake_response_within_6s=("haz__brake_rt_s", lambda x: int(x.notna().sum())),
        throttle_release_within_6s=("haz__throttle_release_rt_s", lambda x: int(x.notna().sum())))
    excluded_types = h[~h["valid"]].groupby("scenario_code")["hazard_outcome"].agg(lambda x: "; ".join(sorted(set(x))))
    return {"hazard_flow": flow, "hazard_availability_by_route": availability,
            "hazard_invalid_by_route": excluded_types.to_frame("annotation")}


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boot", type=int, default=500)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--skip-bootstrap", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    for name, table in exclusion_audit().items():
        table.to_csv(OUT / f"{name}.csv")
    for name, table in hazard_audit().items():
        table.to_csv(OUT / f"{name}.csv")

    primary = analysis_data("primary")
    ref = reference_stats(primary)
    pd.DataFrame([{"measure": m, "ref_mean": v[0], "ref_sd": v[1],
                   "n_ref_trips": int(((primary["round2"] == 0) & (primary["cd"] == 0) & primary[m].notna()).sum())}
                  for m, v in ref.items()]).to_csv(OUT / "standardisation_reference.csv", index=False)
    design_checks(primary).to_csv(OUT / "design_checks.csv", index=False)

    main_est, main_diag = estimate_table(primary, ref, FULL_FORMULA, FULL_ESTIMANDS, label="full (main)")
    main_est.to_csv(OUT / "full_model_estimands.csv", index=False)
    main_diag.to_csv(OUT / "full_model_diagnostics.csv", index=False)
    con_est, con_diag = estimate_table(primary, ref, CONSTRAINED_FORMULA, CONSTRAINED_ESTIMANDS, label="constrained (v2)")
    task_est, task_diag = estimate_table(primary, ref, TASK_FORMULA, TASK_ESTIMANDS, label="task (full)")
    task_est.to_csv(OUT / "task_model_estimands.csv", index=False)

    sens = [con_est]
    diags = [main_diag, con_diag, task_diag]
    for inclusion, label in [("window_overlap", "inclusion: anomaly outside window kept"),
                             ("all", "inclusion: all 416 trips"), ("complete_case", "inclusion: complete-case drivers")]:
        data = analysis_data(inclusion)
        est, dg = estimate_table(data, ref, FULL_FORMULA, FULL_ESTIMANDS, label=label)
        sens.append(est)
        diags.append(dg)
    for threshold in [0.75]:
        data = primary[primary["beh__gaze_valid_frac"] >= threshold].reset_index(drop=True)
        est, dg = estimate_table(data, ref, FULL_FORMULA, FULL_ESTIMANDS, label=f"gaze validity >= {threshold}")
        sens.append(est)
        diags.append(dg)
    covariate_formula = FULL_FORMULA + " + beh__gaze_valid_frac"
    est, dg = estimate_table(primary, ref, covariate_formula, FULL_ESTIMANDS,
                             measures=[m for m in BEHAVIOUR_MEASURES if m != "beh__gaze_valid_frac"],
                             label="+ gaze validity covariate")
    sens.append(est)
    diags.append(dg)
    # Heavy-tailed dispersion outcomes: refit on the log scale (each re-standardised on its own
    # sober R1 no-task reference), so estimates are SD units of log(outcome).
    logged = primary.copy()
    for m in LOG_MEASURES:
        logged[m] = np.log(logged[m])
    est, dg = estimate_table(logged, reference_stats(logged), FULL_FORMULA, FULL_ESTIMANDS, measures=LOG_MEASURES,
                             label="log scale (dispersion outcomes)")
    sens.append(est)
    diags.append(dg)
    # Participant random slopes for CD and session part (random intercept only in the main model).
    est, dg = estimate_table(primary, ref, FULL_FORMULA, FULL_ESTIMANDS, label="random slopes (CD, session part)",
                             re_formula="~ cd + round2")
    sens.append(est)
    diags.append(dg)
    # MGC around each driver's own sober R1 no-task gaze centre instead of the window median.
    centred = primary.copy()
    centred["beh__prc"] = centred["beh__prc_driver"]
    est, dg = estimate_table(centred, reference_stats(centred), FULL_FORMULA, FULL_ESTIMANDS, measures=["beh__prc"],
                             label="MGC around per-driver baseline centre")
    sens.append(est)
    diags.append(dg)
    pd.concat(sens).to_csv(OUT / "full_model_sensitivity.csv", index=False)
    pd.concat(diags).to_csv(OUT / "model_diagnostics_all.csv", index=False)

    for name, table in cohort_false_alarms(4000).items():
        table.to_csv(OUT / f"{name}.csv", index=False)

    if not args.skip_bootstrap:
        cluster_bootstrap(primary, ref, FULL_FORMULA, FULL_ESTIMANDS, args.boot, args.jobs, "full (main)").to_csv(
            OUT / "full_model_bootstrap.csv", index=False)
        cluster_bootstrap(primary, ref, TASK_FORMULA, TASK_ESTIMANDS, args.boot, args.jobs, "task (full)").to_csv(
            OUT / "task_model_bootstrap.csv", index=False)
        naive_vs_full(primary, ref, args.boot, args.jobs).to_csv(OUT / "naive_vs_full.csv", index=False)

    (OUT / "run_settings.json").write_text(json.dumps({"seed": SEED, "bootstrap_reps": args.boot,
                                                       "false_alarm_bootstrap_reps": 4000}, indent=1))
    print("done")


if __name__ == "__main__":
    main()
