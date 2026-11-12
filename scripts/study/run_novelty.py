"""Stage 2c: post hoc analyses added on 27-28 Oct 2026 (see docs/HYPOTHESES.md).

    python -m scripts.study.run_novelty

1. Cohort shortcut: how often each classifier flags *sober* trips as alcohol, by cohort.
2. Naive vs order-corrected alcohol estimates (within-alcohol-arm R2-R1 vs difference in change).
3. Task-specific CD effects: 1-back vs sentence comprehension.
4. Full cohort x session x CD model (the primary model omits cohort:CD and session:CD).
5. Hazard collisions with a participant-clustered logistic GEE.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from scripts.study.run_analyses import BEHAVIOUR_MEASURES, PRIVATE, RESULTS, load_window

MODELS = {"Logistic regression": "logreg", "Linear SVM": "linear_svm", "Gradient boosting": "gbm",
          "Temporal transformer": "transformer"}
ALCOHOL_LABELS = ["alcohol_intoxication", "combined"]


def _rate(counts: np.ndarray) -> float:
    return counts[:, 0].sum() / counts[:, 1].sum()


def cohort_false_alarms(n_boot: int = 4000, seed: int = 42) -> pd.DataFrame:
    """Share of sober trips predicted as alcohol/combined, by cohort and part, per model."""
    trips = pd.read_pickle(PRIVATE / "trips.pkl")[["participant_name", "scenario_path", "cohort_v1", "round"]]
    rng = np.random.default_rng(seed)
    rows = []
    for model, key in MODELS.items():
        preds = pd.read_csv(PRIVATE / f"pred_model_{key}_pre90.csv").merge(trips, on=["participant_name", "scenario_path"])
        preds["flag"] = preds["pred_label"].isin(ALCOHOL_LABELS)
        sober = preds[~preds["true_label"].isin(ALCOHOL_LABELS)]
        groups = {
            ("Alcohol cohort", "R1"): sober[sober["cohort_v1"] & (sober["round"] == "R1")],
            ("Control cohort", "R1"): sober[~sober["cohort_v1"] & (sober["round"] == "R1")],
            ("Control cohort", "R2"): sober[~sober["cohort_v1"] & (sober["round"] == "R2")],
        }
        counts = {k: g.groupby("participant_name")["flag"].agg(["sum", "count"]).to_numpy() for k, g in groups.items()}
        for (cohort, part), c in counts.items():
            boots = [_rate(c[rng.integers(0, len(c), len(c))]) for _ in range(n_boot)]
            rows.append({"model": model, "comparison": "rate", "cohort": cohort, "part": part, "estimate": _rate(c),
                         "ci_low": np.quantile(boots, .025), "ci_high": np.quantile(boots, .975),
                         "n_trips": int(c[:, 1].sum()), "n_drivers": len(c), "p_value": np.nan})
        a, b = counts[("Alcohol cohort", "R1")], counts[("Control cohort", "R1")]
        diffs = np.array([_rate(a[rng.integers(0, len(a), len(a))]) - _rate(b[rng.integers(0, len(b), len(b))])
                          for _ in range(n_boot)])
        rows.append({"model": model, "comparison": "cohort gap (R1, both sober)", "cohort": "Alcohol − control",
                     "part": "R1", "estimate": _rate(a) - _rate(b), "ci_low": np.quantile(diffs, .025),
                     "ci_high": np.quantile(diffs, .975), "n_trips": int(a[:, 1].sum() + b[:, 1].sum()),
                     "n_drivers": len(a) + len(b),
                     "p_value": min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))})
        ctrl = sober[~sober["cohort_v1"]].groupby(["participant_name", "round"])["flag"].mean().unstack().dropna()
        delta = (ctrl["R2"] - ctrl["R1"]).to_numpy()
        boots = np.array([delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(n_boot)])
        rows.append({"model": model, "comparison": "order change (controls, R2 − R1)", "cohort": "Control cohort",
                     "part": "R2 − R1", "estimate": delta.mean(), "ci_low": np.quantile(boots, .025),
                     "ci_high": np.quantile(boots, .975), "n_trips": np.nan, "n_drivers": len(delta),
                     "p_value": min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))})
    return pd.DataFrame(rows)


def _standardised(data: pd.DataFrame, measure: str) -> pd.Series:
    ref = data.loc[(data["round2"] == 0) & (data["cd"] == 0), measure]
    return (data[measure] - ref.mean()) / ref.std()


def _fit(formula: str, frame: pd.DataFrame):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return smf.mixedlm(formula, frame, groups=frame["participant_name"]).fit(reml=True)


def naive_vs_corrected() -> pd.DataFrame:
    """Alcohol estimate from the alcohol arm alone (R2 vs R1, as in a within-arm design) versus
    the order-corrected difference in change; both in the same baseline-SD units."""
    data = load_window("pre90")
    rows = []
    for measure in BEHAVIOUR_MEASURES:
        frame = data.assign(z=_standardised(data, measure)).dropna(subset=["z"])
        naive = _fit("z ~ round2 + cd + round2:cd + C(scenario_code)", frame[frame["cohort_v1"] == 1])
        corrected = _fit("z ~ cohort_v1 + round2 + alcohol + cd + alcohol:cd + C(scenario_code)", frame)
        for label, fit, term in [("naive (within alcohol arm)", naive, "round2"), ("order-corrected", corrected, "alcohol")]:
            ci = fit.conf_int().loc[term]
            rows.append({"measure": measure, "estimate_type": label, "estimate_sd": fit.params[term],
                         "ci_low": ci[0], "ci_high": ci[1], "p_value": fit.pvalues[term]})
    out = pd.DataFrame(rows)
    out["p_holm"] = out.groupby("estimate_type")["p_value"].transform(_holm)
    return out


def task_specific() -> pd.DataFrame:
    """CD split into 1-back and sentence comprehension, with their difference."""
    data = load_window("pre90")
    data["nback"] = (data["cognitive_task"] == "nback_task").astype(int)
    data["sentence"] = (data["cognitive_task"] == "statement_task").astype(int)
    rows = []
    for measure in BEHAVIOUR_MEASURES:
        frame = data.assign(z=_standardised(data, measure)).dropna(subset=["z"])
        fit = _fit("z ~ cohort_v1 + round2 + alcohol + nback + sentence + alcohol:nback + alcohol:sentence"
                   " + C(scenario_code)", frame)
        for term in ["nback", "sentence"]:
            ci = fit.conf_int().loc[term]
            rows.append({"measure": measure, "term": term, "estimate_sd": fit.params[term], "ci_low": ci[0],
                         "ci_high": ci[1], "p_value": fit.pvalues[term]})
        names = list(fit.fe_params.index)
        contrast = np.zeros((1, len(names)))
        contrast[0, names.index("nback")], contrast[0, names.index("sentence")] = 1.0, -1.0
        test = fit.t_test(contrast)
        ci = np.asarray(test.conf_int()).ravel()
        rows.append({"measure": measure, "term": "nback - sentence", "estimate_sd": float(np.asarray(test.effect).ravel()[0]),
                     "ci_low": ci[0], "ci_high": ci[1], "p_value": float(np.asarray(test.pvalue).ravel()[0])})
    out = pd.DataFrame(rows)
    out["p_holm"] = out.groupby("term")["p_value"].transform(_holm)
    return out


def full_model() -> pd.DataFrame:
    """cohort x session x CD with all lower-order terms; compare alcohol and interaction terms."""
    data = load_window("pre90")
    rows = []
    for measure in BEHAVIOUR_MEASURES:
        frame = data.assign(z=_standardised(data, measure)).dropna(subset=["z"])
        fit = _fit("z ~ cohort_v1 * round2 * cd + C(scenario_code)", frame)
        for term, label in [("cohort_v1:round2", "alcohol (at CD = 0)"), ("cohort_v1:round2:cd", "alcohol × CD"),
                            ("cohort_v1:cd", "cohort × CD"), ("round2:cd", "session × CD")]:
            ci = fit.conf_int().loc[term]
            rows.append({"measure": measure, "term": label, "estimate_sd": fit.params[term], "ci_low": ci[0],
                         "ci_high": ci[1], "p_value": fit.pvalues[term]})
    out = pd.DataFrame(rows)
    out["p_holm"] = out.groupby("term")["p_value"].transform(_holm)
    return out


def collision_gee() -> pd.DataFrame:
    """Hazard collisions: logistic GEE with exchangeable correlation within participants."""
    data = load_window("haz30")
    data = data[(data["hazard_valid"] == 1) & data["haz__onset_before_end_s"].notna() & data["haz__crash"].notna()]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.gee("haz__crash ~ cohort_v1 + round2 + alcohol + cd + alcohol:cd + haz__speed_at_onset",
                      groups="participant_name", data=data, family=sm.families.Binomial(),
                      cov_struct=sm.cov_struct.Exchangeable()).fit()
    ci = fit.conf_int()
    rows = [{"term": term, "odds_ratio": float(np.exp(fit.params[term])), "ci_low": float(np.exp(ci.loc[term, 0])),
             "ci_high": float(np.exp(ci.loc[term, 1])), "p_value": float(fit.pvalues[term]),
             "n_trips": int(len(data)), "n_events": int(data["haz__crash"].sum())}
            for term in ["round2", "alcohol", "cd", "alcohol:cd"]]
    return pd.DataFrame(rows)


MODEL_SPECS = {
    "primary": ("Primary (Eq. 2): both cohorts, pre90",
                "z ~ cohort_v1 + round2 + alcohol + cd + alcohol:cd + C(scenario_code)", None),
    "naive": ("Naive: alcohol cohort only, pre90",
              "z ~ round2 + cd + round2:cd + C(scenario_code)", "cohort_v1 == 1"),
    "task": ("Task-specific CD: both cohorts, pre90",
             "z ~ cohort_v1 + round2 + alcohol + nback + sentence + alcohol:nback + alcohol:sentence + C(scenario_code)", None),
    "full": ("Full cohort x session x CD: both cohorts, pre90",
             "z ~ cohort_v1 * round2 * cd + C(scenario_code)", None),
}


def all_coefficients() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every fixed effect of every behavioural model, plus one fit summary row per model and measure.

    All models: random intercept per participant, REML, outcome z-scored on sober R1 no-task trips
    of the full sample (so all models share units)."""
    data = load_window("pre90")
    data["nback"] = (data["cognitive_task"] == "nback_task").astype(int)
    data["sentence"] = (data["cognitive_task"] == "statement_task").astype(int)
    coef_rows, fit_rows = [], []
    for key, (label, formula, subset) in MODEL_SPECS.items():
        for measure in BEHAVIOUR_MEASURES:
            frame = data.assign(z=_standardised(data, measure)).dropna(subset=["z"])
            if subset:
                frame = frame.query(subset)
            fit = _fit(formula, frame)
            ci = fit.conf_int()
            for term in fit.fe_params.index:
                coef_rows.append({"model": key, "measure": measure, "term": term,
                                  "estimate_sd": fit.fe_params[term], "se": fit.bse_fe[term],
                                  "z": fit.tvalues[term], "p_value": fit.pvalues[term],
                                  "ci_low": ci.loc[term, 0], "ci_high": ci.loc[term, 1]})
            fit_rows.append({"model": key, "description": label, "formula": formula,
                             "random_effects": "(1 | participant)", "estimation": "REML",
                             "subset": subset or "all included trips", "measure": measure,
                             "n_trips": int(fit.nobs), "n_participants": int(frame["participant_name"].nunique()),
                             "var_participant": float(fit.cov_re.iloc[0, 0]), "var_residual": float(fit.scale),
                             "converged": bool(fit.converged)})
    return pd.DataFrame(coef_rows), pd.DataFrame(fit_rows)


KEY_TERMS = {"primary": ["alcohol", "cd", "alcohol:cd", "round2"], "naive": ["round2", "cd", "round2:cd"],
             "task": ["nback", "sentence", "alcohol:nback", "alcohol:sentence"],
             "full": ["cohort_v1:round2", "cohort_v1:round2:cd", "cohort_v1:cd", "round2:cd"]}


def convergence_check(coefs: pd.DataFrame, fits: pd.DataFrame) -> pd.DataFrame:
    """Refit every fit that reported non-convergence with Powell and Nelder-Mead optimisers, and with
    OLS plus participant-clustered SEs; report the largest change in the model's key estimates."""
    data = load_window("pre90")
    data["nback"] = (data["cognitive_task"] == "nback_task").astype(int)
    data["sentence"] = (data["cognitive_task"] == "statement_task").astype(int)
    rows = []
    for _, row in fits[~fits["converged"]].iterrows():
        _, formula, subset = MODEL_SPECS[row["model"]]
        frame = data.assign(z=_standardised(data, row["measure"])).dropna(subset=["z"])
        if subset:
            frame = frame.query(subset)
        base = coefs[(coefs["model"] == row["model"]) & (coefs["measure"] == row["measure"])].set_index("term")
        terms = KEY_TERMS[row["model"]]
        out = {"model": row["model"], "measure": row["measure"]}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for method in ["powell", "nm"]:
                fit = smf.mixedlm(formula, frame, groups=frame["participant_name"]).fit(reml=True, method=method)
                out[f"{method}_converged"] = bool(fit.converged)
                out[f"{method}_max_abs_diff_sd"] = max(abs(fit.fe_params[t] - base.loc[t, "estimate_sd"]) for t in terms)
            ols = smf.ols(formula, frame).fit(cov_type="cluster",
                                             cov_kwds={"groups": pd.factorize(frame["participant_name"])[0]})
            out["ols_cluster_max_abs_diff_sd"] = max(abs(ols.params[t] - base.loc[t, "estimate_sd"]) for t in terms)
            out["ols_cluster_max_abs_diff_p"] = max(abs(ols.pvalues[t] - base.loc[t, "p_value"]) for t in terms)
            out["any_key_term_p_below_05"] = bool((base.loc[terms, "p_value"] < 0.05).any())
        rows.append(out)
    return pd.DataFrame(rows)


def collision_gee_full() -> tuple[pd.DataFrame, dict]:
    """All GEE coefficients and the specification, for the supplementary data."""
    data = load_window("haz30")
    data = data[(data["hazard_valid"] == 1) & data["haz__onset_before_end_s"].notna() & data["haz__crash"].notna()]
    formula = "haz__crash ~ cohort_v1 + round2 + alcohol + cd + alcohol:cd + haz__speed_at_onset"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.gee(formula, groups="participant_name", data=data, family=sm.families.Binomial(),
                      cov_struct=sm.cov_struct.Exchangeable()).fit()
    ci = fit.conf_int()
    rows = [{"term": term, "log_odds": float(fit.params[term]), "robust_se": float(fit.bse[term]),
             "z": float(fit.tvalues[term]), "p_value": float(fit.pvalues[term]),
             "odds_ratio": float(np.exp(fit.params[term])), "or_ci_low": float(np.exp(ci.loc[term, 0])),
             "or_ci_high": float(np.exp(ci.loc[term, 1]))} for term in fit.params.index]
    spec = {"formula": formula, "family": "Binomial (logit link)", "working_correlation": "Exchangeable within participant",
            "standard_errors": "Robust (sandwich)", "estimated_within_participant_correlation": float(fit.cov_struct.dep_params),
            "trips": int(len(data)), "participants": int(data["participant_name"].nunique()),
            "collisions": int(data["haz__crash"].sum()),
            "inclusion": "Valid hazards with an estimated onset (door-open scenario excluded); window haz30"}
    return pd.DataFrame(rows), spec


def _holm(p: pd.Series) -> pd.Series:
    from src.study.models import holm

    return holm(p)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    outputs = {
        "cohort_false_alarms.csv": cohort_false_alarms(),
        "naive_vs_corrected.csv": naive_vs_corrected(),
        "task_specific_cd.csv": task_specific(),
        "full_model.csv": full_model(),
        "collision_gee.csv": collision_gee(),
    }
    for name, frame in outputs.items():
        frame.to_csv(RESULTS / name, index=False)
        print(f"== {name}")
        print(frame.round(3).to_string(index=False))

    coefs, fits = all_coefficients()
    coefs.to_csv(RESULTS / "lmm_all_coefficients.csv", index=False)
    fits.to_csv(RESULTS / "lmm_fit_summaries.csv", index=False)
    gee_coefs, gee_spec = collision_gee_full()
    gee_coefs.to_csv(RESULTS / "collision_gee_all_coefficients.csv", index=False)
    pd.Series(gee_spec).to_csv(RESULTS / "collision_gee_specification.csv", header=["value"])
    print(f"== all coefficients: {len(coefs)} rows; non-converged fits: {int((~fits['converged']).sum())}")
    check = convergence_check(coefs, fits)
    check.to_csv(RESULTS / "lmm_convergence_check.csv", index=False)
    print(check.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
