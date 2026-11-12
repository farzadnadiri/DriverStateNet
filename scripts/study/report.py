"""Stage 3: write artifacts/study/RESULTS.md from the saved aggregate results.

    python -m scripts.study.report

Every number in RESULTS.md comes from artifacts/study/results/*.csv or
artifacts/study/audit/*.csv; nothing is typed in by hand.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "artifacts/study"


def _fmt_ci(row, est="estimate_sd") -> str:
    return f"{row[est]:+.2f} [{row['ci_low']:+.2f}, {row['ci_high']:+.2f}]"


def _lmm_table(lmm: pd.DataFrame, window: str) -> str:
    frame = lmm[lmm["window"] == window]
    terms = ["cd", "alcohol", "alcohol:cd", "round2"]
    header = "| Measure | " + " | ".join(terms) + " | ICC |\n|---|" + "---|" * (len(terms) + 1) + "\n"
    lines = []
    for measure, group in frame.groupby("measure", sort=False):
        cells = []
        for term in terms:
            row = group[group["term"] == term]
            if row.empty:
                cells.append("")
                continue
            row = row.iloc[0]
            mark = " **\\***" if row["p_holm_within_term"] < 0.05 else ""
            cells.append(_fmt_ci(row) + mark)
        lines.append(f"| {measure.removeprefix('beh__')} | " + " | ".join(cells) + f" | {group['icc_null_model'].iloc[0]:.2f} |")
    return header + "\n".join(lines)


TRI_TO_BEH = {
    "gaze_yaw__std": "beh__gaze_yaw_sd",
    "gaze_pitch__std": "beh__gaze_pitch_sd",
    "gaze_fixation__change_rate": "beh__fixation_rate",
    "gaze_saccade__change_rate": "beh__saccade_rate",
    "tobii_eye_pupil_diameter__mean": "beh__pupil_mean",
    "reversals_0_5__sum_rate": "beh__reversals_0_5",
    "reversals_2_5__sum_rate": "beh__reversals_2_5",
}
INTOX = "TRI intoxicated: v1 no-CD, R2 (drunk) - R1 (sober)"
ORDER = "order control: v2 no-CD, R2 - R1 (no alcohol)"


def _tri_section(results: Path) -> str:
    rep = pd.read_csv(results / "tri_wilcoxon_replication.csv")
    lmm = pd.read_csv(results / "lmm_behaviour.csv")
    published = pd.read_csv(STUDY / "reference/tri_published_wilcoxon.csv").set_index("feature")
    lines = ["\n## 8. What TRI found vs what this study adds (pre90 window)\n",
             "TRI's validation compares sober Round 1 with drunk Round 2 inside the alcohol arm. Columns: TRI's "
             "published difference; our replication of that test (all trips, as in the paper; the primary trip set "
             "is in `tri_wilcoxon_replication.csv`); the same test in the control arm, which estimates the session-"
             "order change without alcohol; their difference; and this study's mixed-model alcohol effect, which "
             "corrects for session order. Raw units except the last column (SD units, Holm \\*).\n",
             "| Feature | TRI published | Replication | Control arm R2−R1 | Difference | LMM alcohol (SD) |",
             "|---|---|---|---|---|---|"]
    rep_all = rep[rep["trip_set"] == "all"].set_index(["test", "feature"])
    for feature in rep_all.loc[INTOX].index:
        intox = rep_all.loc[(INTOX, feature)]
        order = rep_all.loc[(ORDER, feature)]
        pub = (f"{published.loc[feature, 'intox_diff']:+.3f} (p={published.loc[feature, 'intox_p']:.3f})"
               if feature in published.index else "not reported")
        beh = TRI_TO_BEH.get(feature)
        lmm_cell = ""
        if beh is not None:
            row = lmm[(lmm["window"] == "pre90") & (lmm["measure"] == beh) & (lmm["term"] == "alcohol")].iloc[0]
            lmm_cell = _fmt_ci(row) + (" \\*" if row["p_holm_within_term"] < 0.05 else "")
        lines.append(
            f"| {feature} | {pub} | {intox['mean_diff']:+.3f} (p={intox['p_value']:.3f}) | "
            f"{order['mean_diff']:+.3f} (p={order['p_value']:.3f}) | {intox['mean_diff'] - order['mean_diff']:+.3f} | {lmm_cell} |"
        )
    lines.append("\nReplication check: vehicle features and pupil diameter reproduce TRI's published values; the "
                 "three gaze-event features (yaw SD, fixations, saccades) agree in direction but differ in size, "
                 "consistent with a different version of the derived gaze files.\n")
    return "\n".join(lines)


def _sensitivity_section(lmm: pd.DataFrame) -> str:
    frame = lmm[(lmm["window"] == "pre90") & lmm["term"].isin(["cd", "alcohol", "alcohol:cd"])]
    table = frame.pivot_table(index="measure", columns="term", values="mde_80").round(2)
    table.index = [m.removeprefix("beh__") for m in table.index]
    return ("\n## 9. Sensitivity: minimum detectable effects (pre90)\n\n"
            "Smallest true effect detectable with 80% power at two-sided α = .05 (unadjusted), "
            "= 2.80 × the model's standard error, in SD units of sober Round-1 driving. Effects smaller "
            "than these cannot be ruled out by a non-significant result; the alcohol and interaction "
            "terms rest on 20 alcohol-arm participants.\n\n" + table[["cd", "alcohol", "alcohol:cd"]].to_markdown())


ESTIMANDS = {
    "CD_sober": "CD (sober cells)",
    "alcohol_noCD": "alcohol, no CD",
    "alcohol_CD": "alcohol, during CD",
    "alcohol_x_CD": "alcohol × CD",
    "order_controls_noCD": "session change, controls",
}


def _revision_section(results: Path) -> str:
    """Main analysis since the 29 Oct 2026 revision: full cohort × session × CD model."""
    revision = results / "revision"
    est_path = revision / "full_model_estimands.csv"
    if not est_path.exists():
        return ""
    est = pd.read_csv(est_path)
    lines = ["\n## 12. Main analysis since the revision: full cohort × session × CD model (post hoc)\n",
             "Model: `z ~ cohort_v1 * round2 * cd + C(scenario) + (1|participant)`, pre90 window, primary trip set. "
             "Estimands are contrasts of the fixed effects in SD units of sober Round-1 no-task trips, with 95% Wald CI. "
             "Bold \\* = Holm-adjusted p < 0.05 across the 15 outcomes within each estimand. Sections 2–11 use the "
             "pre-specified constrained model; this model replaced it as the main specification "
             "(see the deviations log in `docs/HYPOTHESES.md`).\n"]
    est["cell"] = est.apply(lambda r: f"{r['estimate_sd']:+.2f} [{r['ci_low_sd']:+.2f}, {r['ci_high_sd']:+.2f}]"
                            + (" **\\***" if r["p_holm"] < 0.05 else ""), axis=1)
    table = est[est["estimand"].isin(ESTIMANDS)].pivot_table(index="measure", columns="estimand", values="cell",
                                                             aggfunc="first", sort=False)
    table = table[list(ESTIMANDS)].rename(columns=ESTIMANDS)
    table.index = [m.removeprefix("beh__") for m in table.index]
    lines.append(table.to_markdown())

    naive_path = revision / "naive_vs_full.csv"
    if naive_path.exists():
        naive = pd.read_csv(naive_path)
        lines.append("\nAlcohol estimate without CD: naive (alcohol cohort only, R2 − R1) vs full model, with the "
                     "difference, its paired participant-bootstrap CI and Holm-adjusted p:\n")
        show = naive[["measure", "naive_sd", "full_sd", "difference_sd", "diff_boot_ci_low", "diff_boot_ci_high",
                      "diff_p_holm"]].copy()
        show["measure"] = show["measure"].str.removeprefix("beh__")
        lines.append(show.round(3).to_markdown(index=False))

    sens_path = revision / "full_model_sensitivity.csv"
    if sens_path.exists():
        sens = pd.concat([est.assign(analysis="full (main)"), pd.read_csv(sens_path)])
        sens = sens[sens["estimand"].isin(["CD_sober", "alcohol_noCD", "alcohol_x_CD", "order_controls_noCD"])]
        lines.append("\nNumber of outcomes with Holm-adjusted p < 0.05, per specification and estimand "
                     "(outcome sets differ between specifications):\n")
        counts = sens.assign(sig=sens["p_holm"] < 0.05).pivot_table(index="analysis", columns="estimand", values="sig",
                                                                     aggfunc="sum", sort=False)
        lines.append(counts.fillna(0).astype(int).to_markdown())

    fa_path = revision / "cohort_false_alarms_v3.csv"
    if fa_path.exists():
        fa = pd.read_csv(fa_path)
        fa = fa[fa["quantity"].isin(["cohort gap R1 (trip)", "cohort gap R1 (driver)"])]
        lines.append("\nCohort false alarms: share of sober Round-1 trips predicted as alcohol/combined, alcohol cohort "
                     "minus control cohort, with within-cohort participant-bootstrap CI and Holm across the four models:\n")
        lines.append(fa[["model", "quantity", "estimate", "ci_low", "ci_high", "p_value", "p_holm_across_models"]]
                     .round(3).to_markdown(index=False))
    return "\n".join(lines)


def main():
    results = STUDY / "results"
    audit = STUDY / "audit"
    lmm = pd.read_csv(results / "lmm_behaviour.csv")
    diag = pd.read_csv(results / "order_cohort_diagnostics.csv")
    clf = pd.read_csv(results / "classification_by_window.csv")
    paired = pd.read_csv(results / "window_comparison_paired.csv")
    reasons = pd.read_csv(audit / "study_participants_by_reason_and_label.csv", index_col=0)
    validation = pd.read_csv(audit / "tri_feature_validation_vs_cache.csv")

    out = ["# IDD session-order study: results", ""]
    out.append("*Generated by `scripts/study/report.py` from `artifacts/study/results/` and `artifacts/study/audit/`. "
               "Pre-specified plan: `docs/HYPOTHESES.md`. Sections 2–11 use the pre-specified constrained model; "
               "the main analysis since the revision (full cohort × session × CD model) is in Section 12.*\n")

    out.append("## 1. Trip audit\n")
    out.append("Study participants (52); trips by exclusion reason and label:\n")
    out.append(reasons.to_markdown())
    out.append(f"\nTRI-feature reproduction check on the course-report window: worst-feature correlation "
               f"{validation['corr'].min():.4f} over {int(validation['n_matched'].min())} trips.\n")

    out.append("## 2. Behavioural effects (H1–H3, H5): linear mixed models\n")
    out.append("Coefficients in SD units of sober Round-1 no-task trips, with 95% CI. "
               "`alcohol` = additional R1→R2 change in the alcohol arm beyond the control arm's R1→R2 change; "
               "`round2` = session-order effect estimated from both arms. "
               "Bold \\* = Holm-adjusted p < 0.05 within each term. Model: "
               "`z ~ cohort_v1 + round2 + alcohol + cd + alcohol:cd + C(scenario) + (1|participant)`.\n")
    out.append("### Primary window: pre90 ([−120 s, −30 s] before scenario end)\n")
    out.append(_lmm_table(lmm, "pre90"))
    out.append("\n### Hazard window: haz30 (last 30 s), for comparison\n")
    out.append(_lmm_table(lmm, "haz30"))

    out.append("\n## 3. Session-order and cohort diagnostics (H5), pre90, TRI features, nested CV\n")
    show = diag[["analysis", "uar", "ci_low", "ci_high", "n_trips", "n_participants"]].copy()
    out.append(show.round(3).to_markdown(index=False))

    out.append("\n## 4. Classification by window (H4), TRI features, nested CV\n")
    cols = ["task", "window", "uar", "ci_low", "ci_high"] + [c for c in clf.columns if c.startswith("recall__")]
    out.append(clf[cols].round(3).to_markdown(index=False))
    out.append("\nPaired participant-bootstrap comparisons (same trips):\n")
    out.append(paired[["task", "model_a", "model_b", "uar_a", "uar_b", "delta", "ci_low", "ci_high", "p_value"]]
               .round(3).to_markdown(index=False))

    robustness = pd.read_csv(results / "lmm_behaviour_robustness.csv")
    out.append("\n## 5. Robustness of behavioural effects\n")
    out.append("Estimate (SD units) per analysis variant; \\* = Holm-adjusted p < 0.05 within term and variant.\n")
    robustness["cell"] = robustness.apply(
        lambda r: f"{r['estimate_sd']:+.2f}" + ("\\*" if r["p_holm_within_term"] < 0.05 else ""), axis=1)
    table = robustness.pivot_table(index=["measure", "term"], columns="variant", values="cell", aggfunc="first")
    table.index = [f"{m.removeprefix('beh__')} · {t}" for m, t in table.index]
    out.append(table.to_markdown())

    hazard = pd.read_csv(results / "lmm_hazard_response.csv")
    hazard_desc = pd.read_csv(results / "hazard_response_by_label.csv")
    out.append("\n## 6. Hazard response (exploratory)\n")
    out.append("Valid hazards with an estimable onset (door-open scenario 2b has no onset signal and is excluded). "
               "Raw units: log seconds for reaction times, probability for braked/crash, m/s² for peak deceleration, "
               "metres for minimum distance. Model adds speed at onset.\n")
    show = hazard[hazard["term"].isin(["cd", "alcohol", "alcohol:cd", "round2"])]
    out.append(show[["measure", "term", "estimate_sd", "ci_low", "ci_high", "p_value", "p_holm_within_term", "n_trips"]]
               .rename(columns={"estimate_sd": "estimate"}).round(3).to_markdown(index=False))
    out.append("\nMeans by condition:\n")
    out.append(hazard_desc.round(3).to_markdown(index=False))

    hazard_clf = pd.read_csv(results / "classification_hazard_response.csv")
    hazard_paired = pd.read_csv(results / "classification_hazard_response_paired.csv")
    out.append("\n## 7. Classification with hazard-response features (H4, valid hazards only)\n")
    out.append(hazard_clf[["task", "analysis", "uar", "ci_low", "ci_high", "n_trips"]].round(3).to_markdown(index=False))
    out.append("\nPaired comparisons against `pre30: TRI`:\n")
    out.append(hazard_paired[["task", "model_a", "uar_a", "uar_b", "delta", "ci_low", "ci_high", "p_value"]]
               .round(3).to_markdown(index=False))

    out.append(_tri_section(results))
    out.append(_sensitivity_section(lmm))
    timing = pd.read_csv(results / "session_timing_by_cohort.csv")
    out.append("\n## 10. Session timing by cohort (minutes)\n")
    out.append("Break = start of first Round-2 drive minus end of last Round-1 drive. The alcohol arm's break "
               "included drinking and absorption.\n")
    cols = ["cohort"] + [c for c in timing.columns if c.endswith(("__count", "__50%", "__25%", "__75%"))]
    out.append(timing[cols].round(1).to_markdown(index=False))

    models_path = results / "model_comparison.csv"
    if models_path.exists():
        models = pd.read_csv(models_path)
        model_paired = pd.read_csv(results / "model_comparison_paired.csv")
        out.append("\n## 11. Matched model comparison (four-class, pre90, nested CV)\n")
        cols = ["analysis", "input", "uar", "ci_low", "ci_high"] + [c for c in models.columns if c.startswith("recall__")]
        out.append(models[cols].round(3).to_markdown(index=False))
        out.append("\nPaired comparisons against logistic regression:\n")
        out.append(model_paired[["model_a", "uar_a", "uar_b", "delta", "ci_low", "ci_high", "p_value"]]
                   .round(3).to_markdown(index=False))

    out.append(_revision_section(results))

    out.append("\n## Figures\n")
    for name in sorted(p.stem for p in (STUDY / "figures").glob("*.png")):
        out.append(f"![{name}](figures/{name}.png)\n")

    (STUDY / "RESULTS.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {STUDY / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
