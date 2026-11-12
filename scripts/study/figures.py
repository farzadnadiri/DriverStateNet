"""Stage 4: study figures from saved aggregate results.

    python -m scripts.study.figures

Writes PNG (300 dpi) and PDF to artifacts/study/figures/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "artifacts/study/results"
FIGURES = ROOT / "artifacts/study/figures"

# First three categorical slots of the validated reference palette (all-pairs safe),
# plus a marker shape per series so identity never relies on colour alone.
SERIES = [("#2a78d6", "o"), ("#eb6834", "s"), ("#1baf7a", "^")]
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

MEASURE_NAMES = {
    "beh__gaze_yaw_sd": "Gaze yaw SD",
    "beh__gaze_pitch_sd": "Gaze pitch SD",
    "beh__prc": "Percent road centre",
    "beh__fixation_rate": "Fixation rate",
    "beh__saccade_rate": "Saccade rate",
    "beh__pupil_mean": "Pupil diameter",
    "beh__gaze_valid_frac": "Gaze validity",
    "beh__lateral_dev_sd_straight": "Lateral deviation SD (straight)",
    "beh__steer_hf_sd_straight": "Steering HF SD (straight)",
    "beh__reversals_0_5": "Steering reversals 0.5°",
    "beh__reversals_2_5": "Steering reversals 2.5°",
    "beh__lat_accel_sd": "Lateral acceleration SD",
    "beh__speed_mean": "Speed mean",
    "beh__speed_sd": "Speed SD",
}
GAZE = ["beh__gaze_yaw_sd", "beh__gaze_pitch_sd", "beh__prc", "beh__fixation_rate", "beh__saccade_rate",
        "beh__pupil_mean", "beh__gaze_valid_frac"]
VEHICLE = ["beh__lateral_dev_sd_straight", "beh__steer_hf_sd_straight", "beh__reversals_0_5", "beh__reversals_2_5",
           "beh__lat_accel_sd", "beh__speed_mean", "beh__speed_sd"]


def _style(ax):
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK, length=0)


def _save(fig, name: str):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def forest_plot():
    lmm = pd.read_csv(RESULTS / "lmm_behaviour.csv")
    lmm = lmm[lmm["window"] == "pre90"]
    terms = [("cd", "Cognitive distraction"), ("alcohol", "Alcohol (order-corrected)"), ("round2", "Session order (R2 vs R1)")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True)
    for ax, (title, measures) in zip(axes, [("Gaze", GAZE), ("Vehicle control", VEHICLE)]):
        n = len(measures)
        for j, (term, label) in enumerate(terms):
            colour, marker = SERIES[j]
            for i, measure in enumerate(measures):
                row = lmm[(lmm["measure"] == measure) & (lmm["term"] == term)].iloc[0]
                y = (n - 1 - i) + (1 - j) * 0.24
                sig = row["p_holm_within_term"] < 0.05
                ax.plot([row["ci_low"], row["ci_high"]], [y, y], color=colour, linewidth=2, solid_capstyle="round")
                ax.plot(row["estimate_sd"], y, marker=marker, markersize=8, color=colour,
                        markerfacecolor=colour if sig else "white", markeredgewidth=2, linestyle="none",
                        label=label if i == 0 else None)
        ax.axvline(0, color=INK_MUTED, linewidth=1)
        ax.set_yticks(range(n))
        ax.set_yticklabels([MEASURE_NAMES[m] for m in reversed(measures)], color=INK)
        ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
        ax.set_xlabel("Effect (SD of sober Round-1 driving), 95% CI", color=INK_MUTED)
        _style(ax)
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=SERIES[j][0], marker=SERIES[j][1], markersize=8, linewidth=2, label=label)
               for j, (_, label) in enumerate(terms)]
    handles.append(Line2D([], [], color=INK_MUTED, marker="o", markersize=8, markerfacecolor="white",
                          markeredgewidth=2, linestyle="none", label="Hollow: not significant"))
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.text(0.5, -0.04, "Filled marker: Holm-adjusted p < 0.05. Pre-hazard window [−120 s, −30 s]; "
             "linear mixed models with participant random intercept and scenario fixed effects.",
             ha="center", color=INK_MUTED, fontsize=8)
    fig.tight_layout()
    _save(fig, "fig_behaviour_effects")


def _dot_ci(ax, rows: pd.DataFrame, label_col: str, chance: float):
    n = len(rows)
    for i, (_, row) in enumerate(rows.iterrows()):
        y = n - 1 - i
        colour, marker = SERIES[0]
        ax.plot([row["ci_low"], row["ci_high"]], [y, y], color=colour, linewidth=2, solid_capstyle="round")
        ax.plot(row["uar"], y, marker=marker, markersize=8, color=colour, linestyle="none")
        ax.text(row["ci_high"] + 0.01, y, f"{row['uar']:.2f}", va="center", color=INK, fontsize=8)
    ax.axvline(chance, color=INK_MUTED, linewidth=1, linestyle=(0, (3, 3)))
    ax.set_yticks(range(n))
    ax.set_yticklabels(list(reversed(rows[label_col].tolist())), color=INK)
    ax.set_ylim(-0.7, n - 0.3)
    _style(ax)


def order_diagnostics():
    diag = pd.read_csv(RESULTS / "order_cohort_diagnostics.csv")
    names = {
        "order_only": "Session order only\n(control cohort, R2 vs R1)",
        "cohort_only": "Cohort only\n(sober R1, alcohol vs control cohort)",
        "alcohol arm": "Alcohol cohort\n(R2 intoxicated vs R1 sober)",
        "transfer": "Order model from controls\napplied to alcohol cohort",
    }
    diag["name"] = [next(v for k, v in names.items() if a.startswith(k)) for a in diag["analysis"]]
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    _dot_ci(ax, diag, "name", chance=0.5)
    ax.set_xlim(0.4, 0.85)
    ax.set_xlabel("UAR (nested participant-grouped CV), 95% participant-bootstrap CI; dashed = chance",
                  color=INK_MUTED, fontsize=8)
    fig.tight_layout()
    _save(fig, "fig_order_cohort_diagnostics")


def window_comparison():
    clf = pd.read_csv(RESULTS / "classification_by_window.csv")
    tasks = [("four_class", "Four-class", 0.25), ("cd_binary", "Distraction (binary)", 0.5),
             ("intox_binary", "Intoxication (binary)", 0.5)]
    windows = [("pre90", "Pre-hazard 90 s"), ("pre30", "Pre-hazard 30 s"), ("haz30", "Hazard 30 s")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.6), sharey=True)
    for ax, (task, title, chance) in zip(axes, tasks):
        for j, (window, label) in enumerate(windows):
            row = clf[(clf["task"] == task) & (clf["window"] == window)].iloc[0]
            colour, marker = SERIES[j]
            y = 2 - j
            ax.plot([row["ci_low"], row["ci_high"]], [y, y], color=colour, linewidth=2, solid_capstyle="round")
            ax.plot(row["uar"], y, marker=marker, markersize=8, color=colour, linestyle="none")
            ax.text(row["ci_high"] + 0.01, y, f"{row['uar']:.2f}", va="center", color=INK, fontsize=8)
        ax.axvline(chance, color=INK_MUTED, linewidth=1, linestyle=(0, (3, 3)))
        ax.set_title(title, loc="left", color=INK, fontsize=10, fontweight="bold")
        ax.set_yticks([2, 1, 0])
        ax.set_yticklabels([label for _, label in windows], color=INK)
        ax.set_xlim(chance - 0.05, chance + 0.45)
        _style(ax)
    axes[1].set_xlabel("UAR, ten dataset features, nested CV, 95% participant-bootstrap CI; dashed = chance", color=INK_MUTED)
    fig.tight_layout()
    _save(fig, "fig_window_comparison")


def design_schematic():
    """Session timeline (top) and analysis windows within one trip (bottom)."""
    from matplotlib.patches import FancyBboxPatch

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 4.2), gridspec_kw={"height_ratios": [1.1, 1]})
    for ax in (top, bottom):
        ax.set_axis_off()

    def box(ax, x, y, w, h, label, colour, text_colour=INK, alpha=0.18):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02",
                                    facecolor=colour, alpha=alpha, edgecolor=colour, linewidth=1.5))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9, color=text_colour)

    top.set_xlim(0, 10)
    top.set_ylim(0, 2.4)
    top.text(0, 2.25, "Session (both arms)", fontsize=10, fontweight="bold", color=INK)
    box(top, 0.0, 1.25, 3.0, 0.7, "Part R1: 4 routes\n(2 with cognitive task)", SERIES[0][0])
    box(top, 3.2, 1.25, 2.3, 0.7, "Break\nalcohol arm: drink to ~0.10% BAC", SERIES[1][0])
    box(top, 5.7, 1.25, 3.0, 0.7, "Part R2: 4 routes\n(2 with cognitive task)", SERIES[0][0])
    top.text(0.0, 0.75, "Alcohol arm (n = 20):  R1 sober  →  R2 intoxicated   ·   median break 62 min",
             fontsize=9, color=INK)
    top.text(0.0, 0.3, "Control arm (n = 32):  R1 sober  →  R2 sober            ·   median break 21 min",
             fontsize=9, color=INK)

    bottom.set_xlim(-130, 8)
    bottom.set_ylim(0, 2.4)
    bottom.text(-130, 2.25, "One trip (~3.4 min), windows relative to trip end", fontsize=10, fontweight="bold", color=INK)
    bottom.plot([-125, 0], [1.6, 1.6], color=INK_MUTED, linewidth=2)
    for t in [-120, -90, -60, -30, 0]:
        bottom.plot([t, t], [1.52, 1.68], color=INK_MUTED, linewidth=1)
        bottom.text(t, 1.8, f"{t} s", ha="center", fontsize=8, color=INK_MUTED)
    bottom.text(-1.5, 1.95, "hazard", ha="right", fontsize=8, color=SERIES[1][0])
    bottom.plot([-3, -3], [1.45, 1.75], color=SERIES[1][0], linewidth=2.5)
    box(bottom, -120, 0.85, 90, 0.42, "pre90 (primary): routine driving", SERIES[0][0], alpha=0.25)
    box(bottom, -60, 0.3, 30, 0.42, "pre30", SERIES[0][0], alpha=0.12)
    box(bottom, -30, 0.3, 30, 0.42, "haz30", SERIES[1][0], alpha=0.2)
    fig.tight_layout()
    _save(fig, "fig_design")


def order_did_plot():
    """Participant means in R1 and R2 per arm (no-task trips, pre90), with 95% bootstrap CIs."""
    import numpy as np

    trips = pd.read_pickle(ROOT / "artifacts/study/private/trips.pkl")
    feats = pd.read_pickle(ROOT / "artifacts/study/private/window_features.pkl")
    data = trips[trips["included_primary"] & (trips["cognitive_task"] == "no_task")].merge(
        feats[feats["window"] == "pre90"], on=["participant_name", "scenario_path"])
    panels = [("beh__gaze_yaw_sd", "Horizontal gaze dispersion (°)"), ("beh__fixation_rate", "Fixation-transition rate (/min)"),
              ("beh__pupil_mean", "Pupil diameter (mm)")]
    arms = [(True, "Alcohol cohort (R2 intoxicated)", SERIES[1]), (False, "Control cohort (R2 sober)", (INK_MUTED, "o"))]
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, (measure, label) in zip(axes, panels):
        for k, (is_v1, arm_label, (colour, marker)) in enumerate(arms):
            arm = data[data["cohort_v1"] == is_v1]
            means = arm.groupby(["participant_name", "round"])[measure].mean().unstack().dropna()
            xs = np.array([0, 1]) + (k - 0.5) * 0.06
            centre = means.mean().to_numpy()
            boots = np.array([means.iloc[rng.integers(0, len(means), len(means))].mean().to_numpy()
                              for _ in range(2000)])
            lo, hi = np.quantile(boots, [0.025, 0.975], axis=0)
            ax.plot(xs, centre, color=colour, linewidth=2, marker=marker, markersize=7, label=arm_label)
            ax.vlines(xs, lo, hi, color=colour, linewidth=2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["R1", "R2"], color=INK)
        ax.set_title(label, loc="left", color=INK, fontsize=10, fontweight="bold")
        ax.set_xlim(-0.3, 1.3)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=INK_MUTED, labelcolor=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.text(0.5, -0.04, "No-task trips, pre90 window; means of participant means with 95% participant-bootstrap CIs.",
             ha="center", color=INK_MUTED, fontsize=8)
    fig.tight_layout()
    _save(fig, "fig_order_did")


def hazard_plot():
    """Hazard collision rate by condition with participant-bootstrap CIs (exploratory)."""
    import numpy as np

    desc = pd.read_csv(RESULTS / "hazard_response_by_label.csv")
    trips = pd.read_pickle(ROOT / "artifacts/study/private/trips.pkl")
    feats = pd.read_pickle(ROOT / "artifacts/study/private/window_features.pkl")
    data = trips[trips["included_primary"] & (trips["hazard_valid"] == 1)].merge(
        feats[feats["window"] == "haz30"], on=["participant_name", "scenario_path"])
    data = data[data["haz__onset_before_end_s"].notna()]
    outcome = data["hazard_outcome"].fillna("")
    data = data.assign(crash=outcome.str.contains("crash").astype(float))
    names = {"baseline": "Baseline", "cognitive_distraction": "CD", "alcohol_intoxication": "Alcohol", "combined": "Combined"}
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    for i, label in enumerate(desc["label"]):
        subset = data[data["label"] == label]
        groups = subset.groupby("participant_name")["crash"]
        sums, counts = groups.sum().to_numpy(), groups.count().to_numpy()
        boots = []
        for _ in range(2000):
            pick = rng.integers(0, len(sums), len(sums))
            boots.append(sums[pick].sum() / counts[pick].sum())
        rate = subset["crash"].mean()
        lo, hi = np.quantile(boots, [0.025, 0.975])
        colour = SERIES[0][0] if label == "combined" else "#b4b2ab"
        ax.bar(i, rate * 100, width=0.55, color=colour, edgecolor="white", linewidth=2)
        ax.vlines(i, lo * 100, hi * 100, color=INK_MUTED, linewidth=1.5)
        ax.text(i, hi * 100 + 1.5, f"{rate * 100:.0f}%\n(n = {len(subset)})", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(desc)))
    ax.set_xticklabels([names[label] for label in desc["label"]], color=INK)
    ax.set_ylabel("Hazard collisions (% of trips)", color=INK_MUTED)
    ax.set_ylim(0, 60)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    _save(fig, "fig_hazard_collisions")


def model_comparison_plot():
    """UAR by model (left) and normalized confusion matrix of the best feature model (right)."""
    import numpy as np

    path = RESULTS / "model_comparison.csv"
    if not path.exists():
        return
    models = pd.read_csv(path)
    fig, (ax, cm_ax) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1.2, 1]})
    n = len(models)
    for i, (_, row) in enumerate(models.iterrows()):
        y = n - 1 - i
        colour = SERIES[0][0] if "transformer" in row["analysis"].lower() else INK_MUTED
        ax.plot([row["ci_low"], row["ci_high"]], [y, y], color=colour, linewidth=2, solid_capstyle="round")
        ax.plot(row["uar"], y, "o", markersize=8, color=colour)
        ax.text(row["ci_high"] + 0.01, y, f"{row['uar']:.2f}", va="center", fontsize=8, color=INK)
    ax.axvline(0.25, color=INK_MUTED, linewidth=1, linestyle=(0, (3, 3)))
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{a}\n({inp.replace('TRI 10 features', '10 dataset features')})"
                        for a, inp in zip(models["analysis"][::-1], models["input"][::-1])],
                       color=INK, fontsize=8)
    ax.set_xlim(0.2, 0.65)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("Four-class UAR, pre90, nested CV, 95% CI; dashed = chance", color=INK_MUTED, fontsize=8)
    _style(ax)

    best = models.loc[models["analysis"] != "Temporal transformer", ["analysis", "uar"]].sort_values("uar").iloc[-1]
    key = {"Logistic regression": "logreg", "Linear SVM": "linear_svm", "Gradient boosting": "gbm"}[best["analysis"]]
    preds = pd.read_csv(ROOT / f"artifacts/study/private/pred_model_{key}_pre90.csv")
    labels = ["baseline", "cognitive_distraction", "alcohol_intoxication", "combined"]
    short = ["Base", "CD", "Alc", "Comb"]
    cm = pd.crosstab(preds["true_label"], preds["pred_label"]).reindex(index=labels, columns=labels, fill_value=0)
    norm = cm.div(cm.sum(axis=1), axis=0).to_numpy()
    cm_ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for r in range(4):
        for c in range(4):
            cm_ax.text(c, r, f"{norm[r, c]:.2f}", ha="center", va="center", fontsize=9,
                       color="white" if norm[r, c] > 0.5 else INK)
    cm_ax.set_xticks(range(4))
    cm_ax.set_xticklabels(short, color=INK)
    cm_ax.set_yticks(range(4))
    cm_ax.set_yticklabels(short, color=INK)
    cm_ax.set_xlabel("Predicted", color=INK_MUTED)
    cm_ax.set_ylabel("True", color=INK_MUTED)
    cm_ax.set_title(f"{best['analysis']}: row-normalized confusion", loc="left", fontsize=9, color=INK)
    for side in cm_ax.spines.values():
        side.set_visible(False)
    fig.tight_layout()
    _save(fig, "fig_model_comparison")


def _legend(ax, entries, **kwargs):
    """Legend with filled markers per series plus a hollow 'not significant' key."""
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=c, marker=m, markersize=7, linewidth=2, label=l) for l, c, m in entries]
    handles.append(Line2D([], [], color=INK_MUTED, marker="o", markersize=7, markerfacecolor="white",
                          markeredgewidth=2, linestyle="none", label="Hollow: not significant"))
    ax.legend(handles=handles, frameon=False, fontsize=8, **kwargs)


def cohort_shortcut_plot():
    """False 'alcohol' flags on sober trips: alcohol cohort R1 vs control R1 and R2, per model."""
    path = RESULTS / "cohort_false_alarms.csv"
    if not path.exists():
        return
    rates = pd.read_csv(path)
    rates = rates[rates["comparison"] == "rate"]
    models = list(dict.fromkeys(rates["model"]))
    series = [(("Alcohol cohort", "R1"), "Alcohol cohort, R1 (sober)", SERIES[1][0], "s"),
              (("Control cohort", "R1"), "Control cohort, R1", INK_MUTED, "o"),
              (("Control cohort", "R2"), "Control cohort, R2", "#b4b2ab", "^")]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    n = len(models)
    for j, (key, label, colour, marker) in enumerate(series):
        for i, model in enumerate(models):
            row = rates[(rates["model"] == model) & (rates["cohort"] == key[0]) & (rates["part"] == key[1])].iloc[0]
            y = n - 1 - i + (1 - j) * 0.22
            ax.plot([row["ci_low"] * 100, row["ci_high"] * 100], [y, y], color=colour, linewidth=2, solid_capstyle="round")
            ax.plot(row["estimate"] * 100, y, marker=marker, markersize=8, color=colour, linestyle="none",
                    label=label if i == 0 else None)
    ax.set_yticks(range(n))
    ax.set_yticklabels(list(reversed(models)), color=INK)
    ax.set_xlabel("Sober trips classified as alcohol or combined (%), 95% participant-bootstrap CI", color=INK_MUTED, fontsize=8)
    ax.set_xlim(0, 80)
    ax.set_ylim(-0.6, n - 0.4)
    _style(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, "fig_cohort_shortcut")


def naive_vs_corrected_plot():
    """Alcohol estimate within the alcohol arm only (naive) vs order-corrected, per measure."""
    path = RESULTS / "naive_vs_corrected.csv"
    if not path.exists():
        return
    est = pd.read_csv(path)
    measures = GAZE + VEHICLE
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    n = len(measures)
    styles = [("naive (within alcohol arm)", "Within alcohol cohort only (R2 − R1)", INK_MUTED, "o"),
              ("order-corrected", "Corrected for session order (difference in change)", SERIES[1][0], "s")]
    for j, (kind, label, colour, marker) in enumerate(styles):
        for i, measure in enumerate(measures):
            row = est[(est["measure"] == measure) & (est["estimate_type"] == kind)].iloc[0]
            y = n - 1 - i + (0.5 - j) * 0.36
            sig = row["p_holm"] < 0.05
            ax.plot([row["ci_low"], row["ci_high"]], [y, y], color=colour, linewidth=2, solid_capstyle="round")
            ax.plot(row["estimate_sd"], y, marker=marker, markersize=7, color=colour,
                    markerfacecolor=colour if sig else "white", markeredgewidth=2, linestyle="none",
                    label=label if i == 0 else None)
    ax.axvline(0, color=INK_MUTED, linewidth=1)
    ax.set_yticks(range(n))
    ax.set_yticklabels([MEASURE_NAMES[m] for m in reversed(measures)], color=INK, fontsize=8)
    ax.set_xlabel("Alcohol estimate (SD of sober R1 baseline driving), 95% CI; filled: Holm p < .05", color=INK_MUTED, fontsize=8)
    ax.set_ylim(-0.7, n - 0.3)
    _style(ax)
    _legend(ax, [(l, c, m) for _, l, c, m in styles], loc="upper center", bbox_to_anchor=(0.45, 1.13), ncol=1)
    fig.tight_layout()
    _save(fig, "fig_naive_vs_corrected")


def task_specific_plot():
    """CD effect by task: 1-back vs sentence comprehension."""
    path = RESULTS / "task_specific_cd.csv"
    if not path.exists():
        return
    est = pd.read_csv(path)
    measures = GAZE + VEHICLE
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    n = len(measures)
    styles = [("nback", "Auditory 1-back", SERIES[0][0], "o"), ("sentence", "Sentence comprehension", SERIES[2][0], "^")]
    for j, (term, label, colour, marker) in enumerate(styles):
        for i, measure in enumerate(measures):
            row = est[(est["measure"] == measure) & (est["term"] == term)].iloc[0]
            y = n - 1 - i + (0.5 - j) * 0.36
            sig = row["p_holm"] < 0.05
            ax.plot([row["ci_low"], row["ci_high"]], [y, y], color=colour, linewidth=2, solid_capstyle="round")
            ax.plot(row["estimate_sd"], y, marker=marker, markersize=7, color=colour,
                    markerfacecolor=colour if sig else "white", markeredgewidth=2, linestyle="none",
                    label=label if i == 0 else None)
        diff = est[est["term"] == "nback - sentence"].set_index("measure")
    for i, measure in enumerate(measures):
        if diff.loc[measure, "p_holm"] < 0.05:
            ax.text(1.32, n - 1 - i, "differ*", va="center", fontsize=7, color=INK)
    ax.axvline(0, color=INK_MUTED, linewidth=1)
    ax.set_yticks(range(n))
    ax.set_yticklabels([MEASURE_NAMES[m] for m in reversed(measures)], color=INK, fontsize=8)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel("Effect vs no task (SD of sober R1 baseline driving), 95% CI; filled: Holm p < .05;\n"
                  "'differ*': 1-back vs sentence difference, Holm p < .05", color=INK_MUTED, fontsize=8)
    _style(ax)
    _legend(ax, [(l, c, m) for _, l, c, m in styles], loc="upper center", bbox_to_anchor=(0.45, 1.1), ncol=3)
    fig.tight_layout()
    _save(fig, "fig_task_specific")


REVISION = RESULTS / "revision"
# Outcome names used for the revision (full-model) figures.
V3_NAMES = dict(MEASURE_NAMES, beh__gaze_yaw_sd="Horizontal gaze dispersion", beh__gaze_pitch_sd="Vertical gaze dispersion",
                beh__prc="Gaze concentration (MGC)", beh__fixation_rate="Fixation-transition rate",
                beh__saccade_rate="Saccade-transition rate", beh__steer_hf_sd_straight="High-pass steering SD, straight",
                beh__lateral_dev_sd_straight="Lateral deviation SD, straight", beh__lat_accel_sd="Lateral-acceleration SD",
                beh__reversals_0_5="Steering reversals 0.5°", beh__reversals_2_5="Steering reversals 2.5°")


def _forest(ax, est: pd.DataFrame, series, measures, value="estimate_sd", lo="ci_low_sd", hi="ci_high_sd", p="p_holm"):
    n = len(measures)
    k = len(series)
    for j, (key, label, colour, marker) in enumerate(series):
        for i, m in enumerate(measures):
            row = est[(est["measure"] == m) & (est["estimand"] == key)]
            if row.empty:
                continue
            row = row.iloc[0]
            y = n - 1 - i + ((k - 1) / 2 - j) * 0.26
            sig = row[p] < 0.05
            ax.plot([row[lo], row[hi]], [y, y], color=colour, linewidth=2, solid_capstyle="round")
            ax.plot(row[value], y, marker=marker, markersize=7, color=colour, markerfacecolor=colour if sig else "white",
                    markeredgewidth=2, linestyle="none")
    ax.axvline(0, color=INK_MUTED, linewidth=1)
    ax.set_yticks(range(n))
    ax.set_yticklabels([V3_NAMES[m] for m in reversed(measures)], color=INK, fontsize=9)
    ax.set_ylim(-0.7, n - 0.3)
    _style(ax)


def v3_effects_plot():
    """Main full-model estimands: CD among sober drivers, alcohol (no CD), alcohol (during CD), session order."""
    est = pd.read_csv(REVISION / "full_model_estimands.csv")
    series = [("CD_sober", "Cognitive distraction (sober drivers)", SERIES[0][0], "o"),
              ("alcohol_noCD", "Alcohol-associated change, no CD", SERIES[1][0], "s"),
              ("alcohol_CD", "Alcohol-associated change, during CD", "#f4a582", "D"),
              ("order_controls_noCD", "Session change, controls", INK_MUTED, "^")]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 6.4), sharex=True)
    for ax, (title, measures) in zip(axes, [("Gaze", GAZE), ("Vehicle control", VEHICLE)]):
        _forest(ax, est, series, measures)
        ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
        ax.set_xlabel("Estimate (baseline SD), 95% CI", color=INK_MUTED, fontsize=9)
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=c, marker=m, markersize=7, linewidth=2, label=l) for _, l, c, m in series]
    handles.append(Line2D([], [], color=INK_MUTED, marker="o", markersize=7, markerfacecolor="white",
                          markeredgewidth=2, linestyle="none", label="Hollow: not significant"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.09), ncol=3, frameon=False, fontsize=9)
    fig.tight_layout()
    _save(fig, "fig_v3_effects")


def v3_naive_vs_full_plot():
    path = REVISION / "naive_vs_full.csv"
    if not path.exists():
        return
    d = pd.read_csv(path)
    long = pd.concat([
        d.assign(estimand="naive", estimate_sd=d["naive_sd"], ci_low_sd=d["naive_ci_low"], ci_high_sd=d["naive_ci_high"],
                 p_holm=d["naive_p_holm"]),
        d.assign(estimand="full", estimate_sd=d["full_sd"], ci_low_sd=d["full_ci_low"], ci_high_sd=d["full_ci_high"],
                 p_holm=d["full_p_holm"])])
    series = [("naive", "Within alcohol cohort only (R2 − R1)", INK_MUTED, "o"),
              ("full", "Difference in change vs controls (full model)", SERIES[1][0], "s")]
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    measures = GAZE + VEHICLE
    _forest(ax, long, series, measures)
    for i, m in enumerate(measures):
        row = d[d["measure"] == m].iloc[0]
        if row["diff_boot_ci_low"] > 0 or row["diff_boot_ci_high"] < 0:
            ax.text(1.55, len(measures) - 1 - i, "Δ≠0", va="center", fontsize=7, color=INK)
    ax.set_xlabel("Alcohol-associated estimate at no CD (baseline SD), 95% CI; filled: Holm p < .05;\n"
                  "'Δ≠0': participant-bootstrap 95% CI of the difference excludes 0", color=INK_MUTED, fontsize=8)
    _legend(ax, [(l, c, m) for _, l, c, m in series], loc="upper center", bbox_to_anchor=(0.45, 1.13), ncol=1)
    fig.tight_layout()
    _save(fig, "fig_v3_naive_vs_full")


def v3_task_plot():
    est = pd.read_csv(REVISION / "task_model_estimands.csv")
    series = [("nback_sober", "Auditory 1-back (sober drivers)", SERIES[0][0], "o"),
              ("sentence_sober", "Sentence comprehension (sober drivers)", SERIES[2][0], "^")]
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    measures = GAZE + VEHICLE
    _forest(ax, est, series, measures)
    diff = est[est["estimand"] == "nback_minus_sentence_sober"].set_index("measure")
    for i, m in enumerate(measures):
        if diff.loc[m, "p_holm"] < 0.05:
            ax.text(1.4, len(measures) - 1 - i, "differ*", va="center", fontsize=7, color=INK)
    ax.set_xlim(-1.7, 1.7)
    ax.set_xlabel("Effect vs no task (baseline SD), 95% CI; filled: Holm p < .05; 'differ*': task difference Holm p < .05",
                  color=INK_MUTED, fontsize=8)
    _legend(ax, [(l, c, m) for _, l, c, m in series], loc="upper center", bbox_to_anchor=(0.45, 1.1), ncol=2)
    fig.tight_layout()
    _save(fig, "fig_v3_task")


def v3_cohort_fa_plot():
    fa = pd.read_csv(REVISION / "cohort_false_alarms_v3.csv")
    models = list(dict.fromkeys(fa["model"]))
    cells = [("rate Alcohol cohort R1 (trip-weighted)", "Alcohol cohort, R1 (sober)", SERIES[1][0], "s"),
             ("rate Control cohort R1 (trip-weighted)", "Control cohort, R1", INK_MUTED, "o"),
             ("rate Control cohort R2 (trip-weighted)", "Control cohort, R2", "#b4b2ab", "^")]
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    n = len(models)
    for j, (q, label, colour, marker) in enumerate(cells):
        for i, model in enumerate(models):
            row = fa[(fa["model"] == model) & (fa["quantity"] == q)].iloc[0]
            y = n - 1 - i + (1 - j) * 0.22
            ax.plot([row["ci_low"] * 100, row["ci_high"] * 100], [y, y], color=colour, linewidth=2, solid_capstyle="round")
            ax.plot(row["estimate"] * 100, y, marker=marker, markersize=8, color=colour, linestyle="none",
                    label=label if i == 0 else None)
    ax.set_yticks(range(n))
    ax.set_yticklabels(list(reversed(models)), color=INK)
    ax.set_xlim(0, 80)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("Sober trips classified as alcohol or combined (%), trip-weighted, 95% participant-bootstrap CI",
                  color=INK_MUTED, fontsize=8)
    _style(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, "fig_v3_cohort_fa")


def main():
    for f in [v3_effects_plot, v3_naive_vs_full_plot, v3_task_plot, v3_cohort_fa_plot]:
        if (REVISION / "full_model_estimands.csv").exists():
            f()
    cohort_shortcut_plot()
    naive_vs_corrected_plot()
    task_specific_plot()
    design_schematic()
    forest_plot()
    order_did_plot()
    order_diagnostics()
    window_comparison()
    hazard_plot()
    model_comparison_plot()
    print(f"figures written to {FIGURES}")


if __name__ == "__main__":
    main()
