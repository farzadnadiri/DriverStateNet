from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, recall_score

from .classical_baseline import classification_report_frame, confusion_matrix_frame
from .data_bootstrap import LABEL_ORDER

# Every prediction frame produced by classical_baseline / temporal_transformer /
# multimodal_fusion carries these columns (they all derive from the same
# engineer_multimodal_sample metadata), so they double as a stable join key for
# comparing or ensembling models trained by different notebooks.
JOIN_KEYS = ["participant_name", "scenario_name", "round", "cognitive_task", "intoxicated", "fixed_fold"]


def load_predictions(path: str | Path) -> pd.DataFrame:
    """Load a predictions.csv, restoring dtypes CSV round-tripping loses."""
    predictions = pd.read_csv(path)
    if "round" in predictions.columns:
        predictions["round"] = predictions["round"].astype(str)
    if "intoxicated" in predictions.columns:
        predictions["intoxicated"] = predictions["intoxicated"].astype(bool)
    return predictions


def _macro_recall(true_label, pred_label, labels: list[str]) -> float:
    return float(recall_score(true_label, pred_label, labels=labels, average="macro", zero_division=0))


def bootstrap_uar_ci(
    predictions: pd.DataFrame,
    labels: list[str] | None = None,
    true_col: str = "true_label",
    pred_col: str = "pred_label",
    group_col: str = "participant_name",
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Participant-level bootstrap CI for macro recall (UAR).

    Resamples whole participants, not individual windows: windows from the same
    participant are correlated, and with only 52 participants total (~7-10 per fixed
    fold) a per-window bootstrap would understate the true uncertainty. Use this before
    reporting a UAR number as if it were precise -- on this dataset the CI half-width
    is typically comparable to the gaps between models in artifacts/*/overall_metrics.csv.
    """
    if labels is None:
        labels = LABEL_ORDER
    rng = np.random.default_rng(seed)
    group_indices = predictions.groupby(group_col).indices
    groups = list(group_indices.keys())
    true_values = predictions[true_col].to_numpy()
    pred_values = predictions[pred_col].to_numpy()

    point_estimate = _macro_recall(true_values, pred_values, labels)

    num_groups = len(groups)
    scores = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sampled = rng.integers(0, num_groups, size=num_groups)
        idx = np.concatenate([group_indices[groups[j]] for j in sampled])
        scores[i] = _macro_recall(true_values[idx], pred_values[idx], labels)

    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(scores, [alpha, 1.0 - alpha])
    return {
        "uar": point_estimate,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "ci_level": ci,
        "n_boot": n_boot,
        "n_groups": num_groups,
    }


def paired_bootstrap_delta(
    predictions_a: pd.DataFrame,
    predictions_b: pd.DataFrame,
    model_a_name: str = "a",
    model_b_name: str = "b",
    labels: list[str] | None = None,
    true_col: str = "true_label",
    pred_col: str = "pred_label",
    group_col: str = "participant_name",
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap significance test for UAR_a - UAR_b.

    Resamples the same participants for both models on each draw, which is the
    correct way to compare two models evaluated on the same (small, non-i.i.d.) test
    set. Use this before claiming a model "beats" another -- e.g. the tuned
    temporal-transformer vs. linear-SVM baseline gap (0.391 vs. 0.376 UAR) is well
    within what this test calls noise at n=371 windows / 52 participants (see
    artifacts/ensemble/significance_vs_baseline.csv). The ensemble's larger gap over
    that same baseline (0.478 vs. 0.376) does clear this bar.
    """
    if labels is None:
        labels = LABEL_ORDER
    groups_a = set(predictions_a[group_col])
    groups_b = set(predictions_b[group_col])
    if groups_a != groups_b:
        raise ValueError(
            "predictions_a and predictions_b must cover the same participants "
            f"(only in a: {groups_a - groups_b}, only in b: {groups_b - groups_a})."
        )

    rng = np.random.default_rng(seed)
    idx_a = predictions_a.groupby(group_col).indices
    idx_b = predictions_b.groupby(group_col).indices
    groups = list(idx_a.keys())

    true_a = predictions_a[true_col].to_numpy()
    pred_a = predictions_a[pred_col].to_numpy()
    true_b = predictions_b[true_col].to_numpy()
    pred_b = predictions_b[pred_col].to_numpy()

    point_a = _macro_recall(true_a, pred_a, labels)
    point_b = _macro_recall(true_b, pred_b, labels)

    num_groups = len(groups)
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sampled = rng.integers(0, num_groups, size=num_groups)
        rows_a = np.concatenate([idx_a[groups[j]] for j in sampled])
        rows_b = np.concatenate([idx_b[groups[j]] for j in sampled])
        deltas[i] = _macro_recall(true_a[rows_a], pred_a[rows_a], labels) - _macro_recall(
            true_b[rows_b], pred_b[rows_b], labels
        )

    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(deltas, [alpha, 1.0 - alpha])
    p_value = min(2.0 * min(float((deltas <= 0).mean()), float((deltas >= 0).mean())), 1.0)

    return {
        "model_a": model_a_name,
        "model_b": model_b_name,
        "uar_a": point_a,
        "uar_b": point_b,
        "delta": point_a - point_b,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "ci_level": ci,
        "p_value": p_value,
        "significant": bool(lo > 0.0 or hi < 0.0),
        "n_boot": n_boot,
        "n_groups": num_groups,
    }


def soft_vote_ensemble(
    prediction_frames: dict[str, pd.DataFrame],
    labels: list[str] | None = None,
    join_keys: Sequence[str] = JOIN_KEYS,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Average per-class probabilities across independently-trained models.

    Each model's notebook already saves `proba__<label>` columns in predictions.csv,
    so combining them needs no retraining -- just row alignment and a weighted average.
    Small, differently-biased models often average out each other's errors on
    noisy/small data even when neither beats the other individually -- but which
    models to include matters more than how many: see
    artifacts/ensemble/ensemble_component_set_comparison.csv, where a curated
    four-model set clearly beats the six-model kitchen sink.
    """
    if labels is None:
        labels = LABEL_ORDER
    if len(prediction_frames) < 2:
        raise ValueError("Provide at least two model prediction frames to ensemble.")

    model_names = list(prediction_frames.keys())
    if weights is None:
        weights = {name: 1.0 for name in model_names}
    total_weight = sum(weights[name] for name in model_names)

    merged = None
    for model_name, frame in prediction_frames.items():
        proba_cols = [f"proba__{label}" for label in labels]
        missing = [c for c in [*proba_cols, *join_keys, "true_label"] if c not in frame.columns]
        if missing:
            raise ValueError(f"Model '{model_name}' predictions are missing columns: {missing}")

        subset = frame[[*join_keys, "true_label", *proba_cols]].copy()
        subset = subset.rename(columns={col: f"{model_name}__{col}" for col in proba_cols})
        if merged is None:
            merged = subset
        else:
            merged = merged.merge(
                subset.drop(columns=["true_label"]),
                on=list(join_keys),
                how="inner",
                validate="one_to_one",
            )

    if merged is None or len(merged) == 0:
        raise ValueError("No overlapping rows found across the provided prediction frames.")

    ensemble_proba = np.zeros((len(merged), len(labels)), dtype=np.float64)
    for model_name in model_names:
        proba = merged[[f"{model_name}__proba__{label}" for label in labels]].to_numpy()
        ensemble_proba += (weights[model_name] / total_weight) * proba

    pred_idx = ensemble_proba.argmax(axis=1)
    merged["pred_label"] = [labels[i] for i in pred_idx]
    for label_idx, label in enumerate(labels):
        merged[f"proba__{label}"] = ensemble_proba[:, label_idx]

    overall_confusion = confusion_matrix_frame(merged["true_label"], merged["pred_label"], labels=labels)
    per_class_report = classification_report_frame(merged["true_label"], merged["pred_label"], labels=labels)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        merged["true_label"], merged["pred_label"], labels=labels, average="macro", zero_division=0
    )

    overall_metrics = pd.Series(
        {
            "model_name": "ensemble(" + "+".join(model_names) + ")",
            "uar": float(recall_macro),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "f1_macro": float(f1_macro),
            "accuracy": float(accuracy_score(merged["true_label"], merged["pred_label"])),
            "num_samples": int(len(merged)),
            "num_models": len(model_names),
            "component_models": "+".join(model_names),
        },
        name="overall",
    )

    return {
        "predictions": merged,
        "overall_metrics": overall_metrics,
        "overall_confusion": overall_confusion,
        "per_class_report": per_class_report,
    }
