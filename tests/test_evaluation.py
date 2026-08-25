from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation import bootstrap_uar_ci, paired_bootstrap_delta, soft_vote_ensemble

LABELS = ["a", "b"]


def _predictions_frame(participant_true_pred: list[tuple[str, str, str]]) -> pd.DataFrame:
    """Build a minimal predictions.csv-shaped frame from (participant, true, pred) rows."""
    return pd.DataFrame(
        {
            "participant_name": [row[0] for row in participant_true_pred],
            "true_label": [row[1] for row in participant_true_pred],
            "pred_label": [row[2] for row in participant_true_pred],
        }
    )


def test_bootstrap_uar_ci_perfect_predictions_has_zero_width_ci():
    rows = [(f"p{i}", "a" if i % 2 == 0 else "b", "a" if i % 2 == 0 else "b") for i in range(10)]
    frame = _predictions_frame(rows)

    result = bootstrap_uar_ci(frame, labels=LABELS, n_boot=200, seed=0)

    assert result["uar"] == pytest.approx(1.0)
    assert result["ci_low"] == pytest.approx(1.0)
    assert result["ci_high"] == pytest.approx(1.0)
    assert result["n_groups"] == 10


def test_bootstrap_uar_ci_matches_macro_recall_point_estimate():
    # 4 of class "a" (3 correct), 4 of class "b" (2 correct) -> macro recall = mean(3/4, 2/4) = 0.625
    rows = [
        ("p0", "a", "a"),
        ("p1", "a", "a"),
        ("p2", "a", "a"),
        ("p3", "a", "b"),
        ("p4", "b", "b"),
        ("p5", "b", "b"),
        ("p6", "b", "a"),
        ("p7", "b", "a"),
    ]
    frame = _predictions_frame(rows)

    result = bootstrap_uar_ci(frame, labels=LABELS, n_boot=500, seed=1)

    assert result["uar"] == pytest.approx(0.625)
    assert result["ci_low"] <= result["uar"] <= result["ci_high"]


def test_paired_bootstrap_delta_detects_a_clear_winner():
    participants = [f"p{i}" for i in range(12)]
    true_labels = ["a", "b"] * 6

    # model_a is always correct, model_b is always wrong -- the gap should be
    # unambiguous even through bootstrap resampling.
    perfect = _predictions_frame(list(zip(participants, true_labels, true_labels)))
    flipped = ["b" if label == "a" else "a" for label in true_labels]
    always_wrong = _predictions_frame(list(zip(participants, true_labels, flipped)))

    result = paired_bootstrap_delta(
        perfect, always_wrong, model_a_name="perfect", model_b_name="always_wrong", labels=LABELS, n_boot=500, seed=2
    )

    assert result["uar_a"] == pytest.approx(1.0)
    assert result["uar_b"] == pytest.approx(0.0)
    assert result["delta"] == pytest.approx(1.0)
    assert result["significant"] is True
    assert result["ci_low"] > 0.0


def test_paired_bootstrap_delta_requires_matching_participants():
    a = _predictions_frame([("p0", "a", "a"), ("p1", "b", "b")])
    b = _predictions_frame([("p0", "a", "a"), ("p2", "b", "b")])

    with pytest.raises(ValueError):
        paired_bootstrap_delta(a, b, labels=LABELS)


def test_soft_vote_ensemble_averages_probabilities_and_picks_argmax():
    join_keys = ["participant_name"]
    model_1 = pd.DataFrame(
        {
            "participant_name": ["p0", "p1"],
            "true_label": ["a", "b"],
            "proba__a": [0.9, 0.4],
            "proba__b": [0.1, 0.6],
        }
    )
    model_2 = pd.DataFrame(
        {
            "participant_name": ["p0", "p1"],
            "true_label": ["a", "b"],
            "proba__a": [0.7, 0.2],
            "proba__b": [0.3, 0.8],
        }
    )

    result = soft_vote_ensemble({"m1": model_1, "m2": model_2}, labels=LABELS, join_keys=join_keys)

    predictions = result["predictions"]
    expected_proba_a = np.array([0.8, 0.3])  # mean of [0.9,0.7] and [0.4,0.2]
    np.testing.assert_allclose(predictions["proba__a"].to_numpy(), expected_proba_a)
    assert predictions["pred_label"].tolist() == ["a", "b"]
    assert result["overall_metrics"]["num_models"] == 2


def test_soft_vote_ensemble_requires_at_least_two_models():
    model_1 = pd.DataFrame(
        {
            "participant_name": ["p0"],
            "true_label": ["a"],
            "proba__a": [0.9],
            "proba__b": [0.1],
        }
    )

    with pytest.raises(ValueError):
        soft_vote_ensemble({"m1": model_1}, labels=LABELS, join_keys=["participant_name"])


def test_soft_vote_ensemble_raises_on_missing_columns():
    model_1 = pd.DataFrame({"participant_name": ["p0"], "true_label": ["a"], "proba__a": [0.9]})
    model_2 = pd.DataFrame(
        {
            "participant_name": ["p0"],
            "true_label": ["a"],
            "proba__a": [0.9],
            "proba__b": [0.1],
        }
    )

    with pytest.raises(ValueError):
        soft_vote_ensemble({"m1": model_1, "m2": model_2}, labels=LABELS, join_keys=["participant_name"])
