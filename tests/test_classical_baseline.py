from __future__ import annotations

import pandas as pd

from src.classical_baseline import classification_report_frame, confusion_matrix_frame, feature_columns_from_table

LABELS = ["a", "b", "c"]


def test_confusion_matrix_frame_counts_correctly():
    y_true = ["a", "a", "b", "b", "c"]
    y_pred = ["a", "b", "b", "b", "a"]

    matrix = confusion_matrix_frame(y_true, y_pred, labels=LABELS)

    assert matrix.index.tolist() == LABELS
    assert matrix.columns.tolist() == LABELS
    assert matrix.loc["a", "a"] == 1
    assert matrix.loc["a", "b"] == 1
    assert matrix.loc["b", "b"] == 2
    assert matrix.loc["c", "a"] == 1
    assert matrix.to_numpy().sum() == len(y_true)


def test_classification_report_frame_matches_hand_computed_metrics():
    y_true = ["a", "a", "b", "b"]
    y_pred = ["a", "b", "b", "b"]

    report = classification_report_frame(y_true, y_pred, labels=LABELS)

    # class "a": 1 true positive, 1 false negative -> recall 0.5, precision 1.0
    assert report.loc["a", "recall"] == 0.5
    assert report.loc["a", "precision"] == 1.0
    assert report.loc["a", "support"] == 2
    # class "b": both predicted as "b", both correct -> recall 1.0
    assert report.loc["b", "recall"] == 1.0
    # class "c" never appears -> zero_division=0 gives 0.0, not NaN or an error
    assert report.loc["c", "support"] == 0
    assert report.loc["c", "precision"] == 0.0


def test_feature_columns_from_table_only_returns_feat_prefixed_columns_sorted():
    df = pd.DataFrame(
        {
            "participant_name": ["p0"],
            "feat__zeta__mean": [1.0],
            "feat__alpha__mean": [2.0],
            "label_id": [0],
        }
    )

    columns = feature_columns_from_table(df)

    assert columns == ["feat__alpha__mean", "feat__zeta__mean"]
