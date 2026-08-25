from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .data_bootstrap import LABEL_ORDER, engineer_multimodal_sample

MERGE_KEYS = [
    "participant_name",
    "scenario_name",
    "round",
    "cognitive_task",
    "intoxicated",
]

METADATA_COLUMNS = [
    "sample_idx",
    "participant_name",
    "scenario_name",
    "round",
    "cognitive_task",
    "intoxicated",
    "impairment_label",
    "label_id",
    "fixed_fold",
    "scenario_code",
    "scenario_path",
    "duration_s",
    "num_frames",
    "chunk_start_ns",
    "chunk_end_ns",
]


def _normalize_merge_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in ["participant_name", "scenario_name", "round", "cognitive_task"]:
        if column in df.columns:
            df[column] = df[column].astype(str)
    if "intoxicated" in df.columns:
        df["intoxicated"] = df["intoxicated"].astype(bool)
    return df


def _series_to_feature_dict(series: pd.Series, suffix: str) -> dict[str, float]:
    output: dict[str, float] = {}
    for column, value in series.sort_index().items():
        output[f"feat__{column}__{suffix}"] = float(value)
    return output


def summarize_engineered_window(engineered: dict[str, Any]) -> dict[str, float]:
    values = engineered["filled_values"]
    masks = engineered["masks"]
    delta_s = engineered["delta_s"]

    summary: dict[str, float] = {}
    summary.update(_series_to_feature_dict(values.mean(axis=0), "mean"))
    summary.update(_series_to_feature_dict(values.std(axis=0).fillna(0.0), "std"))
    summary.update(_series_to_feature_dict(values.min(axis=0), "min"))
    summary.update(_series_to_feature_dict(values.max(axis=0), "max"))
    summary.update(_series_to_feature_dict(masks.mean(axis=0), "obs_rate"))
    summary.update(_series_to_feature_dict(delta_s.mean(axis=0), "delta_mean"))
    summary.update(_series_to_feature_dict(delta_s.max(axis=0), "delta_max"))
    return summary


def build_scenario_feature_table(
    dataset,
    manifest: pd.DataFrame | None = None,
    target_fps: int = 10,
    modality_names: list[str] | None = None,
    progress_every: int = 25,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    total = len(dataset)
    for sample_idx in range(total):
        sample = dataset[sample_idx]
        try:
            engineered = engineer_multimodal_sample(
                sample,
                target_fps=target_fps,
                modality_names=modality_names,
            )
            row = {
                "sample_idx": sample_idx,
                **engineered["metadata"],
                **summarize_engineered_window(engineered),
            }
            rows.append(row)
        finally:
            if hasattr(sample, "close"):
                sample.close()

        if progress_every and ((sample_idx + 1) % progress_every == 0 or sample_idx + 1 == total):
            print(f"Built features for {sample_idx + 1}/{total} scenario windows")

    feature_table = pd.DataFrame(rows)

    if manifest is not None and len(feature_table) > 0:
        manifest = _normalize_merge_keys(manifest)
        feature_table = _normalize_merge_keys(feature_table)

        extra_columns = [
            column
            for column in ["fixed_fold", "scenario_code", "scenario_path", "duration_s"]
            if column in manifest.columns
        ]
        lookup = manifest[MERGE_KEYS + extra_columns].drop_duplicates(subset=MERGE_KEYS)
        feature_table = feature_table.merge(lookup, on=MERGE_KEYS, how="left")

    feature_columns = feature_columns_from_table(feature_table)
    ordered_columns = [
        column
        for column in METADATA_COLUMNS
        if column in feature_table.columns
    ] + feature_columns
    other_columns = [column for column in feature_table.columns if column not in ordered_columns]
    return feature_table[ordered_columns + other_columns]


def feature_columns_from_table(feature_table: pd.DataFrame) -> list[str]:
    return sorted(column for column in feature_table.columns if column.startswith("feat__"))


def confusion_matrix_frame(
    y_true,
    y_pred,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    if labels is None:
        labels = LABEL_ORDER
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(matrix, index=labels, columns=labels)


def classification_report_frame(
    y_true,
    y_pred,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    if labels is None:
        labels = LABEL_ORDER
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        },
        index=labels,
    )


def _make_classifier(
    model_name: str,
    c_value: float,
    class_weight: str | dict[str, float] | None,
    max_iter: int,
):
    if model_name == "logreg":
        return LogisticRegression(
            C=float(c_value),
            class_weight=class_weight,
            max_iter=int(max_iter),
            random_state=0,
            solver="lbfgs",
        )
    if model_name == "linear_svm":
        return LinearSVC(
            C=float(c_value),
            class_weight=class_weight,
            max_iter=int(max_iter),
            random_state=0,
            dual="auto",
        )
    raise ValueError(f"Unknown model_name: {model_name}")


def _predict_proba(pipeline: Pipeline, features: pd.DataFrame, labels: list[str]) -> np.ndarray:
    """Per-class probabilities for ensembling, even for models without predict_proba.

    LogisticRegression exposes predict_proba directly. LinearSVC does not, so its
    one-vs-rest decision_function scores are converted to probabilities via softmax
    (this does not change LinearSVC's own predictions, only adds a probability view
    used for soft-vote ensembling elsewhere).
    """
    model = pipeline.named_steps["model"]
    class_order = list(model.classes_)
    if hasattr(model, "predict_proba"):
        raw = pipeline.predict_proba(features)
    else:
        scores = pipeline.decision_function(features)
        scores = np.atleast_2d(scores)
        if scores.shape[1] == 1:  # binary decision_function returns a single column
            scores = np.hstack([-scores, scores])
        exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
        raw = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    proba = np.zeros((raw.shape[0], len(labels)), dtype=np.float64)
    for label_idx, label in enumerate(class_order):
        if label in labels:
            proba[:, labels.index(label)] = raw[:, label_idx]
    return proba


def _display_model_name(model_name: str) -> str:
    if model_name == "logreg":
        return "logistic_regression"
    if model_name == "linear_svm":
        return "linear_svm"
    return model_name


def run_fixed_fold_baseline(
    feature_table: pd.DataFrame,
    feature_columns: list[str] | None = None,
    label_column: str = "impairment_label",
    fold_column: str = "fixed_fold",
    labels: list[str] | None = None,
    model_name: str = "logreg",
    c_value: float = 1.0,
    class_weight: str | dict[str, float] | None = "balanced",
    max_iter: int = 2000,
) -> dict[str, Any]:
    if labels is None:
        labels = LABEL_ORDER
    if feature_columns is None:
        feature_columns = feature_columns_from_table(feature_table)

    if len(feature_columns) == 0:
        raise ValueError("No feature columns were found. Expected columns starting with 'feat__'.")
    if fold_column not in feature_table.columns:
        raise ValueError(f"Missing fold column: {fold_column}")
    if feature_table[fold_column].isna().any():
        raise ValueError(f"Fold column '{fold_column}' contains missing values.")

    folds = sorted(int(fold) for fold in feature_table[fold_column].unique())
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    confusion_by_fold: dict[int, pd.DataFrame] = {}
    display_name = _display_model_name(model_name)

    for fold in folds:
        train_df = feature_table.loc[feature_table[fold_column] != fold].copy()
        test_df = feature_table.loc[feature_table[fold_column] == fold].copy()

        train_labels = set(train_df[label_column].astype(str))
        missing_labels = [label for label in labels if label not in train_labels]
        if missing_labels:
            raise ValueError(
                f"Training folds for held-out fold {fold} are missing labels: {missing_labels}"
            )

        pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", _make_classifier(model_name, c_value, class_weight, max_iter)),
            ]
        )

        pipeline.fit(train_df[feature_columns], train_df[label_column])
        predictions = pipeline.predict(test_df[feature_columns])
        proba = _predict_proba(pipeline, test_df[feature_columns], labels)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            test_df[label_column],
            predictions,
            labels=labels,
            average="macro",
            zero_division=0,
        )

        fold_confusion = confusion_matrix_frame(test_df[label_column], predictions, labels=labels)
        confusion_by_fold[fold] = fold_confusion

        fold_rows.append(
            {
                "model_name": display_name,
                "fixed_fold": fold,
                "num_train": int(len(train_df)),
                "num_test": int(len(test_df)),
                "uar": float(recall_macro),
                "precision_macro": float(precision_macro),
                "recall_macro": float(recall_macro),
                "f1_macro": float(f1_macro),
                "accuracy": float(accuracy_score(test_df[label_column], predictions)),
            }
        )

        fold_predictions = pd.DataFrame(
            {
                "model_name": display_name,
                "fixed_fold": fold,
                "participant_name": test_df["participant_name"].to_numpy(),
                "scenario_name": test_df["scenario_name"].to_numpy(),
                "round": test_df["round"].to_numpy(),
                "cognitive_task": test_df["cognitive_task"].to_numpy(),
                "intoxicated": test_df["intoxicated"].to_numpy(),
                "true_label": test_df[label_column].to_numpy(),
                "pred_label": predictions,
            }
        )
        for label_idx, label in enumerate(labels):
            fold_predictions[f"proba__{label}"] = proba[:, label_idx]
        prediction_rows.append(fold_predictions)

    fold_metrics = pd.DataFrame(fold_rows).sort_values("fixed_fold").reset_index(drop=True)
    predictions_df = pd.concat(prediction_rows, ignore_index=True)
    overall_confusion = confusion_matrix_frame(
        predictions_df["true_label"],
        predictions_df["pred_label"],
        labels=labels,
    )
    per_class_report = classification_report_frame(
        predictions_df["true_label"],
        predictions_df["pred_label"],
        labels=labels,
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        predictions_df["true_label"],
        predictions_df["pred_label"],
        labels=labels,
        average="macro",
        zero_division=0,
    )

    overall_metrics = pd.Series(
        {
            "model_name": display_name,
            "uar": float(recall_macro),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "f1_macro": float(f1_macro),
            "accuracy": float(accuracy_score(predictions_df["true_label"], predictions_df["pred_label"])),
            "num_samples": int(len(predictions_df)),
            "num_features": int(len(feature_columns)),
            "num_folds": int(len(folds)),
        },
        name="overall",
    )

    return {
        "feature_columns": feature_columns,
        "fold_metrics": fold_metrics,
        "predictions": predictions_df,
        "overall_metrics": overall_metrics,
        "overall_confusion": overall_confusion,
        "per_class_report": per_class_report,
        "confusion_by_fold": confusion_by_fold,
    }


def run_fixed_fold_logistic_baseline(
    feature_table: pd.DataFrame,
    feature_columns: list[str] | None = None,
    label_column: str = "impairment_label",
    fold_column: str = "fixed_fold",
    labels: list[str] | None = None,
    c_value: float = 1.0,
    class_weight: str | dict[str, float] | None = "balanced",
    max_iter: int = 2000,
) -> dict[str, Any]:
    return run_fixed_fold_baseline(
        feature_table=feature_table,
        feature_columns=feature_columns,
        label_column=label_column,
        fold_column=fold_column,
        labels=labels,
        model_name="logreg",
        c_value=c_value,
        class_weight=class_weight,
        max_iter=max_iter,
    )


def run_fixed_fold_linear_svm_baseline(
    feature_table: pd.DataFrame,
    feature_columns: list[str] | None = None,
    label_column: str = "impairment_label",
    fold_column: str = "fixed_fold",
    labels: list[str] | None = None,
    c_value: float = 1.0,
    class_weight: str | dict[str, float] | None = "balanced",
    max_iter: int = 4000,
) -> dict[str, Any]:
    return run_fixed_fold_baseline(
        feature_table=feature_table,
        feature_columns=feature_columns,
        label_column=label_column,
        fold_column=fold_column,
        labels=labels,
        model_name="linear_svm",
        c_value=c_value,
        class_weight=class_weight,
        max_iter=max_iter,
    )
