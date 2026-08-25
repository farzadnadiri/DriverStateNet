from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .classical_baseline import classification_report_frame, confusion_matrix_frame
from .data_bootstrap import LABEL_ORDER, engineer_multimodal_sample

_TORCH_IMPORT_ERROR: Exception | None = None

try:  # pragma: no cover - depends on local torch install
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
except Exception as exc:  # pragma: no cover - depends on local torch install
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    WeightedRandomSampler = None
    _TORCH_IMPORT_ERROR = exc


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


def require_torch():
    if torch is None or _TORCH_IMPORT_ERROR is not None:
        raise RuntimeError(
            "PyTorch import failed in the current environment. "
            "Fix the local torch installation before running the temporal-transformer notebook."
        ) from _TORCH_IMPORT_ERROR
    return torch


def _normalize_merge_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in ["participant_name", "scenario_name", "round", "cognitive_task"]:
        if column in df.columns:
            df[column] = df[column].astype(str)
    if "intoxicated" in df.columns:
        df["intoxicated"] = df["intoxicated"].astype(bool)
    return df


def build_temporal_window_cache(
    dataset,
    manifest: pd.DataFrame | None = None,
    target_fps: int = 10,
    modality_names: list[str] | None = None,
    progress_every: int = 25,
) -> dict[str, Any]:
    model_inputs: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, Any]] = []
    feature_names: set[str] = set()
    sequence_lengths: list[int] = []

    total = len(dataset)
    for sample_idx in range(total):
        sample = dataset[sample_idx]
        try:
            engineered = engineer_multimodal_sample(
                sample,
                target_fps=target_fps,
                modality_names=modality_names,
            )
            model_input = engineered["model_input"].copy()
            model_inputs.append(model_input)
            feature_names.update(model_input.columns.tolist())
            sequence_lengths.append(int(model_input.shape[0]))
            metadata_rows.append(
                {
                    "sample_idx": sample_idx,
                    **engineered["metadata"],
                }
            )
        finally:
            if hasattr(sample, "close"):
                sample.close()

        if progress_every and ((sample_idx + 1) % progress_every == 0 or sample_idx + 1 == total):
            print(f"Built temporal windows for {sample_idx + 1}/{total} scenario windows")

    if not model_inputs:
        raise ValueError("No temporal windows were created from the dataset.")

    metadata = pd.DataFrame(metadata_rows)
    if manifest is not None and len(metadata) > 0:
        metadata = _normalize_merge_keys(metadata)
        manifest = _normalize_merge_keys(manifest)
        extra_columns = [
            column
            for column in ["fixed_fold", "scenario_code", "scenario_path", "duration_s"]
            if column in manifest.columns
        ]
        lookup = manifest[MERGE_KEYS + extra_columns].drop_duplicates(subset=MERGE_KEYS)
        metadata = metadata.merge(lookup, on=MERGE_KEYS, how="left")

    ordered_feature_names = sorted(feature_names)
    max_len = max(sequence_lengths)
    num_samples = len(model_inputs)
    num_features = len(ordered_feature_names)

    x = np.zeros((num_samples, max_len, num_features), dtype=np.float32)
    valid_mask = np.zeros((num_samples, max_len), dtype=bool)

    for sample_idx, model_input in enumerate(model_inputs):
        aligned = model_input.reindex(columns=ordered_feature_names, fill_value=0.0)
        values = aligned.to_numpy(dtype=np.float32)
        length = values.shape[0]
        x[sample_idx, :length, :] = values
        valid_mask[sample_idx, :length] = True

    if "fixed_fold" not in metadata.columns:
        raise ValueError("Temporal cache metadata is missing fixed_fold.")

    y = metadata["label_id"].astype(np.int64).to_numpy()
    folds = metadata["fixed_fold"].astype(int).to_numpy()

    ordered_columns = [column for column in METADATA_COLUMNS if column in metadata.columns]
    other_columns = [column for column in metadata.columns if column not in ordered_columns]
    metadata = metadata[ordered_columns + other_columns]

    return {
        "x": x,
        "valid_mask": valid_mask,
        "y": y,
        "folds": folds,
        "sequence_lengths": np.asarray(sequence_lengths, dtype=np.int64),
        "feature_names": ordered_feature_names,
        "metadata": metadata,
        "label_names": LABEL_ORDER,
        "target_fps": int(target_fps),
    }


def save_temporal_window_cache(cache: dict[str, Any], cache_dir: str | Path) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        cache_dir / "window_cache.npz",
        x=cache["x"],
        valid_mask=cache["valid_mask"],
        y=cache["y"],
        folds=cache["folds"],
        sequence_lengths=cache["sequence_lengths"],
    )
    cache["metadata"].to_pickle(cache_dir / "window_metadata.pkl")

    with (cache_dir / "feature_names.json").open("w", encoding="utf-8") as handle:
        json.dump(cache["feature_names"], handle, indent=2)

    with (cache_dir / "cache_info.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "label_names": cache.get("label_names", LABEL_ORDER),
                "target_fps": int(cache.get("target_fps", 10)),
            },
            handle,
            indent=2,
        )


def load_temporal_window_cache(cache_dir: str | Path) -> dict[str, Any]:
    cache_dir = Path(cache_dir)
    arrays = np.load(cache_dir / "window_cache.npz")
    metadata = pd.read_pickle(cache_dir / "window_metadata.pkl")

    with (cache_dir / "feature_names.json").open("r", encoding="utf-8") as handle:
        feature_names = json.load(handle)

    info = {"label_names": LABEL_ORDER, "target_fps": 10}
    info_path = cache_dir / "cache_info.json"
    if info_path.exists():
        with info_path.open("r", encoding="utf-8") as handle:
            info.update(json.load(handle))

    return {
        "x": arrays["x"],
        "valid_mask": arrays["valid_mask"].astype(bool),
        "y": arrays["y"].astype(np.int64),
        "folds": arrays["folds"].astype(np.int64),
        "sequence_lengths": arrays["sequence_lengths"].astype(np.int64),
        "feature_names": feature_names,
        "metadata": metadata,
        "label_names": info["label_names"],
        "target_fps": int(info["target_fps"]),
    }


def temporal_cache_summary(cache: dict[str, Any]) -> pd.DataFrame:
    metadata = cache["metadata"]
    return pd.DataFrame(
        {
            "num_samples": [int(cache["x"].shape[0])],
            "sequence_length": [int(cache["x"].shape[1])],
            "num_features": [int(cache["x"].shape[2])],
            "num_participants": [int(metadata["participant_name"].nunique())],
            "num_folds": [int(metadata["fixed_fold"].nunique())],
            "target_fps": [int(cache.get("target_fps", 10))],
        }
    )


if nn is not None:

    class TemporalTransformerClassifier(nn.Module):
        def __init__(
            self,
            input_dim: int,
            num_classes: int,
            model_dim: int = 128,
            num_heads: int = 4,
            num_layers: int = 2,
            ff_dim: int = 256,
            dropout: float = 0.1,
            max_len: int = 512,
        ):
            super().__init__()
            self.input_projection = nn.Linear(input_dim, model_dim)
            self.input_dropout = nn.Dropout(dropout)
            self.position_embedding = nn.Parameter(torch.zeros(1, max_len, model_dim))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output_norm = nn.LayerNorm(model_dim)
            self.classifier = nn.Linear(model_dim, num_classes)

        def forward(self, x, valid_mask):
            x = self.input_projection(x)
            x = self.input_dropout(x)
            x = x + self.position_embedding[:, : x.shape[1], :]
            encoded = self.encoder(x, src_key_padding_mask=~valid_mask)
            valid = valid_mask.unsqueeze(-1).type_as(encoded)
            pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
            pooled = self.output_norm(pooled)
            return self.classifier(pooled)

    class FocalLoss(nn.Module):
        """Cross-entropy with a (1 - p_t) ** gamma down-weighting of easy examples.

        Used instead of plain weighted CE to push harder on the minority classes
        (alcohol_intoxication / combined), which class weighting alone under-corrects
        for on this dataset (see per-class recall in the baseline reports).
        """

        def __init__(self, weight=None, gamma: float = 2.0, reduction: str = "mean"):
            super().__init__()
            self.weight = weight
            self.gamma = float(gamma)
            self.reduction = reduction

        def forward(self, logits, target):
            log_probs = torch.log_softmax(logits, dim=-1)
            target_log_probs = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
            target_probs = target_log_probs.exp()
            focal_term = (1.0 - target_probs).clamp(min=0.0) ** self.gamma
            loss = -focal_term * target_log_probs

            if self.weight is not None:
                sample_weight = self.weight[target]
                loss = loss * sample_weight
                if self.reduction == "mean":
                    return loss.sum() / sample_weight.sum().clamp_min(1e-8)

            if self.reduction == "mean":
                return loss.mean()
            if self.reduction == "sum":
                return loss.sum()
            return loss

else:

    class TemporalTransformerClassifier:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            require_torch()

    class FocalLoss:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            require_torch()


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def choose_device(prefer_cuda: bool = True) -> str:
    torch_module = require_torch()
    if prefer_cuda and torch_module.cuda.is_available():
        return "cuda"
    return "cpu"


def _make_loader(
    x: np.ndarray,
    valid_mask: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    sampler=None,
):
    torch_module = require_torch()
    dataset = TensorDataset(
        torch_module.from_numpy(x).float(),
        torch_module.from_numpy(valid_mask).bool(),
        torch_module.from_numpy(y).long(),
    )
    if sampler is not None:
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _compute_class_weights(y: np.ndarray, num_classes: int):
    torch_module = require_torch()
    counts = np.bincount(y.astype(np.int64), minlength=num_classes).astype(np.float32)
    counts[counts == 0.0] = 1.0
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return torch_module.tensor(weights, dtype=torch_module.float32)


def build_weighted_sampler(y: np.ndarray, num_classes: int) -> "WeightedRandomSampler":
    """Per-example sampler that equalizes class frequency within each training batch.

    Complements (not replaces) loss-level class weighting: weighting alone still lets
    minibatches go several steps without seeing a minority-class example on a ~290-row
    training split, which this directly fixes by oversampling those rows.
    """
    torch_module = require_torch()
    counts = np.bincount(y.astype(np.int64), minlength=num_classes).astype(np.float64)
    counts[counts == 0.0] = 1.0
    class_sample_weight = 1.0 / counts
    sample_weights = class_sample_weight[y.astype(np.int64)]
    sample_weights_tensor = torch_module.tensor(sample_weights, dtype=torch_module.double)
    return WeightedRandomSampler(sample_weights_tensor, num_samples=len(sample_weights_tensor), replacement=True)


def build_criterion(loss_type: str, class_weights, focal_gamma: float = 2.0):
    if loss_type == "ce":
        return nn.CrossEntropyLoss(weight=class_weights)
    if loss_type == "focal":
        return FocalLoss(weight=class_weights, gamma=focal_gamma)
    raise ValueError(f"Unknown loss_type: {loss_type!r}. Expected 'ce' or 'focal'.")


def _compute_feature_normalization(
    x: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    feature_mask = valid_mask[..., None].astype(np.float32)
    valid_counts = feature_mask.sum(axis=(0, 1)).clip(min=1.0)
    mean = (x * feature_mask).sum(axis=(0, 1)) / valid_counts
    centered = (x - mean.reshape(1, 1, -1)) * feature_mask
    variance = (centered * centered).sum(axis=(0, 1)) / valid_counts
    std = np.sqrt(variance).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std


def _apply_feature_normalization(
    x: np.ndarray,
    valid_mask: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    normalized = (x - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    normalized = normalized.astype(np.float32, copy=False)
    normalized[~valid_mask] = 0.0
    return normalized


def train_one_epoch(model, loader, optimizer, criterion, device: str) -> dict[str, float]:
    require_torch()
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for x_batch, valid_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        valid_batch = valid_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x_batch, valid_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        predictions = logits.argmax(dim=1)
        total_loss += float(loss.item()) * int(y_batch.shape[0])
        total_correct += int((predictions == y_batch).sum().item())
        total_examples += int(y_batch.shape[0])

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
    }


def evaluate_model(
    model,
    loader,
    device: str,
    label_names: list[str] | None = None,
    criterion=None,
) -> dict[str, Any]:
    torch_module = require_torch()
    if label_names is None:
        label_names = LABEL_ORDER

    model.eval()
    total_loss = 0.0
    total_examples = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob_batches: list[np.ndarray] = []

    with torch_module.no_grad():
        for x_batch, valid_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            valid_batch = valid_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch, valid_batch)
            if criterion is not None:
                loss = criterion(logits, y_batch)
                total_loss += float(loss.item()) * int(y_batch.shape[0])

            probs = torch_module.softmax(logits, dim=-1)
            predictions = probs.argmax(dim=1)
            total_examples += int(y_batch.shape[0])
            y_true.extend(y_batch.cpu().numpy().astype(int).tolist())
            y_pred.extend(predictions.cpu().numpy().astype(int).tolist())
            y_prob_batches.append(probs.cpu().numpy())

    true_labels = [label_names[index] for index in y_true]
    pred_labels = [label_names[index] for index in y_pred]
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels,
        pred_labels,
        labels=label_names,
        average="macro",
        zero_division=0,
    )
    y_prob = (
        np.concatenate(y_prob_batches, axis=0)
        if y_prob_batches
        else np.zeros((0, len(label_names)), dtype=np.float32)
    )

    return {
        "loss": total_loss / max(total_examples, 1) if criterion is not None else np.nan,
        "accuracy": float(accuracy_score(true_labels, pred_labels)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "uar": float(recall_macro),
        "confusion": confusion_matrix_frame(true_labels, pred_labels, labels=label_names),
        "per_class_report": classification_report_frame(true_labels, pred_labels, labels=label_names),
        "y_true": np.asarray(y_true, dtype=np.int64),
        "y_pred": np.asarray(y_pred, dtype=np.int64),
        "y_prob": y_prob,
        "true_labels": true_labels,
        "pred_labels": pred_labels,
    }


def _pick_validation_fold(unique_folds: list[int], test_fold: int) -> int:
    fold_index = unique_folds.index(test_fold)
    for offset in range(1, len(unique_folds)):
        candidate = unique_folds[(fold_index + offset) % len(unique_folds)]
        if candidate != test_fold:
            return candidate
    raise ValueError("Could not choose a validation fold.")


def run_temporal_transformer_fold(
    cache: dict[str, Any],
    test_fold: int,
    val_fold: int | None = None,
    batch_size: int = 32,
    epochs: int = 20,
    patience: int = 5,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    model_dim: int = 128,
    num_heads: int = 4,
    num_layers: int = 2,
    ff_dim: int = 256,
    dropout: float = 0.1,
    normalize_inputs: bool = True,
    loss_type: str = "ce",
    focal_gamma: float = 2.0,
    use_weighted_sampler: bool = False,
    device: str | None = None,
    seed: int = 42,
    label_names: list[str] | None = None,
) -> dict[str, Any]:
    torch_module = require_torch()
    set_seed(seed)

    if label_names is None:
        label_names = cache.get("label_names", LABEL_ORDER)
    if device is None:
        device = choose_device(prefer_cuda=True)

    unique_folds = sorted(int(fold) for fold in np.unique(cache["folds"]))
    if val_fold is None:
        val_fold = _pick_validation_fold(unique_folds, int(test_fold))

    folds = cache["folds"].astype(int)
    train_idx = np.where((folds != int(test_fold)) & (folds != int(val_fold)))[0]
    val_idx = np.where(folds == int(val_fold))[0]
    test_idx = np.where(folds == int(test_fold))[0]

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Train/val/test split produced an empty subset.")

    train_x = cache["x"][train_idx].copy()
    train_valid_mask = cache["valid_mask"][train_idx].copy()
    val_x = cache["x"][val_idx].copy()
    val_valid_mask = cache["valid_mask"][val_idx].copy()
    test_x = cache["x"][test_idx].copy()
    test_valid_mask = cache["valid_mask"][test_idx].copy()

    normalization_mean = None
    normalization_std = None
    if normalize_inputs:
        normalization_mean, normalization_std = _compute_feature_normalization(train_x, train_valid_mask)
        train_x = _apply_feature_normalization(train_x, train_valid_mask, normalization_mean, normalization_std)
        val_x = _apply_feature_normalization(val_x, val_valid_mask, normalization_mean, normalization_std)
        test_x = _apply_feature_normalization(test_x, test_valid_mask, normalization_mean, normalization_std)

    train_sampler = build_weighted_sampler(cache["y"][train_idx], len(label_names)) if use_weighted_sampler else None
    train_loader = _make_loader(
        train_x, train_valid_mask, cache["y"][train_idx], batch_size, shuffle=(train_sampler is None), sampler=train_sampler
    )
    val_loader = _make_loader(val_x, val_valid_mask, cache["y"][val_idx], batch_size, False)
    test_loader = _make_loader(test_x, test_valid_mask, cache["y"][test_idx], batch_size, False)

    model = TemporalTransformerClassifier(
        input_dim=int(cache["x"].shape[2]),
        num_classes=len(label_names),
        model_dim=model_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=ff_dim,
        dropout=dropout,
        max_len=int(cache["x"].shape[1]),
    ).to(device)

    class_weights = _compute_class_weights(cache["y"][train_idx], len(label_names)).to(device)
    criterion = build_criterion(loss_type, class_weights, focal_gamma=focal_gamma)
    optimizer = torch_module.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_epoch = 0
    best_val_uar = -np.inf
    epochs_without_improvement = 0
    history_rows: list[dict[str, Any]] = []

    for epoch in range(1, int(epochs) + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate_model(model, val_loader, device, label_names=label_names, criterion=criterion)

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_uar": val_metrics["uar"],
                "val_f1_macro": val_metrics["f1_macro"],
            }
        )

        if val_metrics["uar"] > best_val_uar + 1e-6:
            best_val_uar = float(val_metrics["uar"])
            best_epoch = int(epoch)
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if patience and epochs_without_improvement >= int(patience):
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    history = pd.DataFrame(history_rows)
    test_metrics = evaluate_model(model, test_loader, device, label_names=label_names, criterion=criterion)
    predictions = cache["metadata"].iloc[test_idx].copy().reset_index(drop=True)
    predictions["fixed_fold"] = int(test_fold)
    predictions["validation_fold"] = int(val_fold)
    predictions["true_label"] = test_metrics["true_labels"]
    predictions["pred_label"] = test_metrics["pred_labels"]
    predictions["true_label_id"] = test_metrics["y_true"]
    predictions["pred_label_id"] = test_metrics["y_pred"]
    for class_idx, label in enumerate(label_names):
        predictions[f"proba__{label}"] = test_metrics["y_prob"][:, class_idx]

    fold_metrics = pd.Series(
        {
            "model_name": "temporal_transformer",
            "fixed_fold": int(test_fold),
            "validation_fold": int(val_fold),
            "num_train": int(len(train_idx)),
            "num_val": int(len(val_idx)),
            "num_test": int(len(test_idx)),
            "best_epoch": int(best_epoch),
            "normalize_inputs": bool(normalize_inputs),
            "loss_type": loss_type,
            "use_weighted_sampler": bool(use_weighted_sampler),
            "uar": float(test_metrics["uar"]),
            "precision_macro": float(test_metrics["precision_macro"]),
            "recall_macro": float(test_metrics["recall_macro"]),
            "f1_macro": float(test_metrics["f1_macro"]),
            "accuracy": float(test_metrics["accuracy"]),
        }
    )

    return {
        "history": history,
        "fold_metrics": fold_metrics,
        "predictions": predictions,
        "overall_confusion": test_metrics["confusion"],
        "per_class_report": test_metrics["per_class_report"],
        "normalization_mean": normalization_mean,
        "normalization_std": normalization_std,
    }


def run_fixed_fold_temporal_experiment(
    cache: dict[str, Any],
    folds_to_run: list[int] | None = None,
    batch_size: int = 32,
    epochs: int = 20,
    patience: int = 5,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    model_dim: int = 128,
    num_heads: int = 4,
    num_layers: int = 2,
    ff_dim: int = 256,
    dropout: float = 0.1,
    normalize_inputs: bool = True,
    loss_type: str = "ce",
    focal_gamma: float = 2.0,
    use_weighted_sampler: bool = False,
    device: str | None = None,
    seed: int = 42,
    label_names: list[str] | None = None,
) -> dict[str, Any]:
    require_torch()

    if label_names is None:
        label_names = cache.get("label_names", LABEL_ORDER)
    if folds_to_run is None:
        folds_to_run = sorted(int(fold) for fold in np.unique(cache["folds"]))

    fold_rows = []
    predictions = []
    histories = []
    confusion_by_fold: dict[int, pd.DataFrame] = {}

    for test_fold in folds_to_run:
        print(f"Running temporal transformer for held-out fold {test_fold}")
        result = run_temporal_transformer_fold(
            cache=cache,
            test_fold=int(test_fold),
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            lr=lr,
            weight_decay=weight_decay,
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            normalize_inputs=normalize_inputs,
            loss_type=loss_type,
            focal_gamma=focal_gamma,
            use_weighted_sampler=use_weighted_sampler,
            device=device,
            seed=seed,
            label_names=label_names,
        )
        fold_rows.append(result["fold_metrics"].to_dict())
        predictions.append(result["predictions"])
        history = result["history"].copy()
        history["fixed_fold"] = int(test_fold)
        histories.append(history)
        confusion_by_fold[int(test_fold)] = result["overall_confusion"]

    fold_metrics = pd.DataFrame(fold_rows).sort_values("fixed_fold").reset_index(drop=True)
    predictions_df = pd.concat(predictions, ignore_index=True)
    history_df = pd.concat(histories, ignore_index=True)

    overall_confusion = confusion_matrix_frame(
        predictions_df["true_label"],
        predictions_df["pred_label"],
        labels=label_names,
    )
    per_class_report = classification_report_frame(
        predictions_df["true_label"],
        predictions_df["pred_label"],
        labels=label_names,
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        predictions_df["true_label"],
        predictions_df["pred_label"],
        labels=label_names,
        average="macro",
        zero_division=0,
    )
    overall_metrics = pd.Series(
        {
            "model_name": "temporal_transformer",
            "uar": float(recall_macro),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "f1_macro": float(f1_macro),
            "accuracy": float(accuracy_score(predictions_df["true_label"], predictions_df["pred_label"])),
            "num_samples": int(len(predictions_df)),
            "num_features": int(cache["x"].shape[2]),
            "sequence_length": int(cache["x"].shape[1]),
            "num_folds": int(len(fold_metrics)),
            "normalize_inputs": bool(normalize_inputs),
            "loss_type": loss_type,
            "use_weighted_sampler": bool(use_weighted_sampler),
        },
        name="overall",
    )

    return {
        "fold_metrics": fold_metrics,
        "predictions": predictions_df,
        "history": history_df,
        "overall_metrics": overall_metrics,
        "overall_confusion": overall_confusion,
        "per_class_report": per_class_report,
        "confusion_by_fold": confusion_by_fold,
    }
