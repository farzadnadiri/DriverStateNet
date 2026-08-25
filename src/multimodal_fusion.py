from __future__ import annotations

import contextlib
import copy
import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .classical_baseline import classification_report_frame, confusion_matrix_frame
from .data_bootstrap import LABEL_ORDER, engineer_multimodal_sample
from .temporal_transformer import (
    build_criterion,
    build_weighted_sampler,
    choose_device,
    require_torch,
    set_seed,
)

try:  # pragma: no cover - depends on local torch install
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - depends on local torch install
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

try:  # pragma: no cover - depends on local torchvision install
    from torchvision import models as tv_models
except Exception:  # pragma: no cover - depends on local torchvision install
    tv_models = None

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


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
    "num_video_frames_available",
]


def _normalize_merge_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in ["participant_name", "scenario_name", "round", "cognitive_task"]:
        if column in df.columns:
            df[column] = df[column].astype(str)
    if "intoxicated" in df.columns:
        df["intoxicated"] = df["intoxicated"].astype(bool)
    return df


def _compute_class_weights(y: np.ndarray, num_classes: int):
    torch_module = require_torch()
    counts = np.bincount(y.astype(np.int64), minlength=num_classes).astype(np.float32)
    counts[counts == 0.0] = 1.0
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return torch_module.tensor(weights, dtype=torch_module.float32)


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


def _sample_video_frames(
    sample,
    num_video_frames: int,
    image_size: int,
    video_feature_name: str = "cam_front",
) -> tuple[np.ndarray, np.ndarray, int]:
    clip = np.zeros((num_video_frames, 3, image_size, image_size), dtype=np.uint8)
    valid_mask = np.zeros(num_video_frames, dtype=bool)

    video = sample.features.get(video_feature_name)
    if video is None or not hasattr(video, "timing_data") or video.timing_data.empty:
        return clip, valid_mask, 0

    timing = video.timing_data
    available = len(timing)
    if available == 0:
        return clip, valid_mask, 0

    frame_indices = np.linspace(0, available - 1, num=min(num_video_frames, available), dtype=int)
    if len(frame_indices) < num_video_frames:
        frame_indices = np.pad(frame_indices, (0, num_video_frames - len(frame_indices)), mode="edge")

    reader = video.get_frame_reader()
    try:
        for slot_idx, frame_idx in enumerate(frame_indices[:num_video_frames]):
            time_ns = int(timing.index[int(frame_idx)])
            frame_data = reader.read_at_time(time_ns)
            if frame_data is None:
                continue
            frame = frame_data.image_data
            if frame is None or frame.ndim < 2:
                continue
            if frame.ndim == 2:
                frame = np.repeat(frame[..., None], 3, axis=2)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
            clip[slot_idx] = np.transpose(frame, (2, 0, 1)).astype(np.uint8)
            valid_mask[slot_idx] = True
    finally:
        reader.close()

    return clip, valid_mask, int(available)


def build_multimodal_window_cache(
    dataset,
    manifest: pd.DataFrame | None = None,
    target_fps: int = 10,
    modality_names: list[str] | None = None,
    num_video_frames: int = 6,
    image_size: int = 64,
    video_feature_name: str = "cam_front",
    progress_every: int = 10,
) -> dict[str, Any]:
    model_inputs: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, Any]] = []
    feature_names: set[str] = set()
    sequence_lengths: list[int] = []
    video_clips: list[np.ndarray] = []
    video_masks: list[np.ndarray] = []

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
            clip, clip_valid_mask, available_frames = _sample_video_frames(
                sample,
                num_video_frames=num_video_frames,
                image_size=image_size,
                video_feature_name=video_feature_name,
            )
            model_inputs.append(model_input)
            feature_names.update(model_input.columns.tolist())
            sequence_lengths.append(int(model_input.shape[0]))
            video_clips.append(clip)
            video_masks.append(clip_valid_mask)
            metadata_rows.append(
                {
                    "sample_idx": sample_idx,
                    **engineered["metadata"],
                    "num_video_frames_available": int(available_frames),
                }
            )
        finally:
            if hasattr(sample, "close"):
                sample.close()

        if progress_every and ((sample_idx + 1) % progress_every == 0 or sample_idx + 1 == total):
            print(f"Built multimodal windows for {sample_idx + 1}/{total} scenario windows")

    if not model_inputs:
        raise ValueError("No multimodal windows were created from the dataset.")

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

    tabular_x = np.zeros((num_samples, max_len, num_features), dtype=np.float32)
    tabular_valid_mask = np.zeros((num_samples, max_len), dtype=bool)

    for sample_idx, model_input in enumerate(model_inputs):
        aligned = model_input.reindex(columns=ordered_feature_names, fill_value=0.0)
        values = aligned.to_numpy(dtype=np.float32)
        length = values.shape[0]
        tabular_x[sample_idx, :length, :] = values
        tabular_valid_mask[sample_idx, :length] = True

    video_x = np.stack(video_clips, axis=0).astype(np.uint8)
    video_valid_mask = np.stack(video_masks, axis=0).astype(bool)

    if "fixed_fold" not in metadata.columns:
        raise ValueError("Multimodal cache metadata is missing fixed_fold.")

    y = metadata["label_id"].astype(np.int64).to_numpy()
    folds = metadata["fixed_fold"].astype(int).to_numpy()

    ordered_columns = [column for column in METADATA_COLUMNS if column in metadata.columns]
    other_columns = [column for column in metadata.columns if column not in ordered_columns]
    metadata = metadata[ordered_columns + other_columns]

    return {
        "tabular_x": tabular_x,
        "tabular_valid_mask": tabular_valid_mask,
        "video_x": video_x,
        "video_valid_mask": video_valid_mask,
        "y": y,
        "folds": folds,
        "sequence_lengths": np.asarray(sequence_lengths, dtype=np.int64),
        "feature_names": ordered_feature_names,
        "metadata": metadata,
        "label_names": LABEL_ORDER,
        "target_fps": int(target_fps),
        "num_video_frames": int(num_video_frames),
        "image_size": int(image_size),
        "video_feature_name": str(video_feature_name),
    }


def save_multimodal_window_cache(cache: dict[str, Any], cache_dir: str | Path) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        cache_dir / "multimodal_cache.npz",
        tabular_x=cache["tabular_x"],
        tabular_valid_mask=cache["tabular_valid_mask"],
        video_x=cache["video_x"],
        video_valid_mask=cache["video_valid_mask"],
        y=cache["y"],
        folds=cache["folds"],
        sequence_lengths=cache["sequence_lengths"],
    )
    cache["metadata"].to_pickle(cache_dir / "multimodal_metadata.pkl")

    with (cache_dir / "feature_names.json").open("w", encoding="utf-8") as handle:
        json.dump(cache["feature_names"], handle, indent=2)

    with (cache_dir / "cache_info.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "label_names": cache.get("label_names", LABEL_ORDER),
                "target_fps": int(cache.get("target_fps", 10)),
                "num_video_frames": int(cache.get("num_video_frames", 6)),
                "image_size": int(cache.get("image_size", 64)),
                "video_feature_name": cache.get("video_feature_name", "cam_front"),
            },
            handle,
            indent=2,
        )


def load_multimodal_window_cache(cache_dir: str | Path) -> dict[str, Any]:
    cache_dir = Path(cache_dir)
    arrays = np.load(cache_dir / "multimodal_cache.npz")
    metadata = pd.read_pickle(cache_dir / "multimodal_metadata.pkl")

    with (cache_dir / "feature_names.json").open("r", encoding="utf-8") as handle:
        feature_names = json.load(handle)

    info = {
        "label_names": LABEL_ORDER,
        "target_fps": 10,
        "num_video_frames": 6,
        "image_size": 64,
        "video_feature_name": "cam_front",
    }
    info_path = cache_dir / "cache_info.json"
    if info_path.exists():
        with info_path.open("r", encoding="utf-8") as handle:
            info.update(json.load(handle))

    return {
        "tabular_x": arrays["tabular_x"],
        "tabular_valid_mask": arrays["tabular_valid_mask"].astype(bool),
        "video_x": arrays["video_x"].astype(np.uint8),
        "video_valid_mask": arrays["video_valid_mask"].astype(bool),
        "y": arrays["y"].astype(np.int64),
        "folds": arrays["folds"].astype(np.int64),
        "sequence_lengths": arrays["sequence_lengths"].astype(np.int64),
        "feature_names": feature_names,
        "metadata": metadata,
        "label_names": info["label_names"],
        "target_fps": int(info["target_fps"]),
        "num_video_frames": int(info["num_video_frames"]),
        "image_size": int(info["image_size"]),
        "video_feature_name": str(info["video_feature_name"]),
    }


def multimodal_cache_summary(cache: dict[str, Any]) -> pd.DataFrame:
    metadata = cache["metadata"]
    return pd.DataFrame(
        {
            "num_samples": [int(cache["tabular_x"].shape[0])],
            "tabular_sequence_length": [int(cache["tabular_x"].shape[1])],
            "tabular_num_features": [int(cache["tabular_x"].shape[2])],
            "num_video_frames": [int(cache["video_x"].shape[1])],
            "image_size": [int(cache["video_x"].shape[-1])],
            "num_participants": [int(metadata["participant_name"].nunique())],
            "num_folds": [int(metadata["fixed_fold"].nunique())],
            "target_fps": [int(cache.get("target_fps", 10))],
        }
    )


if nn is not None:

    class LightweightSceneEncoder(nn.Module):
        def __init__(self, output_dim: int = 64, dropout: float = 0.1):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.GELU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.GELU(),
                nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(96),
                nn.GELU(),
                nn.Conv2d(96, output_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(output_dim),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            x = self.features(x)
            x = x.flatten(start_dim=1)
            return self.dropout(x)


    class PretrainedSceneEncoder(nn.Module):
        """ImageNet-pretrained frame encoder with a small trainable projection head.

        The from-scratch CNN (LightweightSceneEncoder) has to learn visual features from
        only ~250-300 training clips, which is why the video-only model scores near chance
        (see artifacts/multimodal_fusion/all_folds/video_only). Starting from ImageNet
        features and only training a projection head (backbone frozen by default) needs far
        less data to produce a usable representation.
        """

        _BACKBONE_INPUT_SIZE = {"resnet18": 224, "mobilenet_v3_small": 224}
        _BACKBONE_FEATURE_DIM = {"resnet18": 512, "mobilenet_v3_small": 576}

        def __init__(
            self,
            output_dim: int = 64,
            dropout: float = 0.1,
            backbone: str = "resnet18",
            freeze_backbone: bool = True,
        ):
            super().__init__()
            if tv_models is None:
                raise RuntimeError(
                    "torchvision is required for the pretrained video encoder. "
                    "Install it via `pip install torchvision` (see requirements.txt)."
                )
            if backbone not in self._BACKBONE_INPUT_SIZE:
                raise ValueError(f"Unsupported video backbone: {backbone!r}")

            self.backbone_name = backbone
            self.input_size = self._BACKBONE_INPUT_SIZE[backbone]
            self.frozen = bool(freeze_backbone)

            if backbone == "resnet18":
                net = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
                self.backbone = nn.Sequential(*list(net.children())[:-1])
            else:
                net = tv_models.mobilenet_v3_small(weights=tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
                self.backbone = nn.Sequential(net.features, nn.AdaptiveAvgPool2d(1))

            if self.frozen:
                for param in self.backbone.parameters():
                    param.requires_grad = False
                self.backbone.eval()

            feature_dim = self._BACKBONE_FEATURE_DIM[backbone]
            self.project = nn.Sequential(
                nn.Linear(feature_dim, output_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

        def train(self, mode: bool = True):
            super().train(mode)
            if self.frozen:
                self.backbone.eval()
            return self

        def forward(self, x):
            if x.shape[-2] != self.input_size or x.shape[-1] != self.input_size:
                x = nn.functional.interpolate(
                    x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False
                )
            x = (x - self.mean) / self.std
            grad_context = torch.no_grad() if self.frozen else contextlib.nullcontext()
            with grad_context:
                features = self.backbone(x)
            features = features.flatten(start_dim=1)
            if self.frozen:
                features = features.detach()
            return self.project(features)


    class MultimodalFusionClassifier(nn.Module):
        def __init__(
            self,
            tabular_input_dim: int,
            num_classes: int,
            model_dim: int = 64,
            num_heads: int = 4,
            tabular_num_layers: int = 1,
            video_num_layers: int = 1,
            ff_dim: int = 128,
            dropout: float = 0.2,
            max_tabular_len: int = 512,
            max_video_frames: int = 8,
            mode: str = "fusion",
            video_backbone: str = "resnet18",
            freeze_video_backbone: bool = True,
        ):
            super().__init__()
            if mode not in {"fusion", "tabular_only", "video_only"}:
                raise ValueError(f"Unsupported multimodal mode: {mode}")
            self.mode = mode

            needs_tabular = mode in {"fusion", "tabular_only"}
            needs_video = mode in {"fusion", "video_only"}

            if needs_tabular:
                self.tabular_input_projection = nn.Linear(tabular_input_dim, model_dim)
                self.tabular_input_dropout = nn.Dropout(dropout)
                self.tabular_position_embedding = nn.Parameter(torch.zeros(1, max_tabular_len, model_dim))
                tabular_encoder_layer = nn.TransformerEncoderLayer(
                    d_model=model_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_dim,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.tabular_encoder = nn.TransformerEncoder(tabular_encoder_layer, num_layers=tabular_num_layers)

            if needs_video:
                if video_backbone == "scratch":
                    self.video_frame_encoder = LightweightSceneEncoder(output_dim=model_dim, dropout=dropout)
                else:
                    self.video_frame_encoder = PretrainedSceneEncoder(
                        output_dim=model_dim,
                        dropout=dropout,
                        backbone=video_backbone,
                        freeze_backbone=freeze_video_backbone,
                    )
                self.video_position_embedding = nn.Parameter(torch.zeros(1, max_video_frames, model_dim))
                video_encoder_layer = nn.TransformerEncoderLayer(
                    d_model=model_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_dim,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.video_encoder = nn.TransformerEncoder(video_encoder_layer, num_layers=video_num_layers)

            if mode == "fusion":
                self.tabular_to_video = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout, batch_first=True)
                self.video_to_tabular = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout, batch_first=True)
                self.tabular_fusion_norm = nn.LayerNorm(model_dim)
                self.video_fusion_norm = nn.LayerNorm(model_dim)

            self.output_dropout = nn.Dropout(dropout)

            if mode == "fusion":
                self.classifier = nn.Linear(model_dim * 2, num_classes)
            else:
                self.classifier = nn.Linear(model_dim, num_classes)

        @staticmethod
        def _ensure_non_empty_mask(mask):
            safe_mask = mask.clone()
            empty_rows = ~safe_mask.any(dim=1)
            if empty_rows.any():
                safe_mask[empty_rows, 0] = True
            return safe_mask

        @staticmethod
        def _masked_mean(tokens, mask):
            valid = mask.unsqueeze(-1).type_as(tokens)
            return (tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

        def _encode_tabular(self, tabular_x, tabular_valid_mask):
            tabular_valid_mask = self._ensure_non_empty_mask(tabular_valid_mask)
            tokens = self.tabular_input_projection(tabular_x)
            tokens = self.tabular_input_dropout(tokens)
            tokens = tokens + self.tabular_position_embedding[:, : tokens.shape[1], :]
            tokens = self.tabular_encoder(tokens, src_key_padding_mask=~tabular_valid_mask)
            return tokens, tabular_valid_mask

        def _encode_video(self, video_x, video_valid_mask):
            video_valid_mask = self._ensure_non_empty_mask(video_valid_mask)
            batch_size, num_frames, channels, height, width = video_x.shape
            frames = video_x.reshape(batch_size * num_frames, channels, height, width).float() / 255.0
            frame_tokens = self.video_frame_encoder(frames).reshape(batch_size, num_frames, -1)
            frame_tokens = frame_tokens + self.video_position_embedding[:, : num_frames, :]
            frame_tokens = self.video_encoder(frame_tokens, src_key_padding_mask=~video_valid_mask)
            return frame_tokens, video_valid_mask

        def forward(self, tabular_x, tabular_valid_mask, video_x, video_valid_mask):
            if self.mode == "tabular_only":
                tabular_tokens, tabular_valid_mask = self._encode_tabular(tabular_x, tabular_valid_mask)
                pooled = self._masked_mean(tabular_tokens, tabular_valid_mask)
                return self.classifier(self.output_dropout(pooled))

            if self.mode == "video_only":
                video_tokens, video_valid_mask = self._encode_video(video_x, video_valid_mask)
                pooled = self._masked_mean(video_tokens, video_valid_mask)
                return self.classifier(self.output_dropout(pooled))

            tabular_tokens, tabular_valid_mask = self._encode_tabular(tabular_x, tabular_valid_mask)
            video_tokens, video_valid_mask = self._encode_video(video_x, video_valid_mask)

            tabular_cross, _ = self.tabular_to_video(
                query=tabular_tokens,
                key=video_tokens,
                value=video_tokens,
                key_padding_mask=~video_valid_mask,
            )
            video_cross, _ = self.video_to_tabular(
                query=video_tokens,
                key=tabular_tokens,
                value=tabular_tokens,
                key_padding_mask=~tabular_valid_mask,
            )
            tabular_tokens = self.tabular_fusion_norm(tabular_tokens + tabular_cross)
            video_tokens = self.video_fusion_norm(video_tokens + video_cross)

            pooled_tabular = self._masked_mean(tabular_tokens, tabular_valid_mask)
            pooled_video = self._masked_mean(video_tokens, video_valid_mask)
            fused = torch.cat([pooled_tabular, pooled_video], dim=1)
            return self.classifier(self.output_dropout(fused))

else:

    class LightweightSceneEncoder:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            require_torch()


    class PretrainedSceneEncoder:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            require_torch()


    class MultimodalFusionClassifier:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            require_torch()


def _make_loader(
    tabular_x: np.ndarray,
    tabular_valid_mask: np.ndarray,
    video_x: np.ndarray,
    video_valid_mask: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    sampler=None,
):
    torch_module = require_torch()
    dataset = TensorDataset(
        torch_module.from_numpy(tabular_x).float(),
        torch_module.from_numpy(tabular_valid_mask).bool(),
        torch_module.from_numpy(video_x),
        torch_module.from_numpy(video_valid_mask).bool(),
        torch_module.from_numpy(y).long(),
    )
    if sampler is not None:
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_multimodal_one_epoch(model, loader, optimizer, criterion, device: str) -> dict[str, float]:
    require_torch()
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for tabular_x, tabular_valid_mask, video_x, video_valid_mask, y_batch in loader:
        tabular_x = tabular_x.to(device)
        tabular_valid_mask = tabular_valid_mask.to(device)
        video_x = video_x.to(device)
        video_valid_mask = video_valid_mask.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(tabular_x, tabular_valid_mask, video_x, video_valid_mask)
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


def evaluate_multimodal_model(
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
        for tabular_x, tabular_valid_mask, video_x, video_valid_mask, y_batch in loader:
            tabular_x = tabular_x.to(device)
            tabular_valid_mask = tabular_valid_mask.to(device)
            video_x = video_x.to(device)
            video_valid_mask = video_valid_mask.to(device)
            y_batch = y_batch.to(device)

            logits = model(tabular_x, tabular_valid_mask, video_x, video_valid_mask)
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


def run_multimodal_fold(
    cache: dict[str, Any],
    test_fold: int,
    val_fold: int | None = None,
    batch_size: int = 8,
    epochs: int = 15,
    patience: int = 5,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    model_dim: int = 64,
    num_heads: int = 4,
    tabular_num_layers: int = 1,
    video_num_layers: int = 1,
    ff_dim: int = 128,
    dropout: float = 0.2,
    normalize_inputs: bool = True,
    mode: str = "fusion",
    loss_type: str = "ce",
    focal_gamma: float = 2.0,
    use_weighted_sampler: bool = False,
    video_backbone: str = "resnet18",
    freeze_video_backbone: bool = True,
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

    train_tabular_x = cache["tabular_x"][train_idx].copy()
    train_tabular_valid_mask = cache["tabular_valid_mask"][train_idx].copy()
    val_tabular_x = cache["tabular_x"][val_idx].copy()
    val_tabular_valid_mask = cache["tabular_valid_mask"][val_idx].copy()
    test_tabular_x = cache["tabular_x"][test_idx].copy()
    test_tabular_valid_mask = cache["tabular_valid_mask"][test_idx].copy()

    normalization_mean = None
    normalization_std = None
    if normalize_inputs:
        normalization_mean, normalization_std = _compute_feature_normalization(train_tabular_x, train_tabular_valid_mask)
        train_tabular_x = _apply_feature_normalization(train_tabular_x, train_tabular_valid_mask, normalization_mean, normalization_std)
        val_tabular_x = _apply_feature_normalization(val_tabular_x, val_tabular_valid_mask, normalization_mean, normalization_std)
        test_tabular_x = _apply_feature_normalization(test_tabular_x, test_tabular_valid_mask, normalization_mean, normalization_std)

    train_sampler = build_weighted_sampler(cache["y"][train_idx], len(label_names)) if use_weighted_sampler else None
    train_loader = _make_loader(
        train_tabular_x,
        train_tabular_valid_mask,
        cache["video_x"][train_idx],
        cache["video_valid_mask"][train_idx],
        cache["y"][train_idx],
        batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
    )
    val_loader = _make_loader(
        val_tabular_x,
        val_tabular_valid_mask,
        cache["video_x"][val_idx],
        cache["video_valid_mask"][val_idx],
        cache["y"][val_idx],
        batch_size,
        False,
    )
    test_loader = _make_loader(
        test_tabular_x,
        test_tabular_valid_mask,
        cache["video_x"][test_idx],
        cache["video_valid_mask"][test_idx],
        cache["y"][test_idx],
        batch_size,
        False,
    )

    model = MultimodalFusionClassifier(
        tabular_input_dim=int(cache["tabular_x"].shape[2]),
        num_classes=len(label_names),
        model_dim=model_dim,
        num_heads=num_heads,
        tabular_num_layers=tabular_num_layers,
        video_num_layers=video_num_layers,
        ff_dim=ff_dim,
        dropout=dropout,
        max_tabular_len=int(cache["tabular_x"].shape[1]),
        max_video_frames=int(cache["video_x"].shape[1]),
        mode=mode,
        video_backbone=video_backbone,
        freeze_video_backbone=freeze_video_backbone,
    ).to(device)

    class_weights = _compute_class_weights(cache["y"][train_idx], len(label_names)).to(device)
    criterion = build_criterion(loss_type, class_weights, focal_gamma=focal_gamma)
    optimizer = torch_module.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad), lr=lr, weight_decay=weight_decay
    )

    best_state = None
    best_epoch = 0
    best_val_uar = -np.inf
    epochs_without_improvement = 0
    history_rows: list[dict[str, Any]] = []

    for epoch in range(1, int(epochs) + 1):
        train_metrics = train_multimodal_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate_multimodal_model(model, val_loader, device, label_names=label_names, criterion=criterion)

        history_rows.append(
            {
                "epoch": epoch,
                "mode": mode,
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
    test_metrics = evaluate_multimodal_model(model, test_loader, device, label_names=label_names, criterion=criterion)
    predictions = cache["metadata"].iloc[test_idx].copy().reset_index(drop=True)
    predictions["fixed_fold"] = int(test_fold)
    predictions["validation_fold"] = int(val_fold)
    predictions["model_name"] = mode
    predictions["true_label"] = test_metrics["true_labels"]
    predictions["pred_label"] = test_metrics["pred_labels"]
    predictions["true_label_id"] = test_metrics["y_true"]
    predictions["pred_label_id"] = test_metrics["y_pred"]
    for class_idx, label in enumerate(label_names):
        predictions[f"proba__{label}"] = test_metrics["y_prob"][:, class_idx]

    fold_metrics = pd.Series(
        {
            "model_name": mode,
            "fixed_fold": int(test_fold),
            "validation_fold": int(val_fold),
            "num_train": int(len(train_idx)),
            "num_val": int(len(val_idx)),
            "num_test": int(len(test_idx)),
            "best_epoch": int(best_epoch),
            "normalize_inputs": bool(normalize_inputs),
            "loss_type": loss_type,
            "use_weighted_sampler": bool(use_weighted_sampler),
            "video_backbone": video_backbone,
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


def run_fixed_fold_multimodal_experiment(
    cache: dict[str, Any],
    mode: str = "fusion",
    folds_to_run: list[int] | None = None,
    batch_size: int = 8,
    epochs: int = 15,
    patience: int = 5,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    model_dim: int = 64,
    num_heads: int = 4,
    tabular_num_layers: int = 1,
    video_num_layers: int = 1,
    ff_dim: int = 128,
    dropout: float = 0.2,
    normalize_inputs: bool = True,
    loss_type: str = "ce",
    focal_gamma: float = 2.0,
    use_weighted_sampler: bool = False,
    video_backbone: str = "resnet18",
    freeze_video_backbone: bool = True,
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
        print(f"Running {mode} multimodal model for held-out fold {test_fold}")
        result = run_multimodal_fold(
            cache=cache,
            test_fold=int(test_fold),
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            lr=lr,
            weight_decay=weight_decay,
            model_dim=model_dim,
            num_heads=num_heads,
            tabular_num_layers=tabular_num_layers,
            video_num_layers=video_num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            normalize_inputs=normalize_inputs,
            mode=mode,
            loss_type=loss_type,
            focal_gamma=focal_gamma,
            use_weighted_sampler=use_weighted_sampler,
            video_backbone=video_backbone,
            freeze_video_backbone=freeze_video_backbone,
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
            "model_name": mode,
            "uar": float(recall_macro),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "f1_macro": float(f1_macro),
            "accuracy": float(accuracy_score(predictions_df["true_label"], predictions_df["pred_label"])),
            "num_samples": int(len(predictions_df)),
            "tabular_num_features": int(cache["tabular_x"].shape[2]),
            "tabular_sequence_length": int(cache["tabular_x"].shape[1]),
            "num_video_frames": int(cache["video_x"].shape[1]),
            "image_size": int(cache["video_x"].shape[-1]),
            "num_folds": int(len(fold_metrics)),
            "normalize_inputs": bool(normalize_inputs),
            "loss_type": loss_type,
            "use_weighted_sampler": bool(use_weighted_sampler),
            "video_backbone": video_backbone,
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
