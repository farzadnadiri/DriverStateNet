"""Temporal transformer under the study protocol: pre-hazard window, nested selection.

Reuses the course-project code unchanged (`src.temporal_transformer`): only the window and
the model-selection loop differ. For each outer fold, each candidate configuration is scored
by leave-one-fold-out over the four training folds (each run early-stops on a further
training fold); the best configuration is then retrained on the training folds (early
stopping on one of them) and scores the untouched outer fold once.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import data_bootstrap, temporal_transformer

# Candidate configurations. A = the course project's tuned setting (notebook 03), which was
# chosen on the same folds; B = a smaller model. Nested selection picks between them per fold.
TRANSFORMER_CONFIGS = {
    "A_dim64": dict(model_dim=64, num_heads=4, num_layers=1, ff_dim=128, dropout=0.3, lr=5e-4, weight_decay=1e-3),
    "B_dim32": dict(model_dim=32, num_heads=4, num_layers=1, ff_dim=64, dropout=0.3, lr=5e-4, weight_decay=1e-3),
}
TRAINING = dict(batch_size=16, epochs=30, patience=8, normalize_inputs=True, loss_type="ce", use_weighted_sampler=False)


def build_sequence_cache(dbm_repo: str | Path, processed_root: str | Path, end_offset_s: float, duration_s: float,
                         cache_dir: str | Path) -> dict:
    """Value/mask/delta sequences (84 channels, 10 Hz) for one window, cached to disk."""
    cache_dir = Path(cache_dir)
    if (cache_dir / "window_cache.npz").exists():
        return temporal_transformer.load_temporal_window_cache(cache_dir)
    manifest = data_bootstrap.build_scenario_manifest(dbm_repo, processed_root, r"driving/[0-9].*")
    dataset = data_bootstrap.build_inspection_dataset(
        dbm_repo=dbm_repo,
        processed_root=processed_root,
        chunk_strategy="end",
        chunk_duration_s=duration_s,
        chunks_per_scenario=1,
        chunk_end_offset_s=end_offset_s,
        include_video=False,
    )
    cache = temporal_transformer.build_temporal_window_cache(dataset, manifest=manifest, target_fps=10)
    temporal_transformer.save_temporal_window_cache(cache, cache_dir)
    return cache


def _subset(cache: dict, keep: np.ndarray) -> dict:
    out = dict(cache)
    for key in ["x", "valid_mask", "y", "folds", "sequence_lengths"]:
        out[key] = cache[key][keep]
    out["metadata"] = cache["metadata"].iloc[np.where(keep)[0]].reset_index(drop=True)
    return out


def _run(cache: dict, test_fold: int, config: dict, device: str, seed: int) -> dict:
    return temporal_transformer.run_temporal_transformer_fold(
        cache=cache, test_fold=test_fold, device=device, seed=seed, **TRAINING, **config
    )


def nested_transformer(cache: dict, device: str, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    folds = cache["folds"].astype(int)
    predictions, log = [], []
    for outer in sorted(np.unique(folds)):
        inner_cache = _subset(cache, folds != outer)
        inner_scores = {}
        for name, config in TRANSFORMER_CONFIGS.items():
            scores = []
            for inner in sorted(np.unique(inner_cache["folds"])):
                result = _run(inner_cache, int(inner), config, device, seed)
                scores.append(float(result["fold_metrics"]["uar"]))
            inner_scores[name] = float(np.mean(scores))
            print(f"outer {outer} config {name}: inner UAR {inner_scores[name]:.3f}")
        best = max(inner_scores, key=inner_scores.get)
        final = _run(cache, int(outer), TRANSFORMER_CONFIGS[best], device, seed)
        predictions.append(final["predictions"])
        log.append({"outer_fold": int(outer), "best_config": best, **{f"inner_uar__{k}": v for k, v in inner_scores.items()},
                    "outer_uar": float(final["fold_metrics"]["uar"]), "best_epoch": int(final["fold_metrics"]["best_epoch"])})
        print(f"outer {outer}: chose {best}, outer UAR {log[-1]['outer_uar']:.3f}")
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(log)
