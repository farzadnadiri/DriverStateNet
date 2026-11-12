"""Stage 2b: matched model comparison on the primary pre90 window (four-class task).

    python -m scripts.study.run_model_comparison [--skip-transformer]

Classical models (logistic regression, linear SVM, gradient boosting) use TRI's 10 features;
the temporal transformer uses the 84-channel value/mask/delta sequences of the same window.
All use the same 371 trips, the same five participant folds, and nested selection.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.study.run_analyses import PRIVATE, RESULTS, _ci_row, _paired, load_window
from src.study.features import TRI_COLUMNS
from src.study.models import nested_cv
from src.study.trips import LABEL_ORDER

ROOT = Path(__file__).resolve().parents[2]
CLASSICAL = {"logreg": "Logistic regression", "linear_svm": "Linear SVM", "gbm": "Gradient boosting"}


def run_transformer(trips: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    import torch

    from src.study.sequence import build_sequence_cache, nested_transformer

    cache = build_sequence_cache(ROOT / "dataset", ROOT / "data/IDD", 30.0, 90.0, PRIVATE / "seq_pre90")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"transformer on {device}, cache {cache['x'].shape}")
    preds, log = nested_transformer(cache, device=device)
    keys = ["participant_name", "scenario_name", "round", "cognitive_task"]
    preds = preds.copy()
    for key in ["participant_name", "scenario_name", "round", "cognitive_task"]:
        preds[key] = preds[key].astype(str)
    lookup = trips[keys + ["scenario_path"]].astype(str)
    preds = preds.drop(columns=[c for c in ["scenario_path"] if c in preds.columns]).merge(lookup, on=keys, how="left")
    return preds[["participant_name", "scenario_path", "fixed_fold", "true_label", "pred_label"]], log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-transformer", action="store_true")
    args = parser.parse_args()

    data = load_window("pre90")
    trips = pd.read_pickle(PRIVATE / "trips.pkl")
    predictions, rows, logs = {}, [], []
    for model, name in CLASSICAL.items():
        preds, log = nested_cv(data, TRI_COLUMNS, "label", model=model)
        preds.to_csv(PRIVATE / f"pred_model_{model}_pre90.csv", index=False)
        predictions[name] = preds
        log.insert(0, "model", name)
        logs.append(log)
        rows.append(_ci_row(name, preds, LABEL_ORDER, input="TRI 10 features"))
        print(f"{name}: UAR {rows[-1]['uar']:.3f}")

    if not args.skip_transformer:
        preds, log = run_transformer(trips)
        missing = preds["scenario_path"].isna().sum()
        if missing:
            raise ValueError(f"{missing} transformer predictions could not be matched to trips")
        preds.to_csv(PRIVATE / "pred_model_transformer_pre90.csv", index=False)
        name = "Temporal transformer"
        predictions[name] = preds
        log.insert(0, "model", name)
        logs.append(log)
        rows.append(_ci_row(name, preds, LABEL_ORDER, input="84-channel sequences (900 steps)"))
        print(f"{name}: UAR {rows[-1]['uar']:.3f} on {len(preds)} trips")

    pd.DataFrame(rows).to_csv(RESULTS / "model_comparison.csv", index=False)
    pd.concat(logs, ignore_index=True).to_csv(RESULTS / "model_comparison_selection_log.csv", index=False)
    reference = "Logistic regression"
    deltas = [{**_paired(predictions[name], predictions[reference], name, reference, LABEL_ORDER)}
              for name in predictions if name != reference]
    pd.DataFrame(deltas).to_csv(RESULTS / "model_comparison_paired.csv", index=False)
    print(pd.DataFrame(deltas)[["model_a", "uar_a", "uar_b", "delta", "ci_low", "ci_high", "p_value"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
