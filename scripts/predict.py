"""CLI demo for the project's best model (handcrafted-feature logistic regression).

Usage:
    python scripts/predict.py --input examples/sample_windows.csv
    python scripts/predict.py --input examples/sample_windows.csv --output predictions_out.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import pandas as pd

MODEL_PATH = _ROOT / "artifacts" / "handcrafted_baseline" / "final_model.joblib"


def load_model(model_path: Path = MODEL_PATH) -> tuple[object, list[str]]:
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found. Run `python scripts/fit_final_model.py` first.")
    bundle = joblib.load(model_path)
    return bundle["pipeline"], bundle["feature_columns"]


def predict(features: pd.DataFrame, pipeline=None, feature_columns=None) -> pd.DataFrame:
    if pipeline is None or feature_columns is None:
        pipeline, feature_columns = load_model()

    missing = [column for column in feature_columns if column not in features.columns]
    if missing:
        raise ValueError(f"Input is missing required feature columns: {missing}")

    predicted_labels = pipeline.predict(features[feature_columns])
    probabilities = pipeline.predict_proba(features[feature_columns])
    class_order = pipeline.named_steps["model"].classes_

    output = features.copy()
    output["predicted_label"] = predicted_labels
    for class_idx, label in enumerate(class_order):
        output[f"proba__{label}"] = probabilities[:, class_idx]
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV with the 10 feat__* handcrafted columns")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to save predictions as CSV")
    args = parser.parse_args()

    features = pd.read_csv(args.input)
    result = predict(features)

    display_columns = ["predicted_label"] + [c for c in result.columns if c.startswith("proba__")]
    if "true_label" in result.columns:
        display_columns = ["true_label"] + display_columns
    print(result[display_columns].to_string(index=False))

    if args.output is not None:
        result.to_csv(args.output, index=False)
        print(f"\nSaved full output to {args.output}")


if __name__ == "__main__":
    main()
