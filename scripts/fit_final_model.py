"""Fit the project's best model (handcrafted-feature logistic regression) on all
available data and persist it for the CLI/API demo.

The notebooks only ever fit per-fold models for cross-validated evaluation (see
artifacts/handcrafted_baseline/logistic_regression/); there was previously no single
"final" model saved to disk. This script reuses the exact same Pipeline construction
(median-impute -> standardize -> LogisticRegression) so the demo mirrors the evaluated
model rather than introducing a new one.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import pandas as pd

from src.classical_baseline import _make_classifier
from src.handcrafted_baseline import handcrafted_feature_columns_from_table
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    root = _ROOT
    feature_table_path = root / "artifacts" / "handcrafted_baseline" / "handcrafted_feature_table.csv"
    if not feature_table_path.exists():
        raise FileNotFoundError(
            f"{feature_table_path} not found. Run notebooks/02b_handcrafted_baseline.ipynb first "
            "to build the handcrafted feature table."
        )

    feature_table = pd.read_csv(feature_table_path)
    feature_columns = handcrafted_feature_columns_from_table(feature_table)

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", _make_classifier("logreg", c_value=1.0, class_weight="balanced", max_iter=2000)),
        ]
    )
    pipeline.fit(feature_table[feature_columns], feature_table["impairment_label"])

    output_path = root / "artifacts" / "handcrafted_baseline" / "final_model.joblib"
    joblib.dump({"pipeline": pipeline, "feature_columns": feature_columns}, output_path)
    print(f"Fit on {len(feature_table)} rows across all folds -> saved {output_path}")


if __name__ == "__main__":
    main()
