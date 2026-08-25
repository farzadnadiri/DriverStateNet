"""Minimal inference API for the project's best model (handcrafted-feature logistic
regression). Reuses the same `predict()`/`load_model()` used by scripts/predict.py so
the CLI and the API can never drift apart.

Run:
    uvicorn app.main:app --reload

Then:
    curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
      "gaze_fixation_change_rate": 230.0, "gaze_pitch_std": 4.2, "gaze_saccade_change_rate": 55.0,
      "gaze_yaw_std": 6.1, "longitudinal_acceleration_neg_mean": -0.8, "longitudinal_acceleration_pos_mean": 0.6,
      "longitudinal_speed_pos_mean": 8.5, "reversals_0_5_sum_rate": 3.1, "reversals_2_5_sum_rate": 0.4,
      "tobii_eye_pupil_diameter_mean": 3.9
    }'
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from scripts.predict import load_model, predict

_pipeline = None
_feature_columns: list[str] | None = None


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _pipeline, _feature_columns
    _pipeline, _feature_columns = load_model()
    yield


app = FastAPI(
    title="DriverStateNet demo API",
    description="Driver-impairment classification (baseline / cognitive_distraction / "
    "alcohol_intoxication / combined) from a window of handcrafted gaze, pupil, and "
    "vehicle-control features. See the project README for the full case study.",
    lifespan=_lifespan,
)


class FeatureWindow(BaseModel):
    gaze_fixation_change_rate: float = Field(..., description="feat__gaze_fixation__change_rate")
    gaze_pitch_std: float = Field(..., description="feat__gaze_pitch__std")
    gaze_saccade_change_rate: float = Field(..., description="feat__gaze_saccade__change_rate")
    gaze_yaw_std: float = Field(..., description="feat__gaze_yaw__std")
    longitudinal_acceleration_neg_mean: float = Field(..., description="feat__longitudinal_acceleration__neg_mean")
    longitudinal_acceleration_pos_mean: float = Field(..., description="feat__longitudinal_acceleration__pos_mean")
    longitudinal_speed_pos_mean: float = Field(..., description="feat__longitudinal_speed__pos_mean")
    reversals_0_5_sum_rate: float = Field(..., description="feat__reversals_0_5__sum_rate")
    reversals_2_5_sum_rate: float = Field(..., description="feat__reversals_2_5__sum_rate")
    tobii_eye_pupil_diameter_mean: float = Field(..., description="feat__tobii_eye_pupil_diameter__mean")

    def to_feature_row(self) -> dict[str, float]:
        return {_API_FIELD_TO_FEATURE_COLUMN[name]: value for name, value in self}


_API_FIELD_TO_FEATURE_COLUMN = {
    "gaze_fixation_change_rate": "feat__gaze_fixation__change_rate",
    "gaze_pitch_std": "feat__gaze_pitch__std",
    "gaze_saccade_change_rate": "feat__gaze_saccade__change_rate",
    "gaze_yaw_std": "feat__gaze_yaw__std",
    "longitudinal_acceleration_neg_mean": "feat__longitudinal_acceleration__neg_mean",
    "longitudinal_acceleration_pos_mean": "feat__longitudinal_acceleration__pos_mean",
    "longitudinal_speed_pos_mean": "feat__longitudinal_speed__pos_mean",
    "reversals_0_5_sum_rate": "feat__reversals_0_5__sum_rate",
    "reversals_2_5_sum_rate": "feat__reversals_2_5__sum_rate",
    "tobii_eye_pupil_diameter_mean": "feat__tobii_eye_pupil_diameter__mean",
}


class PredictionResponse(BaseModel):
    predicted_label: str
    probabilities: dict[str, float]


@app.get("/health")
def health() -> dict[str, bool]:
    return {"model_loaded": _pipeline is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(window: FeatureWindow) -> PredictionResponse:
    if _pipeline is None or _feature_columns is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    row = pd.DataFrame([window.to_feature_row()])
    result = predict(row, pipeline=_pipeline, feature_columns=_feature_columns)

    proba_columns = [c for c in result.columns if c.startswith("proba__")]
    probabilities = {column.removeprefix("proba__"): float(result[column].iloc[0]) for column in proba_columns}
    return PredictionResponse(predicted_label=str(result["predicted_label"].iloc[0]), probabilities=probabilities)
