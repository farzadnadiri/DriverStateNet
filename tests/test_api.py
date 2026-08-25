from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

pytestmark = pytest.mark.skipif(
    not (
        __import__("pathlib").Path(__file__).resolve().parent.parent
        / "artifacts"
        / "handcrafted_baseline"
        / "final_model.joblib"
    ).exists(),
    reason="run `python scripts/fit_final_model.py` first to produce the demo model",
)

VALID_PAYLOAD = {
    "gaze_fixation_change_rate": 230.0,
    "gaze_pitch_std": 4.2,
    "gaze_saccade_change_rate": 55.0,
    "gaze_yaw_std": 6.1,
    "longitudinal_acceleration_neg_mean": -0.8,
    "longitudinal_acceleration_pos_mean": 0.6,
    "longitudinal_speed_pos_mean": 8.5,
    "reversals_0_5_sum_rate": 3.1,
    "reversals_2_5_sum_rate": 0.4,
    "tobii_eye_pupil_diameter_mean": 3.9,
}


def test_health_reports_model_loaded():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"model_loaded": True}


def test_predict_returns_a_label_and_normalized_probabilities():
    with TestClient(app) as client:
        response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_label"] in {"baseline", "cognitive_distraction", "alcohol_intoxication", "combined"}
    assert set(body["probabilities"].keys()) == {"baseline", "cognitive_distraction", "alcohol_intoxication", "combined"}
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)


def test_predict_rejects_missing_fields():
    incomplete_payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "gaze_pitch_std"}

    with TestClient(app) as client:
        response = client.post("/predict", json=incomplete_payload)

    assert response.status_code == 422
