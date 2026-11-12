import numpy as np
import pandas as pd

from src.study.models import holm, mixed_effects, nested_cv
from src.study.trips import WINDOWS, label_for


def test_label_mapping():
    assert label_for(cd=False, intoxicated=False) == "baseline"
    assert label_for(cd=True, intoxicated=False) == "cognitive_distraction"
    assert label_for(cd=False, intoxicated=True) == "alcohol_intoxication"
    assert label_for(cd=True, intoxicated=True) == "combined"


def test_pre_hazard_windows_exclude_last_30_seconds():
    for name, (end_offset_s, duration_s) in WINDOWS.items():
        if name.startswith("pre"):
            assert end_offset_s >= 30.0
            assert duration_s > 0


def test_holm_matches_reference():
    p = pd.Series([0.01, 0.04, 0.03], index=["a", "b", "c"])
    adjusted = holm(p)
    assert np.allclose(adjusted[["a", "c", "b"]].to_numpy(), [0.03, 0.06, 0.06])


def test_nested_cv_keeps_test_participants_out_of_training():
    rng = np.random.default_rng(0)
    rows = []
    for fold in range(5):
        for participant in range(3):
            for trip in range(6):
                label = "a" if trip % 2 else "b"
                rows.append(
                    {
                        "participant_name": f"p{fold}_{participant}",
                        "scenario_path": f"s{trip}",
                        "fixed_fold": fold,
                        "x1": rng.normal(1.0 if label == "a" else -1.0),
                        "x2": rng.normal(),
                        "y": label,
                    }
                )
    data = pd.DataFrame(rows)
    predictions, log = nested_cv(data, ["x1", "x2"], "y")
    assert len(predictions) == len(data)
    assert set(log["outer_fold"]) == set(range(5))
    for fold in range(5):
        fold_participants = set(data.loc[data.fixed_fold == fold, "participant_name"])
        predicted = set(predictions.loc[predictions.fixed_fold == fold, "participant_name"])
        assert predicted == fold_participants
    assert (log["outer_uar"] > 0.7).all()


def test_mixed_effects_recovers_effect_and_reports_mde():
    rng = np.random.default_rng(1)
    rows = []
    for participant in range(30):
        intercept = rng.normal(0, 0.5)
        for trip in range(8):
            cd = trip % 2
            rows.append({"participant_name": f"p{participant}", "cd": cd,
                         "y": intercept + 0.8 * cd + rng.normal(0, 1.0)})
    data = pd.DataFrame(rows)
    reference = data["cd"] == 0
    result = mixed_effects(data, ["y"], "cd", ["cd"], reference).iloc[0]
    assert result["ci_low"] < 0.8 < result["ci_high"]
    assert np.isclose(result["mde_80"], 2.80 * result["se"])
    assert 0.0 <= result["icc_null_model"] <= 1.0
