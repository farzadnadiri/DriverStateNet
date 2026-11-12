"""Stage 1: trip index, exclusion audit, and window features for the study.

    python -m scripts.study.build_features

Writes row-level tables to artifacts/study/private/ (gitignored) and aggregate audit
tables to artifacts/study/audit/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.study.features import TRI_COLUMNS, build_window_features, driver_centre_prc
from src.study.trips import audit_tables, build_trip_table

ROOT = Path(__file__).resolve().parents[2]


def validate_against_cached(features: pd.DataFrame, trips: pd.DataFrame) -> pd.DataFrame:
    """Compare haz30 TRI features with the course-report cache (same DBM chunk)."""
    cache_path = ROOT / "artifacts/handcrafted_baseline/handcrafted_feature_table.csv"
    if not cache_path.exists():
        return pd.DataFrame()
    cached = pd.read_csv(cache_path, dtype={"participant_name": str})
    cached["key"] = cached["participant_name"] + "|" + cached["scenario_path"]
    ours = features[features["window"] == "haz30"].copy()
    ours["key"] = ours["participant_name"] + "|" + ours["scenario_path"]
    merged = ours.merge(cached, on="key", suffixes=("", "_cached"))
    rows = []
    for column in TRI_COLUMNS:
        cached_column = "feat__" + column.removeprefix("tri__")
        a, b = merged[column].to_numpy(float), merged[cached_column].to_numpy(float)
        ok = ~np.isnan(a) & ~np.isnan(b)
        rows.append(
            {
                "feature": column,
                "n_matched": int(ok.sum()),
                "max_abs_diff": float(np.max(np.abs(a[ok] - b[ok]))) if ok.any() else np.nan,
                "corr": float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dbm-repo", default=str(ROOT / "dataset"))
    parser.add_argument("--processed-root", default=str(ROOT / "data/IDD"))
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()

    private = ROOT / "artifacts/study/private"
    audit_dir = ROOT / "artifacts/study/audit"
    private.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    trips = build_trip_table(args.dbm_repo, args.processed_root)
    trips.to_pickle(private / "trips.pkl")
    for name, table in audit_tables(trips).items():
        table.to_csv(audit_dir / f"{name}.csv")

    study_trips = trips[~trips["removed_participant"]]
    features = build_window_features(study_trips, args.dbm_repo, args.processed_root, n_jobs=args.n_jobs)
    prc_driver = driver_centre_prc(study_trips, args.processed_root, n_jobs=args.n_jobs)
    features = features.merge(prc_driver, on=["participant_name", "scenario_path", "window"], how="left")
    features.to_pickle(private / "window_features.pkl")

    validation = validate_against_cached(features, trips)
    validation.to_csv(audit_dir / "tri_feature_validation_vs_cache.csv", index=False)
    print(validation.to_string(index=False))
    print(f"{len(study_trips)} study trips, {len(features)} trip-windows written to {private}")


if __name__ == "__main__":
    main()
