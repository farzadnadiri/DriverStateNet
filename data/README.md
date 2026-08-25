# Data

This project is built on Toyota Research Institute's **Impaired Driving Dataset (IDD)**:
https://toyotaresearchinstitute.github.io/IDD/

Neither the raw dataset nor TRI's data-pipeline code (`dataset/` in earlier local
copies of this repo) is redistributed here:

- The raw recordings (driving-sim telemetry, gaze tracking, and front-camera video from
  human participants performing distraction/intoxication scenarios) are third-party
  research data with their own access terms, obtained by following TRI's own
  instructions.
- The pipeline code that downloads and indexes it (TRI's `dbm`/DriverStateNet
  `dataset/` submodule) is vendored from https://github.com/toyotaresearchinstitute
  with no LICENSE file, so its terms for redistribution aren't established -- it isn't
  copied into this repo.

Everything in `src/` and `notebooks/` here is original code written on top of that
dataset; it does not require TRI's code to be present except to rebuild the cached
feature/window tensors from raw scenarios (see below).

## Reproducing from raw data

1. Clone TRI's DBM pipeline into `dataset/` at the project root, and follow its own
   `README.md` to set up the environment and download IDD (`no-video` and
   `only-scaled-rgb-video` versions are what this project uses).
2. Point it at a local `data/IDD/Processed` directory (this project's notebooks default
   to `<project_root>/data/IDD`, and expect `Resampled_previous_10` /
   `Resampled_previous_10_derived` subdirectories -- see `src/data_bootstrap.py`).
3. Run `notebooks/01_dataDownload.ipynb` to build and cache the scenario manifest, then
   `02_generic_baseline.ipynb` / `02b_handcrafted_baseline.ipynb` / `03_temporal_transformer.ipynb`
   / `04_multimodal.ipynb` in order. Each notebook caches its engineered features/windows
   under `artifacts/<notebook>/window_cache/` (or `scenario_window_features.*` /
   `handcrafted_feature_table.*`) so it only needs to touch the raw dataset once.

## What's committed instead

This repo ships the *aggregate* outputs of that pipeline -- metrics, confusion
matrices, bootstrap confidence intervals, and figures -- which contain no raw
recordings, imagery, or per-participant rows. Row-level `predictions.csv` files and the
dataset-derived feature/window caches are gitignored because they carry per-scenario
participant identifiers and intoxication labels; see the comments in `.gitignore`.
