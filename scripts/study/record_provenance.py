"""Provenance record for the study analysis.

    python -m scripts.study.record_provenance

Writes artifacts/study/results/revision/provenance.json and environment.txt: software versions,
hashes of the input annotations, folds and every analysis script, seeds, and the analysis
chronology. Hashes let a reader confirm that a rerun used the same inputs and code.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/study/results/revision"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    (OUT / "environment.txt").write_text(f"python {sys.version}\nplatform {platform.platform()}\n\n{freeze}", encoding="utf-8")
    # The DBM code is a vendored copy; only report a commit if it is its own checkout.
    dbm_commit = subprocess.run(["git", "-C", str(ROOT / "dataset"), "rev-parse", "HEAD"], capture_output=True, text=True)
    if not (ROOT / "dataset/.git").exists():
        dbm_commit.stdout = "vendored copy (not a separate git checkout); see data/README.md"
    scripts = sorted((ROOT / "scripts/study").glob("*.py")) + sorted((ROOT / "src/study").glob("*.py"))
    record = {
        "recorded": date.today().isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "dbm_code_commit": dbm_commit.stdout.strip() or "not a git checkout",
        "inputs": {
            "idd_annotation.csv": sha256(ROOT / "data/IDD/idd_annotation.csv"),
            "folds.csv (participant folds)": sha256(ROOT / "dataset/hail-datasets/hail_datasets/datasets/ddd_2024/folds.csv"),
            "dataset version": "IDD public release, 'no-video' variant, downloaded via the DBM tools into data/IDD",
        },
        "seeds": {"classification and transformer": 42, "revision bootstraps": 20261029, "false-alarm bootstraps": 20261029},
        "inclusion": "DBM defaults: 8 pilot/incomplete participants removed; trips with any annotated simulator anomaly removed (416 -> 371)",
        "windows": {"pre90": "[-120 s, -30 s] before trip end", "pre30": "[-60 s, -30 s]", "haz30": "last 30 s"},
        "scripts_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in scripts if p.exists()},
        "chronology": [
            "Apr 2026: course-project experiments on the hazard window (classification only).",
            "27 Oct 2026: hypotheses H1-H5 and windows written (docs/HYPOTHESES.md) before pre-hazard analyses; primary constrained model, robustness, hazard response, model comparison.",
            "28 Oct 2026: exploratory additions - task-specific CD, naive vs corrected, full model, collision GEE, cohort false alarms.",
            "29 Oct 2026: revision - full factorial model promoted to main analysis; exclusion audit; inclusion, gaze-quality and log-scale sensitivity; participant bootstraps; hardened false-alarm diagnostics. All revision choices were made after earlier results were known.",
        ],
    }
    (OUT / "provenance.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"wrote {OUT / 'provenance.json'}")


if __name__ == "__main__":
    main()
