#!/usr/bin/env bash
set -euo pipefail
python src/evaluation/baselines_corrected.py
python src/evaluation/evaluate_corrected.py --limit 200
python src/evaluation/failure_analysis.py
