#!/usr/bin/env bash
# Runs the Update 1 pipeline end to end. Expects the raw CSVs in ../data/SwitchLingua_text
# (or SWITCHLINGUA_DIR) and the dependencies from requirements.txt.
set -euo pipefail
cd "$(dirname "$0")"

python -m pytest -q tests
python split.py
python stats.py
python baselines.py
python dataset.py --demo
python causal_check.py
python audit.py
echo "done: see results/"
