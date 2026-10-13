#!/usr/bin/env bash
# Runs the Update 1 pipeline end to end. Loads Shelton1013/SwitchLingua_text with
# datasets.load_dataset (gated: run `hf auth login` once) and needs requirements.txt.
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
