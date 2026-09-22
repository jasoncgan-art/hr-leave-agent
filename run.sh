#!/usr/bin/env bash
# Quickstart helper: install deps, run tests, run eval.
set -euo pipefail
cd "$(dirname "$0")"
python -m pip install -q -r requirements.txt
python -m pytest -q
python eval/run_eval.py
