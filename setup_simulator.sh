#!/usr/bin/env bash
set -euo pipefail
SIM_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SIM_ROOT"
git submodule update --init --recursive
"${EMILYX_PYTHON:-python3}" -m venv .venv
.venv/bin/python -m pip install --disable-pip-version-check -r requirements.lock.txt
.venv/bin/python -m pip check
mkdir -p .cache/matplotlib runs srtm_data
.venv/bin/python scripts/check_inputs.py --mode dataset
echo 'Ready. See README.md for dataset generation commands.'
