#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
study="reconstructions/image_free_baseline_clean"
mkdir -p "$study"
exec 9>"$study/launcher.lock"
flock -n 9 || { echo 'Image-free baseline is already running' >&2; exit 1; }
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/roughnet-mpl"
export OMP_NUM_THREADS=4
python_bin="${PYTHON_BIN:-../.venv-py313/bin/python}"
if [[ ! -f "$study/manifest.json" ]]; then
    "$python_bin" -u scripts/image_free_baseline.py prepare --out "$study" 2>&1 | tee -a "$study/run.log"
fi
"$python_bin" -u scripts/image_free_baseline.py run --out "$study" 2>&1 | tee -a "$study/run.log"
"$python_bin" -u scripts/image_free_baseline.py summarize --out "$study" 2>&1 | tee -a "$study/run.log"
