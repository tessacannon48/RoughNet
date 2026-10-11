#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
study="reconstructions/conditioning_ablation_clean"
mkdir -p "$study"
exec 9>"$study/study.lock"
flock -n 9 || { echo 'Conditioning study already running' >&2; exit 1; }
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/roughnet-mpl"
export OMP_NUM_THREADS=4
python_bin="${PYTHON_BIN:-../.venv-py313/bin/python}"
if [[ ! -f "$study/manifest.json" ]]; then
    "$python_bin" -u scripts/prepare_corrected_conditioning.py --out "$study" \
        2>&1 | tee -a "$study/run.log"
fi
"$python_bin" -u scripts/conditioning_ablation.py run --out "$study" --device cuda --batch-size 8 --cpu-threads 4 \
    2>&1 | tee -a "$study/run.log"
"$python_bin" -u scripts/conditioning_ablation.py summarize --out "$study" \
    2>&1 | tee -a "$study/run.log"
