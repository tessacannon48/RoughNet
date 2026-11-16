#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
config="${1:-configs/roughness.yaml}"
study="${2:-reconstructions/roughnet_roughness}"
python_bin="${PYTHON_BIN:-python}"
workers="${ROUGHNESS_WORKERS:-2}"
mkdir -p "$study"
exec 9>"$study/launcher.lock"
flock -n 9 || { echo 'Roughness analysis is already running' >&2; exit 1; }
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/roughnet-mpl"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
if [[ ! -f "$study/manifest.json" ]]; then
    "$python_bin" -u scripts/roughness_analysis.py prepare --config "$config" --out "$study" 2>&1 | tee -a "$study/run.log"
fi
"$python_bin" -u scripts/roughness_analysis.py run --out "$study" --workers "$workers" 2>&1 | tee -a "$study/run.log"
"$python_bin" -u scripts/roughness_analysis.py summarize --out "$study" 2>&1 | tee -a "$study/run.log"
"$python_bin" -u scripts/plot_roughness.py --out "$study" --config "$config" 2>&1 | tee -a "$study/run.log"
