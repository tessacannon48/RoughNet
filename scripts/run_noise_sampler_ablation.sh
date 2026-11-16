#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
split="${1:-validation}"
case "$split" in validation|test) ;; *) echo 'Choose validation or test' >&2; exit 2;; esac
study="reconstructions/roughnet_noise_sampler"
mkdir -p "$study"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/roughnet-mpl"
export OMP_NUM_THREADS=4
python_bin="${PYTHON_BIN:-python}"
if [[ ! -f "$study/manifest.json" ]]; then
    "$python_bin" -u scripts/noise_sampler_ablation.py prepare --out "$study" --manifest splits/original_clean_portable.json --checkpoint-registry metadata/checkpoints.json 2>&1 | tee -a "$study/$split.log"
fi
"$python_bin" -u scripts/noise_sampler_ablation.py run --out "$study" --split "$split" --device cuda \
    2>&1 | tee -a "$study/$split.log"
"$python_bin" -u scripts/noise_sampler_ablation.py summarize --out "$study" --split "$split" \
    2>&1 | tee -a "$study/$split.log"
