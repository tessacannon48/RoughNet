#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
split="${1:-test}"
case "$split" in validation|test) ;; *) echo 'Choose validation or test' >&2; exit 2;; esac
study="reconstructions/roughnet_tuk_transfer"
mkdir -p "$study"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/roughnet-mpl"
export OMP_NUM_THREADS=4
python_bin="${PYTHON_BIN:-python}"
if [[ ! -f "$study/manifest.json" ]]; then
    "$python_bin" -u scripts/alternate_fold_evaluation.py prepare --out "$study" --manifest splits/tuk_heldout_clean_portable.json --checkpoint-registry release/checkpoints.json 2>&1 | tee -a "$study/$split.log"
fi
"$python_bin" -u scripts/alternate_fold_evaluation.py run --out "$study" --split "$split" --device cuda \
    2>&1 | tee -a "$study/$split.log"
"$python_bin" -u scripts/alternate_fold_evaluation.py summarize --out "$study" --split "$split" \
    2>&1 | tee -a "$study/$split.log"
