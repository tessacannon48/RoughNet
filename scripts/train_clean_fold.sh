#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
fold="${1:-original_clean}"
case "$fold" in original_clean|original_clean_linear|tuk_heldout_clean) ;; *) echo 'Choose original_clean, original_clean_linear or tuk_heldout_clean' >&2; exit 2;; esac
mkdir -p "models/$fold"
exec 9>"models/$fold/train.lock"
flock -n 9 || { echo 'This fold is already running' >&2; exit 1; }
# Local logs only; no external W&B upload.
export WANDB_MODE=disabled
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/roughnet-mpl"
export OMP_NUM_THREADS=4
../.venv-py313/bin/python -u scripts/main.py --config "splits/$fold.yaml" --resume \
  2>&1 | tee -a "models/$fold/training.log"
