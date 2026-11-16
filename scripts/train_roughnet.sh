#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
variant="${1:-cosine}"
case "$variant" in
  cosine|linear|cosine_tuk_heldout) run="roughnet_$variant" ;;
  *) echo 'Choose cosine, linear or cosine_tuk_heldout' >&2; exit 2 ;;
esac
mkdir -p "models/$run"
exec 9>"models/$run/train.lock"
flock -n 9 || { echo 'This variant is already running' >&2; exit 1; }
export WANDB_MODE=disabled
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/roughnet-mpl"
export OMP_NUM_THREADS=4
python_bin="${PYTHON_BIN:-python}"
"$python_bin" -u scripts/train_roughnet.py --config "splits/$run.yaml" --resume "${@:2}" \
  2>&1 | tee -a "models/$run/training.log"
