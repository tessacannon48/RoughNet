#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${PYTHON_BIN:-python}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/roughnet-mpl"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
# Fresh training is the default. --resume is explicit and only supported for
# checkpoints trained with --num-workers 0; worker RNG states are not recoverable.
"$python_bin" -u scripts/train_unet_baseline.py "$@"
