#!/usr/bin/env bash
# Resume the prepared paper study; retain completed batches and archive checks.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$workspace_root"

study_dir="Dissertation/reconstructions/conditioning_ablation_paper"
python_bin=".venv-py313/bin/python"

"$python_bin" -u Dissertation/scripts/conditioning_ablation.py run \
  --out "$study_dir" --device cuda --batch-size 8 --cpu-threads 4 \
  2>&1 | tee -a "$study_dir/resume.log"

# Only summarize after inference exits successfully; incomplete studies fail validation.
"$python_bin" -u Dissertation/scripts/conditioning_ablation.py summarize \
  --out "$study_dir" \
  2>&1 | tee -a "$study_dir/resume.log"
