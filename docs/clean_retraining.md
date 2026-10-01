# Retraining with footprint-separated validation

Use existing patches. Validation zone membership stays fixed; exclude training
patches whose conservative union footprint (LiDAR plus all six Sentinel-2 views,
transformed to regional LiDAR CRS) intersects any validation union footprint.
This also covers intersections between target and conditioning footprints.
Touching edges are allowed. This prevents shared footprint area, not all
spatial correlation. No test metric was used to choose these splits.

## Prepared splits

| Fold | Source region | Train | Validation | Excluded |
|---|---|---:|---:|---:|
| Both | Pond Inlet, validation zone 4 | 9,811 | 1,122 | 264 |
| Corrected original | Tuktoyaktuk, validation zone 13 | 1,265 | 168 | 243 |
| Alternate | Cambridge Bay, validation zone 4 | 1,778 | 211 | 122 |

Corrected original: 11,076 train / 1,290 validation / 2,111 Cambridge Bay test.
Alternate: 11,589 train / 1,333 validation / 1,676 Tuktoyaktuk test.
All 14,984 patch groups were checked for LiDAR and six S2 raster geometry,
paired file availability, and six attribute records. This is not a full
pixel-value quality audit. The loader additionally rejects dropped patches.

`split_manifest` takes precedence over legacy `validation_regions`. Manifests
record region-qualified IDs, exclusion status, all seven raster bounds,
file sizes and modification timestamps. Startup recomputes the overlap check
from saved bounds and rejects changed input sizes/timestamps or mismatched
source directories. It does not reread all raster headers or hash all pixel
contents at every startup. Reprepare if data change, and use a new run name;
resumption rejects a changed manifest digest.

The large local manifests `splits/{original_clean,tuk_heldout_clean}.json`
are ignored by Git. Commit the YAML configurations, compact `_ids.json`
assignments, preparation/training scripts, utility modules, tests, notebook
and documentation changes. Archive full manifests alongside checkpoints.
To rebuild on another machine, from `Dissertation/`:

```bash
../.venv-py313/bin/python -u scripts/prepare_spatial_splits.py
```

Preparation refuses to overwrite a different existing full manifest. Input
file timestamps are machine-specific; reproduce the compact ID assignments
and retain the local full manifest for that training run.

## Run in tmux

From the workspace directory containing `Dissertation/`:

```bash
tmux new -s roughnet-clean
cd /cs/student/projects2/aisd/2024/tcannon/dissertation/Dissertation
bash scripts/train_clean_fold.sh original_clean && bash scripts/train_clean_fold.sh tuk_heldout_clean
```

Detach with Ctrl-b, then d. Both folds run sequentially on the GPU. Rerun the
same command after interruption; completed folds load their final state and
return. Each fold has a process lock, and `training.log` appends console
output. W&B uploads are disabled by the wrapper. Full training has not been
started during preparation.

The frozen scientific configuration is six randomized context images,
mid attention, depth 4, base channels 64, embedding 256, cosine schedule,
1,000 diffusion timesteps, Adam at 1e-4, batch size 8, hybrid masked MSE
(alpha 1), 200 epochs and seed 42. Both models start from random weights.
Checkpoint selection uses source-region validation loss only. Automatic
reconstruction examples are disabled; full test inference is a separate
post-training operation, using PLMS and the existing metrics.

## Recovery and checks

`models/<fold>/<fold>_latest.pth` stores the latest completed epoch, optimizer,
Python/NumPy/Torch/CUDA RNG state, histories, patience, split digest and best
model. An initial epoch-zero state is saved before training. An interrupted
epoch reruns from its beginning. Checkpoints use temporary files followed by
atomic replacement. `<fold>_best.pth` is the inference checkpoint; do not
resume training from it. Best state is restored from latest on recovery,
including if a crash occurred between best and latest writes.

Resume requires unchanged training/model/data/system configuration. Keep the
same software, GPU and loader settings. The CPU interruption test reproduces
losses and weights exactly; this is not a guarantee of bitwise deterministic
CUDA kernels. Old paper checkpoints lack the recovery state and cannot be
used to resume these fresh runs.

A bounded real-data GPU smoke test passed three optimizer steps at batch 8
on RTX 4070 Ti SUPER, with about 7.9 GiB peak allocated memory. Warm steps
were 0.19–0.22 seconds, excluding data loading; this is not a full epoch timing
estimate. Extrapolating those warm steps gives roughly 16 hours of optimizer
work per 200-epoch fold, before validation, loading and checkpoint overhead.
Budget additional time for test sampling. No full-epoch runtime benchmark
has been run.

## Paper evaluation still required

After training, evaluate the new checkpoints on the saved validation/test
IDs. The legacy evaluation CLI hardcodes the old checkpoint and Tuktoyaktuk
validation subset, so do not invoke it unchanged for the new fold. Its
checkpoint/split selection needs adapting before full test inference.
Likewise, the original conditioning-ablation preparation asserts the old
protocol; prepare a new study for the corrected checkpoint instead of
resuming or overwriting the completed old study. Retain all old artifacts
as historical results. Linear/cosine comparisons require corrected training
for every retained schedule; sampler comparisons reuse each checkpoint.
