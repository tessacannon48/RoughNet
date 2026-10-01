# Training and evaluating baselines with the supplied patch split

Use this guide with `train_val_split_ids.json` and the existing `input_data/`
dataset. The JSON file defines the shared split for training on Pond Inlet
and Tuktoyaktuk and testing on Cambridge Bay. No patch re-extraction is needed
if your dataset contains the same patch files as the supplied dataset.

## Why use this split?

Neighbouring patches overlap spatially. Assigning different patch IDs or
zone labels to training and validation does not necessarily prevent shared
pixels. This split retains the validation patches and excludes training
neighbours whose combined LiDAR and six Sentinel-2 footprints overlap
validation footprints. It prevents this source of training/validation
leakage; it does not make neighbouring evaluation patches statistically
independent.

## Split membership

The JSON has one object per geographic region: `pondinlet`, `tuk`, and
`cambridge`. Each contains four lists of patch IDs:

| Key | Use |
|---|---|
| `train` | Fit model weights and any learned preprocessing. |
| `validation` | Select hyperparameters, perform early stopping and select checkpoints. Do not fit model weights on these patches. |
| `test` | Final evaluation after the model and settings are fixed. Do not use for model selection. |
| `excluded` | Omit from training, validation and test evaluation for this comparison. |

| Region | Training | Validation | Test | Excluded |
|---|---:|---:|---:|---:|
| Pond Inlet | 9,811 | 1,122 | 0 | 264 |
| Tuktoyaktuk | 1,265 | 168 | 0 | 243 |
| Cambridge Bay | 0 | 0 | 2,111 | 0 |
| **Total** | **11,076** | **1,290** | **2,111** | **507** |

Keep IDs as strings, including leading zeros: `"00001"`, not `1`. Use
the region and ID together to identify a sample. Despite its name, the file
also includes the test and excluded lists.

## Loading the split

This example uses only the Python standard library. Set the two paths for
your machine; the JSON can be stored separately from the dataset.

```python
import json
from pathlib import Path

data_root = Path("input_data")
split_path = Path("train_val_split_ids.json")
split_ids = json.loads(split_path.read_text())

expected_counts = {"train": 11076, "validation": 1290, "test": 2111}
records = {name: [] for name in expected_counts}
seen = set()

for region, groups in split_ids.items():
    for split in ("train", "validation", "test", "excluded"):
        for patch_id in groups[split]:
            assert isinstance(patch_id, str), "Preserve IDs as strings"
            key = (region, patch_id)
            assert key not in seen, f"Duplicate split assignment: {key}"
            seen.add(key)
            if split == "excluded":
                continue

            lidar = data_root / f"lidar_patches_{region}" / f"lidar_patch_{patch_id}.tif"
            s2_dir = data_root / f"s2_patches_{region}" / f"s2_patch_{patch_id}"
            s2_views = [s2_dir / f"t{i}.tif" for i in range(6)]
            attrs = s2_dir / "attrs.json"
            for path in [lidar, *s2_views, attrs]:
                if not path.is_file():
                    raise FileNotFoundError(path)

            records[split].append({
                "region": region,
                "patch_id": patch_id,
                "lidar_path": lidar,
                "s2_paths": s2_views,
                "attrs_path": attrs,
            })

for split, expected in expected_counts.items():
    assert len(records[split]) == expected, (split, len(records[split]))

train_records = records["train"]
validation_records = records["validation"]
test_records = records["test"]
```

Pass these separate record lists to your dataset class or training pipeline.
Adapt the directory mapping if your extracted folders have different names;
do not change membership. This example checks membership and file availability,
not the identity of file contents. If you have regenerated, renumbered or
modified patches, confirm dataset compatibility before using this split.

## Training and evaluation rules

1. Build datasets from these explicit lists. Do not randomly split all patches,
   split by `region.json` alone, or treat every non-validation patch as training
   data. In particular, that last approach would reintroduce excluded patches.
2. Train from scratch for the corrected comparison if a previous model used
   excluded patches, validation patches or test patches during fitting. Merely
   changing evaluation IDs does not remove what existing weights learned.
3. Fit dataset-level normalization statistics, imputers, feature transforms
   and other learned preprocessing using `train` only. Apply those fitted
   transforms unchanged to validation and test. Any predefined per-patch
   preprocessing should be consistent across the compared methods.
4. Apply stochastic training augmentation only to training samples. Shuffling
   the training list each epoch is fine; moving samples between splits is not.
5. Use validation for checkpoint and hyperparameter selection. Keep Cambridge
   Bay entirely held out until those choices are fixed. Do not combine training
   and validation for a final refit in this comparison.
6. Verify that your dataset loader retains the expected sample counts. If a
   file is missing or unusable, report it rather than silently dropping samples
   or choosing replacements. Report validation and test results separately,
   retaining region labels for region-specific summaries.

Keep a copy of the split JSON with your experiment outputs so the exact sample
membership remains traceable. Methods with no learned or fitted components
can use the same validation/test cohorts without a training run.
