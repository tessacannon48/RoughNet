"""CPU checks of cached figure provenance and spatial/residual semantics."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import evaluation


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class CachedMosaicTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.study = self.root / "study"
        self.study.mkdir()
        self.checkpoint = self.root / "original_clean_best.pth"
        self.checkpoint.write_bytes(b"frozen selected checkpoint")

    def fixture(self, overlapping=False, shifted=False):
        """Two georeferenced tiles with unsupported pixels and retained biases."""
        lidar_dir = self.root / "input_data" / "lidar_patches_tuk"
        lidar_dir.mkdir(parents=True)
        gt_a = np.array([[-1, 1], [0, 0]], dtype=np.float32)
        mask_a = np.array([[True, True], [False, True]])
        gt_b = np.array([[-2, 2], [0, 0]], dtype=np.float32)
        pred_a = gt_a + np.float32(0.25)
        pred_b = gt_b - np.float32(0.5)
        # This prediction must contribute neither to the mosaic nor its weights.
        pred_a[~mask_a] = 1000000
        records, archives, rows = [], [], []
        arrays = [(gt_a, pred_a, mask_a), (gt_b, pred_b, np.ones((2, 2), dtype=bool))]
        for index, (gt, pred, mask) in enumerate(arrays):
            pid = f"{index + 1:05d}"
            x = 0.0 if overlapping or index == 0 else 2.0
            if shifted and index == 1:
                x += 0.25
            lidar = lidar_dir / f"lidar_patch_{pid}.tif"
            with rasterio.open(
                lidar, "w", driver="GTiff", width=2, height=2, count=2,
                dtype="float32", transform=from_origin(x, 2, 1, 1),
                crs="EPSG:32613", nodata=-9999.0,
            ) as dst:
                dst.write(gt, 1)
                dst.write(mask.astype(np.float32), 2)
            stat = lidar.stat()
            records.append({
                "id": pid, "region": "tuk", "split": "validation", "zone": 13,
                "bounds": [x, 0, x + 2, 2], "crs": "EPSG:32613",
                "files": [{"path": str(lidar.relative_to(self.root)),
                           "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}],
            })
            archive_rel = f"predictions/cosine/plms/tuk/{pid}.npz"
            archive = self.study / archive_rel
            archive.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(archive, gt=gt, prediction=pred, mask=mask)
            archives.append({"path": archive_rel, "sha256": file_hash(archive)})
            rows.append({
                "tile_id": pid, "region": "tuk", "split": "validation",
                "schedule": "cosine", "sampler": "plms",
                **{key: 0.0 for key in ("rmse_m", "bias_m", "sigma_error_pct", "nae_deg",
                                        "zncc", "jsd", "log_psd_rmse", "nrmse_sigma")},
            })
        self.manifest = {
            "batch_size": 2, "seed": 42,
            "checkpoints": {"cosine": {"path": str(self.checkpoint),
                                        "sha256": file_hash(self.checkpoint), "epoch": 166}},
            "cohorts": {"tuk": {"records": records, "split": "validation",
                                    "directories": {"lidar": str(lidar_dir.relative_to(self.root))}}},
        }
        self.shard = {
            "manifest_digest": manifest_hash(self.manifest),
            "ids": [row["id"] for row in records], "archives": archives, "rows": rows,
        }
        (self.study / "manifest.json").write_text(json.dumps(self.manifest))
        shard_path = self.study / "shards" / "cosine_plms_tuk_000000.json"
        shard_path.parent.mkdir()
        shard_path.write_text(json.dumps(self.shard))
        return arrays, shard_path

    def build(self):
        return evaluation.build_cached_demeaned_mosaics(
            self.study, "tuk", self.checkpoint, root=str(self.root),
        )

    def test_adjacent_tiles_preserve_geography_mask_and_prediction_bias(self):
        arrays, _ = self.fixture()
        gt, pred, difference, provenance = self.build()
        expected_gt = np.concatenate([item[0] for item in arrays], axis=1)
        expected_pred = np.concatenate([item[1] for item in arrays], axis=1)
        expected_gt[1, 0] = expected_pred[1, 0] = np.nan
        np.testing.assert_allclose(gt, expected_gt, equal_nan=True)
        np.testing.assert_allclose(pred, expected_pred, equal_nan=True)
        np.testing.assert_allclose(difference, expected_pred - expected_gt, equal_nan=True)
        np.testing.assert_allclose(difference[:, :2][arrays[0][2]], 0.25)
        np.testing.assert_allclose(difference[:, 2:], -0.5)
        self.assertIsInstance(provenance, dict)

    def test_overlap_uses_same_valid_support_for_both_fields(self):
        arrays, _ = self.fixture(overlapping=True)
        gt, pred, difference, _ = self.build()
        # A's invalid pixel must use only B; even a huge invalid prediction is ignored.
        self.assertAlmostEqual(float(gt[1, 0]), float(arrays[1][0][1, 0]))
        self.assertAlmostEqual(float(pred[1, 0]), float(arrays[1][1][1, 0]))
        self.assertAlmostEqual(float(difference[1, 0]), -0.5)
        # Same support/weights ensures a spatially constant per-tile prediction bias
        # remains between the two contributing biases wherever both tiles are valid.
        both = arrays[0][2]
        self.assertTrue(np.all(difference[both] >= -0.500001))
        self.assertTrue(np.all(difference[both] <= 0.250001))
        np.testing.assert_allclose(difference, pred - gt, atol=1e-6)

    def test_wrong_checkpoint_is_rejected(self):
        self.fixture()
        self.checkpoint.write_bytes(b"different fitted weights")
        with self.assertRaises(ValueError):
            self.build()

    def test_corrupted_archive_is_rejected(self):
        self.fixture()
        (self.study / self.shard["archives"][0]["path"]).write_bytes(b"interrupted write")
        with self.assertRaises(ValueError):
            self.build()

    def test_mismatched_shard_provenance_is_rejected(self):
        _, path = self.fixture()
        self.shard["manifest_digest"] = "unrelated study"
        path.write_text(json.dumps(self.shard))
        with self.assertRaises(ValueError):
            self.build()

    def test_subpixel_shift_cannot_silently_snap_to_mosaic_grid(self):
        self.fixture(shifted=True)
        with self.assertRaises(ValueError):
            self.build()


if __name__ == "__main__":
    unittest.main()
