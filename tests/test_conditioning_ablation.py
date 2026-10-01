"""Scientific invariants of the paired experiment; no trained model/GPU required."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from src.diffusion.sampling import p_sample_loop_plms
from src.diffusion.scheduler import CosineDiffusionScheduler
from src.utils.conditioning_ablation import (
    MODES, METRICS, donor_permutation, disjoint, paired_noise, condition_inputs,
    patch_metrics, block_interval, json_digest, write_json, save_predictions, sha256_file,
)

spec = importlib.util.spec_from_file_location("ablation_runner", ROOT / "scripts/conditioning_ablation.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_donors_are_reproducible_one_to_one_and_nonoverlapping(self):
        records = [{"footprint": [i*128, 0, i*128+260, 260]} for i in range(30)]
        p = donor_permutation(records, 123)
        self.assertEqual(p, donor_permutation(records, 123))
        self.assertEqual(sorted(p), list(range(30)))
        self.assertNotEqual(p, donor_permutation(records, 124))
        for i, j in enumerate(p):
            self.assertNotEqual(i, j)
            self.assertTrue(disjoint(records[i]["footprint"], records[j]["footprint"]))
            self.assertGreaterEqual(abs(i-j)*128, 512)

    def test_impossible_donors_fail(self):
        with self.assertRaises(ValueError):
            donor_permutation([{"footprint": [0, 0, 256, 256]}]*2, 42)

    def test_noise_is_independent_of_batching_and_shuffle_rng(self):
        a = paired_noise("tuk", ["a", "b"], 42, (1, 8, 8))
        torch.randperm(100)
        b = torch.cat([paired_noise("tuk", [p], 42, (1, 8, 8)) for p in ["a", "b"]])
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, paired_noise("tuk", ["a", "b"], 43, (1, 8, 8))))

    def test_whole_condition_is_swapped_and_null_is_zero(self):
        target = {"s2": torch.ones(2, 24, 4, 4), "attrs": torch.ones(2, 48)}
        donor = {k: v*2 for k, v in target.items()}
        for actual, key in zip(condition_inputs("shuffle", target, donor), ("s2", "attrs")):
            self.assertTrue(torch.equal(actual, donor[key]))
        self.assertTrue(all(torch.count_nonzero(v) == 0 for v in condition_inputs("null", target, donor)))
        self.assertTrue(torch.equal(target["s2"], torch.ones_like(target["s2"])))

    def test_supplied_noise_preserves_historical_plms(self):
        shape = (2, 1, 8, 8)
        scheduler = CosineDiffusionScheduler(timesteps=6, device="cpu")
        cond, attrs = torch.ones(shape), torch.zeros(2, 8)
        model = lambda x, c, a, t: x*0.1+c*0.01
        torch.manual_seed(42)
        old = p_sample_loop_plms(model, scheduler, shape, cond, attrs, "cpu")
        torch.manual_seed(42)
        noise = torch.randn(shape)
        original = noise.clone()
        new = p_sample_loop_plms(model, scheduler, shape, cond, attrs, "cpu", initial_noise=noise)
        self.assertTrue(torch.equal(old, new))
        self.assertTrue(torch.equal(noise, original))
        with self.assertRaises(ValueError):
            p_sample_loop_plms(model, scheduler, shape, cond, attrs, "cpu", initial_noise=noise[:1])

    def test_metrics_match_existing_paper_evaluator(self):
        from scripts.evaluation import _compute_metrics_tensor, _metric_params_from_cfg
        torch.manual_seed(1)
        gt = torch.randn(64, 64)*0.1
        pred = gt*0.8+0.03
        mask = torch.ones_like(gt, dtype=torch.bool)
        mask[10:20, 10:20] = False
        cfg = {"evaluation": {"jsd_scales_m": [1, 2], "jsd_bins": 32}}
        actual = patch_metrics(gt, pred, mask, cfg)
        historical = _compute_metrics_tensor(torch.where(mask, gt, torch.nan), pred, mask,
                          *_metric_params_from_cfg(cfg), use_absrel_epsilon=False)
        keys = ("rmse_phys_m", "bias_phys_m", "sigma_error_pct", "normal_angle_error_deg",
                "zncc", "jsd", "psd_rmse", "nrmse_std")
        for name, old_name in zip(METRICS, keys):
            self.assertAlmostEqual(actual[name], historical[old_name], places=7, msg=name)
        shifted = patch_metrics(gt, gt+0.05, mask, cfg)
        self.assertAlmostEqual(shifted["bias_m"], 0.05, places=5)
        self.assertAlmostEqual(shifted["rmse_m"], 0.05, places=5)
        pred[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            patch_metrics(gt, pred, mask, cfg)

    def test_prediction_archive_is_lossless(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prediction.npz"
            gt = torch.randn(8, 8)
            predictions = {m: torch.randn(8, 8) for m in MODES}
            save_predictions(path, gt, gt > 0, predictions, 0.25)
            with np.load(path) as saved:
                np.testing.assert_array_equal(saved["gt"], gt.numpy())
                for m in MODES:
                    self.assertEqual(saved[m].dtype, np.float32)
                    np.testing.assert_array_equal(saved[m], predictions[m].numpy())
            self.assertFalse(path.with_suffix(".tmp").exists())

    def test_cluster_bootstrap_and_small_sample_handling(self):
        centers = [(x*1024, 0) for x in range(10) for _ in range(2)]
        a = block_interval([0.2]*20, centers, 1024, 42, 100)
        self.assertEqual(a["n_blocks"], 10)
        self.assertAlmostEqual(a["ci_low"], 0.2)
        self.assertAlmostEqual(a["ci_high"], 0.2)
        b = block_interval([1, 2], [(0, 0), (1, 0)], 1024, 42, 100)
        self.assertIsNone(b["ci_low"])

    def test_complete_summary_and_incomplete_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            records = [{"tile_id": str(i), "center": [i*1024, 0]} for i in range(6)]
            manifest = {"seeds": [42, 43], "subset_max_patches": None,
                "regions": {"tuk": {"records": records, "donor_indices": {str(s): [1, 2, 3, 4, 5, 0] for s in [42, 43]}}}}
            runtime = {"manifest_sha256": json_digest(manifest), "batch_size": 8}
            write_json(out / "manifest.json", manifest)
            write_json(out / "runtime.json", runtime)
            for seed in manifest["seeds"]:
                rows = []
                for i in range(6):
                    for m in MODES:
                        rows.append({"region": "tuk", "seed": seed, "tile_id": str(i), "mode": m,
                            "donor_tile_id": str((i+1)%6) if m == "shuffle" else None,
                            **{k: 1 if m == "normal" else 2 for k in METRICS}})
                archives = []
                for i in range(6):
                    path = Path("predictions") / "tuk" / f"seed{seed}" / f"{i}.npz"
                    save_predictions(out / path, torch.ones(8, 8), torch.ones(8, 8).bool(),
                                     {m: torch.ones(8, 8) for m in MODES}, 0.0)
                    archives.append({"path": str(path), "sha256": sha256_file(out / path)})
                write_json(out / "shards" / f"tuk_seed{seed}_000000.json", {"runtime_sha256": json_digest(runtime), "rows": rows, "archives": archives})
            args = types.SimpleNamespace(out=out, block_sizes=None, bootstrap=100, plot=False)
            runner.summarize(args)
            result = json.loads((out / "summary.json").read_text())
            self.assertTrue(all(r["mean"] == 1 for r in result["paired_deltas"]))
            self.assertTrue((out / "table.tex").exists())
            self.assertTrue(all(r["ci_low"] is None for r in result["paired_deltas"]))
            args.block_sizes = [1024]
            runner.summarize(args)
            result = json.loads((out / "summary.json").read_text())
            self.assertTrue(all(r["ci_low"] == 1 for r in result["paired_deltas"]))
            (out / "shards/tuk_seed43_000000.json").unlink()
            with self.assertRaisesRegex(ValueError, "incomplete"):
                runner.summarize(args)


if __name__ == "__main__":
    unittest.main()
