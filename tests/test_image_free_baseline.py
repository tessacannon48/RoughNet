import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import image_free_baseline as baseline


class BaselineTests(unittest.TestCase):
    def test_metric_parity_ignores_figures_but_detects_scoring_or_import_changes(self):
        source = '''import torch
from example import rmse as score
def _metric_params_from_cfg(cfg):
    return cfg
def _compute_metrics_tensor(gt, pred):
    return score(torch.tensor(gt), pred)
def draw_mosaic():
    return 1
'''
        self.assertEqual(baseline.metric_signature(source), baseline.metric_signature(
            source.replace('return 1', 'return 2')))
        for changed in (source.replace('return score(', 'return -score('),
                        source.replace('from example', 'from another')):
            self.assertNotEqual(baseline.metric_signature(source), baseline.metric_signature(changed))

    def fixture(self, root):
        rows = []
        directories = {r: dict(lidar=f'input_data/{r}') for r in ('pondinlet', 'tuk', 'cambridge')}
        for i, (region, role, full) in enumerate([
                ('pondinlet', 'train', True), ('tuk', 'train', False), ('tuk', 'train', True),
                ('pondinlet', 'excluded', True), ('pondinlet', 'validation', True),
                ('tuk', 'validation', True), ('cambridge', 'test', False), ('cambridge', 'test', True)]):
            pid = f'{i:05d}'
            path = root / directories[region]['lidar'] / f'lidar_patch_{pid}.tif'
            path.parent.mkdir(parents=True, exist_ok=True)
            data = np.random.default_rng(i).normal(0.2, 0.05, (256, 256)).astype('float32')
            mask = np.ones_like(data)
            if not full:
                mask[:20] = 0
                data[:20] = 10  # Must not affect the target's valid-pixel mean.
            with rasterio.open(path, 'w', driver='GTiff', width=256, height=256,
                               count=2, dtype='float32', transform=from_origin(500000, 8000000, 1, 1),
                               crs='EPSG:32617') as dst:
                dst.write(np.stack([data, mask]))
            st = path.stat()
            rows.append(dict(region=region, id=pid, split=role, crs='EPSG:32617',
                             bounds=[i*300, 0, i*300+256, 256],
                             files=[dict(path=str(path.relative_to(root)), size=st.st_size, mtime_ns=st.st_mtime_ns)]))
        split = dict(test_region='cambridge', validation_zones={'pondinlet': 4, 'tuk': 13},
                     directories=directories, patches=rows)
        manifest = root / 'split.json'
        manifest.write_text(json.dumps(split))
        reference = root / 'reference'
        baseline.write_json(reference / 'manifest.json', dict(split_digest=baseline.digest(split),
                            max_patches=None, sources={}, configurations={'cosine': {}},
                            cohorts={'cambridge': dict(split='test', records=[r for r in rows if r['split']=='test'])}))
        baseline.write_csv(reference / 'test/per_patch.csv',
                           [dict(region=r['region'], tile_id=r['id'], schedule='cosine', sampler='plms',
                                 **{k: 1 for k in baseline.METRICS}) for r in rows if r['split']=='test'])
        return SimpleNamespace(out=root/'study', manifest=manifest, reference_study=reference,
                               split='test', seed=42, workers=1, batch_size=1, max_batches=1)

    def test_draws_are_reproducible_and_allow_replacement(self):
        targets = [dict(region='cambridge', tile_id=str(i), path=str(i)) for i in range(20)]
        donors = [dict(region='pondinlet', tile_id='a', path='a'), dict(region='tuk', tile_id='b', path='b')]
        a = baseline.assignments(targets, donors, 42)
        self.assertEqual(a, baseline.assignments(targets, donors, 42))
        self.assertNotEqual(a, baseline.assignments(targets, donors, 43))
        self.assertEqual(len(a), 20)
        self.assertEqual({x['donor_tile_id'] for x in a}, {'a', 'b'})
        with self.assertRaises(ValueError):
            baseline.assignments(targets, [], 42)

    def test_prepare_resume_scoring_and_corruption_guard(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.fixture(root)
            with patch.object(baseline, 'ROOT', root), patch.object(baseline, 'sources', return_value={'test': 'fixed'}):
                baseline.prepare(args)
                manifest = baseline.load_study(args.out)
                self.assertEqual(manifest['donor_counts'], {
                    'pondinlet': dict(training=1, eligible=1), 'tuk': dict(training=2, eligible=1)})
                self.assertTrue(all(p['donor_tile_id'] in ('00000', '00002') for p in manifest['assignments']))
                with self.assertRaises(FileExistsError):
                    baseline.prepare(args)
                baseline.run(args)
                first = args.out / 'shards/000000.json'
                unchanged = first.read_bytes()
                with self.assertRaisesRegex(ValueError, 'Incomplete study'):
                    baseline.summarize(args)
                # The first target has missing pixels. Retain its whole valid mask;
                # score the unchanged donor field with the same metric adapter.
                pair = manifest['assignments'][0]
                gt, mask, _ = baseline.read_lidar(root / pair['target_path'])
                pred, _, _ = baseline.read_lidar(root / pair['donor_path'])
                expected = baseline.patch_metrics(gt, pred, mask, {})
                row = json.loads(unchanged)['rows'][0]
                self.assertEqual(row['valid_pixels'], 236*256)
                for k in baseline.METRICS:
                    self.assertAlmostEqual(row[k], expected[k], places=6)
                with patch.object(baseline, 'patch_metrics', wraps=baseline.patch_metrics) as scorer:
                    baseline.run(args)
                    self.assertEqual(scorer.call_count, 1)
                self.assertEqual(first.read_bytes(), unchanged)
                baseline.summarize(args)
                with (args.out / 'test/comparison.csv').open() as f:
                    summary = list(csv.DictReader(f))
                self.assertEqual(len(summary), 2)
                self.assertEqual({r['n'] for r in summary}, {'2'})
                corrupted = json.loads(unchanged)
                corrupted['rows'][0]['rmse_m'] += 1
                first.write_text(json.dumps(corrupted))
                with self.assertRaisesRegex(ValueError, 'scores changed'):
                    baseline.run(args)

    def test_input_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.fixture(root)
            with patch.object(baseline, 'ROOT', root), patch.object(baseline, 'sources', return_value={}):
                baseline.prepare(args)
                manifest = baseline.load_study(args.out)
                (root / manifest['targets'][0]['path']).write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'Input changed'):
                    baseline.check_inputs(manifest)


if __name__ == '__main__':
    unittest.main()
