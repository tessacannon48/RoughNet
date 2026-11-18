"""Deterministic baseline inference, cohort provenance and training recovery."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils import unet_baseline as baseline
from src.utils.conditioning_ablation import METRICS, patch_metrics
from src.utils.spatial_split import digest, sha256_file
from src.utils.training_state import atomic_save
from scripts import evaluate_unet_baseline as evaluation


def small_config():
    return dict(
        data=dict(pixel_size_m=1.0),
        training=dict(context_k=1, randomize_context=True, batch_size=2, num_workers=0),
        model=dict(base_channels=8, embed_dim=16, unet_depth=1, attention_variant='none'),
        evaluation=dict(jsd_scales_m=[1.0, 2.0, 5.0, 10.0], jsd_bins=256,
                        nae_use_sobel=True, nae_degrees=True, psd_window=True),
        baseline=dict(name='unet_regression'),
    )


def tiny_split():
    ids = {r: {s: [] for s in baseline.SPLITS} for r in baseline.REGIONS}
    for region, role, pid in [('pondinlet', 'train', '00001'),
                               ('tuk', 'validation', '00002'),
                               ('cambridge', 'test', '00003'),
                               ('pondinlet', 'excluded', '00004')]:
        ids[region][role].append(pid)
    return ids


class ConditionModel(torch.nn.Module):
    def forward(self, lidar, s2, attrs, timestep):
        if torch.count_nonzero(lidar) or torch.count_nonzero(timestep):
            raise AssertionError('Deterministic baseline received diffusion inputs')
        return s2[:, :1]


class MetricDataset(torch.utils.data.Dataset):
    def __init__(self, ids=('00001', '00002')):
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        y, x = torch.meshgrid(torch.arange(32), torch.arange(32), indexing='ij')
        gt = torch.sin(x.float() / 5) * .1 + torch.cos(y.float() / 7) * .2
        mask = torch.ones_like(gt)
        if index:
            mask[:16] = 0
        prediction = gt * .8 + (.0134567 if index == 0 else .1234567)
        # Invalid target values must not leak into the paper's spatial metrics.
        target = torch.where(mask.bool(), gt, torch.tensor(1000.0))
        return dict(tile_id=self.ids[index], lidar=target[None], mask=mask,
                    s2=prediction[None].repeat(4, 1, 1), attrs=torch.zeros(8))


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_forward_matches_supplied_zero_input_fixed_timestep_baseline(self):
        torch.manual_seed(42)
        model = baseline.build_model(small_config(), 'cpu').eval()
        s2, attrs = torch.rand(2, 4, 8, 8), torch.rand(2, 8)
        observed = []
        hook = model.register_forward_pre_hook(lambda _model, inputs: observed.append(inputs))
        with torch.no_grad():
            actual = baseline.predict(model, s2, attrs)
            # This is the supplied script's complete inference operation.
            expected = model(torch.zeros(2, 1, 8, 8), s2, attrs,
                             torch.zeros(2, dtype=torch.long))
        hook.remove()
        self.assertEqual(observed[0][0].shape, (2, 1, 8, 8))
        self.assertEqual(observed[0][3].dtype, torch.long)
        self.assertEqual(int(torch.count_nonzero(observed[0][0])), 0)
        self.assertEqual(int(torch.count_nonzero(observed[0][3])), 0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_checkpoint_uses_saved_model_and_rejects_diffusion_weights(self):
        cfg = small_config()
        model = baseline.build_model(cfg, 'cpu')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'colleague_best.pth'
            saved = dict(epoch=7, config=cfg, model_state_dict=model.state_dict())
            torch.save(saved, path)
            loaded, actual_cfg, checkpoint = baseline.load_checkpoint(path)
            self.assertEqual(actual_cfg, cfg)
            self.assertEqual(loaded.base_channels, 8)
            self.assertEqual(checkpoint['epoch'], 7)
            for key, tensor in model.state_dict().items():
                torch.testing.assert_close(loaded.state_dict()[key], tensor, rtol=0, atol=0)
            saved['config'] = copy.deepcopy(cfg)
            del saved['config']['baseline']
            torch.save(saved, path)
            with self.assertRaisesRegex(ValueError, 'deterministic'):
                baseline.load_checkpoint(path)

    def test_primary_split_rejects_reassigned_duplicate_or_numeric_ids(self):
        primary = baseline.read_split_ids(ROOT / 'splits/train_val_split_ids.json')
        self.assertEqual(sum(len(primary[r]['train']) for r in baseline.REGIONS), 11076)
        self.assertEqual(sum(len(primary[r]['validation']) for r in baseline.REGIONS), 1290)
        self.assertEqual(len(primary['cambridge']['test']), 2111)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ids.json'
            for kind in ('reassigned', 'duplicate', 'numeric'):
                changed = copy.deepcopy(primary)
                if kind == 'reassigned':
                    changed['pondinlet']['train'][0] = 'unexpected'
                elif kind == 'duplicate':
                    changed['pondinlet']['train'].append(primary['cambridge']['test'][0])
                else:
                    changed['pondinlet']['train'][0] = int(changed['pondinlet']['train'][0])
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    baseline.read_split_ids(path)

    def test_dataset_rejects_dropped_or_cross_region_samples(self):
        cfg = small_config(); cfg['data']['data_root'] = '/unused'
        def sample(pid, region='pondinlet'):
            return dict(tile_id=pid, lidar_path=f'/unused/lidar_patches_{region}/lidar_patch_{pid}.tif',
                        s2_group_dir=f'/unused/s2_patches_{region}/s2_patch_{pid}')
        for rows in ([sample('00001')], [sample('00001'), sample('00002', 'tuk')]):
            fake = SimpleNamespace(samples=rows)
            with patch.object(baseline, 'LidarS2Dataset', return_value=fake):
                with self.assertRaises(ValueError):
                    baseline.make_dataset(cfg, ['00001', '00002'], 'validation', region='pondinlet')

    def test_legacy_split_proof_requires_checksum_and_exact_membership(self):
        ids = tiny_split()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); split = root / 'ids.json'; old = root / 'original.csv'
            split.write_text(json.dumps(ids))
            rows = [dict(region=r, tile_id=pid, split='val' if role == 'validation' else role)
                    for r, parts in ids.items() for role, pids in parts.items() for pid in pids]
            def save():
                with old.open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=['region', 'tile_id', 'split'])
                    writer.writeheader(); writer.writerows(rows)
            save()
            cfg = dict(baseline=dict(splits_md5=hashlib.md5(old.read_bytes()).hexdigest()))
            with self.assertRaisesRegex(ValueError, 'legacy-splits'):
                evaluation.validate_training_identity(cfg, ids, split)
            proof = evaluation.validate_training_identity(cfg, ids, split, old)
            self.assertIn('canonical_memberships', proof['method'])
            self.assertIn('not recorded', proof['training_input_content_identity'])
            # The supplied trainer also accepts CSVs listing only train/val.
            complete = list(rows)
            rows[:] = [r for r in rows if r['split'] in ('train', 'val')]
            save(); cfg['baseline']['splits_md5'] = hashlib.md5(old.read_bytes()).hexdigest()
            evaluation.validate_training_identity(cfg, ids, split, old)
            rows[:] = complete
            save(); cfg['baseline']['splits_md5'] = hashlib.md5(old.read_bytes()).hexdigest()
            rows[0]['split'] = 'test'; save()
            with self.assertRaisesRegex(ValueError, 'splits_md5'):
                evaluation.validate_training_identity(cfg, ids, split, old)
            cfg['baseline']['splits_md5'] = hashlib.md5(old.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, 'membership'):
                evaluation.validate_training_identity(cfg, ids, split, old)

    def test_bound_checkpoint_rejects_changed_input_bytes_or_inventory(self):
        ids = tiny_split()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); data = root / 'input_data'; data.mkdir(); rows = []
            for region, parts in ids.items():
                for role, pids in parts.items():
                    for index, pid in enumerate(pids):
                        p = data / (pid + '.bin'); p.write_bytes(pid.encode())
                        rows.append(dict(id=pid, region=region, split=role, crs='EPSG:32617',
                            bounds=[int(pid)*3, 0, int(pid)*3+1, 1],
                            files=[dict(path='input_data/'+p.name, size=p.stat().st_size, sha256=sha256_file(p))]))
            manifest = dict(version=2, patches=rows, validation_zones={'pondinlet':4, 'tuk':13}, test_region='cambridge')
            path = root / 'manifest.json'; path.write_text(json.dumps(manifest))
            cfg = dict(data=dict(data_root=str(data), split_manifest=str(path), split_digest=digest(manifest)),
                       baseline=dict(training_split_ids_sha256='recorded'))
            proof = evaluation.verify_evaluation_inputs(cfg, ids, cfg)
            self.assertEqual(proof['training_content_binding'], 'verified')
            first = data / '00001.bin'; first.write_bytes(b'XXXXX')
            with self.assertRaisesRegex(ValueError, 'Input changed'):
                evaluation.verify_evaluation_inputs(cfg, ids, cfg)
            first.write_bytes(b'00001')
            recorded = copy.deepcopy(cfg); recorded['data']['split_digest'] = 'other'
            with self.assertRaisesRegex(ValueError, 'training identity'):
                evaluation.verify_evaluation_inputs(cfg, ids, recorded)

    def test_scores_use_canonical_unrounded_metrics_and_equal_patch_weights(self):
        cfg = small_config(); dataset = MetricDataset()
        rows = evaluation.score_dataset(ConditionModel(), dataset, cfg, torch.device('cpu'),
                       region='cambridge', split='test', batch_size=2, num_workers=0)
        for index, row in enumerate(rows):
            sample = dataset[index]
            expected = patch_metrics(sample['lidar'], sample['s2'][:1], sample['mask'].bool(), cfg)
            self.assertEqual(set(row), {'region', 'tile_id', 'split', *METRICS})
            self.assertEqual({k: row[k] for k in METRICS}, expected)
            self.assertEqual(row['tile_id'], dataset.ids[index])
        summary = evaluation.summarize(rows)[0]
        expected = (rows[0]['rmse_m'] + rows[1]['rmse_m']) / 2
        self.assertEqual(summary['rmse_m_mean'], expected)
        weighted = (rows[0]['rmse_m'] * 1024 + rows[1]['rmse_m'] * 512) / 1536
        self.assertNotAlmostEqual(summary['rmse_m_mean'], weighted, places=5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'per_patch.csv'; evaluation.write_csv(path, rows)
            with path.open(newline='') as stream:
                saved = list(csv.DictReader(stream))
            self.assertEqual(float(saved[0]['rmse_m']), rows[0]['rmse_m'])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            evaluation.score_dataset(ConditionModel(), MetricDataset(('00001', '00001')), cfg,
                torch.device('cpu'), region='cambridge', split='test', batch_size=2, num_workers=0)

    def test_evaluation_rejects_wrong_cohort_before_writing_results(self):
        cfg = small_config(); ids = tiny_split()
        ids['cambridge']['test'] = ['00001', '00002']
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); checkpoint = root / 'checkpoint'; checkpoint.write_bytes(b'fixture')
            split = root / 'split.json'; split.write_text(json.dumps(ids))
            args = SimpleNamespace(checkpoint=checkpoint, split_ids=split, out=root/'output',
                split='test', device='cpu', legacy_splits=None, data_root=None, manifest=None,
                batch_size=2, num_workers=0)
            with patch.object(evaluation, 'load_checkpoint', return_value=(ConditionModel(), cfg, {'epoch':7})), \
                 patch.object(evaluation, 'read_split_ids', return_value=ids), \
                 patch.object(evaluation, 'validate_training_identity', return_value={'method':'fixture'}), \
                 patch.object(evaluation, 'prepare_config', return_value=cfg), \
                 patch.object(evaluation, 'verify_evaluation_inputs', return_value={'status':'fixture'}), \
                 patch.object(evaluation, 'make_dataset', return_value=MetricDataset(('00001', 'unexpected'))) as make:
                with self.assertRaisesRegex(ValueError, 'target IDs'):
                    evaluation.evaluate(args)
                self.assertEqual(make.call_args.args[1], ids['cambridge']['test'])
                self.assertEqual(make.call_args.kwargs['region'], 'cambridge')
                self.assertFalse(args.out.exists())
            args.out.mkdir()
            with patch.object(evaluation, 'load_checkpoint') as load:
                with self.assertRaises(FileExistsError):
                    evaluation.evaluate(args)
                load.assert_not_called()

    def test_zero_worker_loaders_do_not_enable_persistent_workers(self):
        from scripts import train_unet_baseline as training
        cfg = small_config(); cfg['system'] = {'device':'cpu'}
        for validation in (False, True):
            loader = training.make_loader(MetricDataset(), cfg, validation=validation)
            self.assertEqual(loader.num_workers, 0)
            self.assertFalse(loader.persistent_workers)
            self.assertEqual(sum(len(b['tile_id']) for b in loader), 2)

    def test_interrupted_epoch_restores_model_optimizer_and_all_rng_states(self):
        from scripts import train_unet_baseline as training
        class Dataset(torch.utils.data.Dataset):
            def __init__(self, ids): self.ids = ids
            def __len__(self): return len(self.ids)
            def __getitem__(self, index):
                jitter = (random.random() + np.random.random() + float(torch.rand(()))) / 100
                x = torch.full((1, 4, 4), (index+1)/10 + jitter)
                return dict(tile_id=self.ids[index], lidar=x, s2=x.repeat(4,1,1),
                            mask=torch.ones(4,4), attrs=torch.zeros(8))
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight = torch.nn.Parameter(torch.randn(()))
            def forward(self, zeros, s2, attrs, timestep):
                return self.weight * torch.nn.functional.dropout(s2[:, :1], p=.2, training=self.training)
        ids = tiny_split()
        ids['pondinlet']['train'] = ['p0', 'p1']; ids['tuk']['train'] = ['t0', 't1']
        ids['pondinlet']['validation'] = ['v0']; ids['tuk']['validation'] = ['v1']
        cfg = small_config()
        cfg['training'].update(epochs=2, lr=.01, loss=dict(name='masked_mse_loss', alpha=1.0))
        cfg['data'].update(split_ids='fixture.json', split_ids_sha256='fixture', split_digest='fixture')
        cfg['system'] = dict(device='cpu', debug=False)
        cfg['baseline'].update(seed=42, wandb=False)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(training, 'prepare_config', side_effect=lambda c, **kw: copy.deepcopy(c)), \
             patch.object(training, 'read_split_ids', return_value=ids), \
             patch.object(training, 'make_dataset', side_effect=lambda c, pids, *a, **kw: Dataset(pids)), \
             patch.object(training, 'build_model', side_effect=lambda c, d: Model().to(d)):
            cfg['logging'] = dict(save_dir=str(Path(directory)/'full'), run_name='fixture')
            full_paths = training.train_model(copy.deepcopy(cfg))
            full = torch.load(full_paths['latest'], weights_only=False)
            full_next = random.random(), np.random.random(), torch.rand(3)
            cfg['logging']['save_dir'] = str(Path(directory)/'resumed')
            def interrupt(state, path):
                atomic_save(state, path)
                if str(path).endswith('_latest.pth') and state['epoch'] == 1:
                    raise InterruptedError('test interruption after completed epoch')
            with patch.object(training, 'atomic_save', side_effect=interrupt), self.assertRaises(InterruptedError):
                training.train_model(copy.deepcopy(cfg))
            training.set_seed(999)
            resumed_paths = training.train_model(copy.deepcopy(cfg), resume='auto')
            actual = torch.load(resumed_paths['latest'], weights_only=False)
            actual_next = random.random(), np.random.random(), torch.rand(3)
            self.assertEqual(actual['train_losses'], full['train_losses'])
            self.assertEqual(actual['val_losses'], full['val_losses'])
            self.assertEqual(actual['best_val_loss'], full['best_val_loss'])
            torch.testing.assert_close(actual['model_state_dict']['weight'], full['model_state_dict']['weight'], rtol=0, atol=0)
            self.assertEqual(actual['optimizer_state_dict']['param_groups'], full['optimizer_state_dict']['param_groups'])
            for key, value in full['optimizer_state_dict']['state'][0].items():
                torch.testing.assert_close(actual['optimizer_state_dict']['state'][0][key], value, rtol=0, atol=0)
            self.assertEqual(actual_next[:2], full_next[:2])
            torch.testing.assert_close(actual_next[2], full_next[2], rtol=0, atol=0)
            with (Path(cfg['logging']['save_dir'])/'log.csv').open(newline='') as stream:
                self.assertEqual([int(r['epoch']) for r in csv.DictReader(stream)], [1, 2])


if __name__ == '__main__':
    unittest.main()
