import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import yaml
from scipy.spatial.distance import pdist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import figure5_analysis as runner
from scripts import fig5_statistics as metrics


class Figure5Tests(unittest.TestCase):
    def test_roughness_confusion_boundaries_counts_and_empty_classes(self):
        from scripts.plot_figure5 import roughness_confusion
        rows=[dict(sigma_gt=.04,sigma_pred=.05),dict(sigma_gt=.05,sigma_pred=.09),
              dict(sigma_gt=.20,sigma_pred=.21)]
        result=roughness_confusion(rows,[.05,.10,.20])
        self.assertEqual(result['counts'],[[0,1,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,1]])
        self.assertEqual(result['row_fractions'][2],[0,0,0,0])
        self.assertAlmostEqual(result['accuracy'],2/3)
        with self.assertRaises(ValueError):roughness_confusion(rows,[.1,.05])

    def options(self):
        cfg = yaml.safe_load((runner.ROOT/'configs/figure5.yaml').read_text())
        cfg['analysis'].update(min_valid=32, variogram_samples=64, fit_samples=64)
        return cfg

    def test_variogram_matches_notebook_pair_bins_and_is_reproducible(self):
        options = self.options()['analysis']
        image = np.random.default_rng(4).normal(size=(16,16)); mask = np.ones_like(image,bool)
        values, counts, _ = metrics.compute_variogram(image,mask,options)
        y,x = np.where(mask); idx = np.random.RandomState(42).choice(len(x),64,replace=False)
        distances = pdist(np.column_stack((x[idx],y[idx])))
        squared = .5*pdist(image[mask][idx,None],metric='sqeuclidean')
        edges = np.arange(0,204,4); bins = np.digitize(distances,edges)
        expected = np.array([squared[bins==i].mean() if np.any(bins==i) else np.nan for i in range(1,len(edges))])
        np.testing.assert_allclose(values,expected,equal_nan=True)
        np.testing.assert_array_equal(values,metrics.compute_variogram(image,mask,options)[0])
        self.assertEqual(int(counts.sum()),len(distances))

    def test_variogram_failures_and_physical_range(self):
        centers = np.arange(2,200,4)
        values = metrics.exp_variogram(centers,.01,.2,60)
        fit = metrics.fit_variogram(centers,values,400)
        self.assertAlmostEqual(fit['effective_range_m'],60,places=3)
        self.assertIsNone(metrics.fit_variogram(centers,np.zeros_like(centers),400)['effective_range_m'])
        self.assertIsNone(metrics.fit_variogram(centers,np.full_like(centers,np.nan,dtype=float),400)['effective_range_m'])

    def test_centering_does_not_hide_residual_error_and_no_default_proxy(self):
        options = self.options()['analysis']
        image = np.random.default_rng(8).normal(size=(16,16)); mask=np.ones_like(image,bool);mask[:2]=False
        row,_,_ = metrics.analyse_arrays(image,image+2,mask,options)
        self.assertAlmostEqual(row['rmse_residual_m'],2)
        self.assertAlmostEqual(row['rmse_centered_m'],0,places=10)
        self.assertAlmostEqual(row['jsd_centered_1m'],0,places=10)
        self.assertEqual(row['n_valid'],224)
        self.assertNotIn('sigma_class_gt',row)
        with self.assertRaisesRegex(ValueError,'Non-finite'):
            pred=image.copy();pred[3,3]=np.nan;metrics.analyse_arrays(image,pred,mask,options)

    def test_acf_units_and_constant_field(self):
        image=np.random.default_rng(7).normal(size=(32,32));mask=np.ones_like(image,bool)
        a=metrics.acf_efold_length(image,mask,1);b=metrics.acf_efold_length(image,mask,2)
        self.assertAlmostEqual(b,2*a)
        self.assertIsNone(metrics.acf_efold_length(np.zeros((32,32)),mask))

    def fixture(self,root):
        cfg=self.options();study=root/'study';study.mkdir()
        ids=['12556','12557'];cohort=dict(split='validation',directories={},records=[dict(id=i) for i in ids])
        manifest=dict(batch_size=2,cohorts={'tuk':cohort},checkpoints={'cosine':dict(epoch=166,sha256='fixture')},
                      configurations={'cosine':dict(data=dict(pixel_size_m=1.0))})
        runner.write_json(study/'manifest.json',manifest)
        archives=[]
        for i,pid in enumerate(ids):
            path=study/f'predictions/cosine/plms/tuk/{pid}.npz';path.parent.mkdir(parents=True,exist_ok=True)
            gt=np.random.default_rng(i).normal(size=(32,32)).astype('float32')
            np.savez(path,gt=gt,prediction=gt*.8+.01,mask=np.ones_like(gt,bool))
            archives.append(dict(path=str(path.relative_to(study)),sha256=runner.sha(path)))
        runner.write_json(study/'shards/cosine_plms_tuk_000000.json',dict(manifest_digest=runner.digest(manifest),ids=ids,archives=archives))
        splits=root/'ids.json';runner.write_json(splits,{'tuk':{'validation':ids}})
        cfg['sets']={'corrected':dict(type='cache',study=str(study),schedule='cosine',sampler='plms')}
        cfg['split_ids']=str(splits)
        cfg['style'].update(formats=['png'],dpi=60,patch_figsize=[10,10],extended_figsize=[14,10])
        config=root/'figure5.yaml';config.write_text(yaml.safe_dump(cfg))
        return SimpleNamespace(out=root/'out',config=config,regions=['tuk'],patch_ids=None,
                               patch_id='12556',patch_region='tuk',workers=1,max_patches=1),cfg

    def test_verified_cache_resume_summary_and_restyle_without_analysis(self):
        from scripts import plot_figure5 as plotting
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);args,cfg=self.fixture(root)
            runner.prepare(args);m=runner.load_manifest(args.out)
            runner.work((args.out,m['records'][0],m))
            first=runner.result_path(args.out,m['records'][0]);before=first.read_bytes()
            with self.assertRaisesRegex(ValueError,'Incomplete'):
                runner.summarize(args)
            runner.run(args);self.assertEqual(before,first.read_bytes());runner.summarize(args)
            summary=json.loads((args.out/'regional_summary.json').read_text())[0]
            self.assertEqual(summary['n'],2);self.assertEqual(summary['scope'],'subset')
            style=cfg['style'];args.figure_dir=root/'figures'
            plotting.apply_style(style)
            plotting.patch_figure(args,m,style,'corrected')
            inspected={}; original_save=plotting.save
            def inspect_save(fig,name,*positional,**kwargs):
                inspected[name]=dict(titles=[ax.get_title(loc='left') for ax in fig.axes],
                                     yscales=[ax.get_yscale() for ax in fig.axes],
                                     legends=[[t.get_text() for t in ax.get_legend().get_texts()] if ax.get_legend() else [] for ax in fig.axes])
                original_save(fig,name,*positional,**kwargs)
            with patch.object(plotting,'save',side_effect=inspect_save):
                plotting.extended_figure(args,m,style,'corrected')
            roughness=inspected['figure5_extended_roughness_corrected']
            structure=inspected['figure5_extended_structure_corrected']
            self.assertEqual([t[:3] for t in roughness['titles']],['(a)','(b)','(c)'])
            self.assertEqual([t[:3] for t in structure['titles']],['(a)','(b)','(c)'])
            self.assertIn('roughness amplitude',roughness['titles'][0])
            self.assertIn('Pooled regional',roughness['titles'][1])
            self.assertIn('Roughness-class',roughness['titles'][2])
            self.assertEqual(structure['yscales'][1],'log')
            self.assertIn('ρ=',structure['legends'][0][0])
            self.assertIn('open = LiDAR',structure['legends'][2])
            provenance=json.loads((args.figure_dir/'figure5_extended_structure_corrected.json').read_text())
            self.assertEqual(provenance['panel_data']['original_panels'],['b','d','e'])
            sidecar=json.loads((args.figure_dir/'figure5_patch12556_corrected.json').read_text())
            self.assertEqual(sidecar['style']['dpi'],60)
            style['prediction_color']='darkorange';plotting.patch_figure(args,m,style,'corrected')
            self.assertEqual(before,first.read_bytes())
            record=m['records'][0];Path(record['path']).write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'archive changed'):
                runner.load_pair(record)


if __name__=='__main__':unittest.main()
