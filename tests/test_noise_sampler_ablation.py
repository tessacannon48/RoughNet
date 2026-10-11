from pathlib import Path
import sys,tempfile,unittest,json
from types import SimpleNamespace
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.diffusion.sampling import p_sample_loop_ddpm,p_sample_loop_ddim
from src.diffusion.scheduler import LinearDiffusionScheduler
from src.utils.conditioning_ablation import write_json,sha256_file,json_digest,METRICS
from scripts.noise_sampler_ablation import valid_shard,summarize

class Tests(unittest.TestCase):
    def test_paired_noise_not_mutated_and_shape_guard(self):
        scheduler=LinearDiffusionScheduler(timesteps=4,device='cpu');model=lambda x,c,a,t:x*0.1
        noise=torch.randn(2,1,4,4);original=noise.clone()
        for sampler in (p_sample_loop_ddpm,p_sample_loop_ddim):
            a=sampler(model,scheduler,noise.shape,None,None,'cpu',initial_noise=noise)
            self.assertTrue(torch.isfinite(a).all());torch.testing.assert_close(noise,original,rtol=0,atol=0)
            with self.assertRaises(ValueError):sampler(model,scheduler,(1,1,4,4),None,None,'cpu',initial_noise=noise)
        gen=lambda:torch.Generator().manual_seed(17)
        a=p_sample_loop_ddpm(model,scheduler,noise.shape,None,None,'cpu',initial_noise=noise,generator=gen())
        torch.randn(10)
        b=p_sample_loop_ddpm(model,scheduler,noise.shape,None,None,'cpu',initial_noise=noise,generator=gen())
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_summary_completeness_pooling_and_archive_integrity(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);m={'batch_size':2,'cohorts':{'pondinlet':{'split':'validation','records':[{'id':'a'}]},'tuk':{'split':'validation','records':[{'id':'b'},{'id':'c'}]}}}
            write_json(out/'manifest.json',m)
            with self.assertRaises(ValueError):summarize(SimpleNamespace(out=out,split='validation'))
            for schedule in ['linear','cosine']:
                for sampler in ['ddpm','ddim','plms']:
                    for region,value in [('pondinlet',1),('tuk',4)]:
                        ids=[r['id'] for r in m['cohorts'][region]['records']];rows=[];archives=[]
                        for pid in ids:
                            path=f'predictions/{schedule}/{sampler}/{region}/{pid}.npz'
                            (out/path).parent.mkdir(parents=True,exist_ok=True);np.savez(out/path,prediction=np.zeros((2,2)))
                            archives.append(dict(path=path,sha256=sha256_file(out/path)))
                            rows.append(dict(schedule=schedule,sampler=sampler,region=region,split='validation',tile_id=pid,**{k:value for k in METRICS}))
                        write_json(out/'shards'/f'{schedule}_{sampler}_{region}_000000.json',dict(manifest_digest=json_digest(m),ids=ids,rows=rows,archives=archives))
            summarize(SimpleNamespace(out=out,split='validation'))
            summary=json.loads((out/'validation/comparison.json').read_text())
            self.assertTrue(all(r['rmse_m_mean']==3 and r['n']==3 for r in summary if r['region']=='pooled_validation'))
            p=out/'shards/linear_ddpm_pondinlet_000000.json';(out/'predictions/linear/ddpm/pondinlet/a.npz').write_bytes(b'corrupted')
            with self.assertRaises(ValueError):valid_shard(p,m,'linear','ddpm','pondinlet',['a'],out)
if __name__=='__main__':unittest.main()
