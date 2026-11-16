"""Publication compatibility, ranking and packaging safety checks."""
import copy
import csv
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from src.utils.spatial_split import digest,portable_digest,validate_manifest,verify_checkpoint_split,sha256_file
from scripts.export_paper_tables import add_ranks,rounded,validate_ids,groups,METRICS
from scripts.prepare_release import copy_checked,verify_software_files,release_gates

class PortableTests(unittest.TestCase):
 def fixture(self,root):
  rows=[]
  for i,split in enumerate(('train','validation','test')):
   path=root/'input_data'/f'patch{i}.bin';path.parent.mkdir(exist_ok=True);path.write_bytes(bytes([i+1]))
   rows.append(dict(id=str(i),region='test' if split=='test' else 'source',split=split,crs='EPSG:32617',bounds=[i*3,0,i*3+1,1],files=[dict(path=str(path.relative_to(root)),size=1,sha256=sha256_file(path))]))
  return dict(version=2,patches=rows,validation_zones={'source':4},test_region='test')
 def test_copy_and_timestamp_change_preserve_identity(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);first=root/'first';first.mkdir();m=self.fixture(first);expected=digest(m)
   second=root/'second';shutil.copytree(first,second)
   for p in (second/'input_data').glob('*'):os.utime(p,(100000,100000))
   validate_manifest(m,data_root=second/'input_data');self.assertEqual(digest(m),expected)
   (second/'input_data/patch0.bin').write_bytes(b'X')
   with self.assertRaisesRegex(ValueError,'Input changed'):validate_manifest(m,data_root=second/'input_data')
 def test_changed_geometry_membership_or_missing_hash_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   m=self.fixture(Path(d));before=digest(m)
   other=copy.deepcopy(m);other['patches'][0]['bounds']=[0,0,2,1];self.assertNotEqual(digest(other),before)
   other=copy.deepcopy(m);other['patches'][0]['id']='different';self.assertNotEqual(digest(other),before)
   other=copy.deepcopy(m);del other['patches'][0]['files'][0]['sha256']
   with self.assertRaisesRegex(ValueError,'SHA-256'):digest(other)
 def test_legacy_compatibility_is_bound_to_checkpoint_and_contents(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);m=self.fixture(root);ckpt=root/'weights';ckpt.write_bytes(b'original')
   registry=root/'registry.json';registry.write_text(json.dumps(dict(checkpoints=[dict(sha256=sha256_file(ckpt),portable_split_digest=digest(m),legacy_split_digest='legacy')])))
   cfg={'data':{'split_digest':'legacy'}};verify_checkpoint_split(cfg,m,ckpt,registry)
   ckpt.write_bytes(b'changed')
   with self.assertRaises(ValueError):verify_checkpoint_split(cfg,m,ckpt,registry)
   ckpt.write_bytes(b'original');m['patches'][0]['id']='other'
   with self.assertRaises(ValueError):verify_checkpoint_split(cfg,m,ckpt,registry)
 def test_relative_paths_cannot_escape_data_root(self):
  from src.utils.spatial_split import data_path
  for value in ('../secret','/tmp/secret'):
   with self.assertRaises(ValueError):data_path(value,data_root='/tmp/data')
 def test_resume_signature_uses_data_identity_instead_of_locations(self):
  from src.utils.training_state import resume_signature
  cfg=dict(model={},training={},system={},data=dict(split_digest='same',data_root='/a',lidar_dirs=['/a/lidar'],s2_dirs=['/a/s2'],pixel_size_m=1))
  moved=copy.deepcopy(cfg);moved['data'].update(data_root='/b',lidar_dirs=['/b/lidar'],s2_dirs=['/b/s2'])
  self.assertEqual(resume_signature(cfg),resume_signature(moved))
  moved['data']['split_digest']='different';self.assertNotEqual(resume_signature(cfg),resume_signature(moved))

class TableTests(unittest.TestCase):
 def test_bias_direction_ties_and_half_up(self):
  first={m:1 for m in METRICS};second=dict(first)
  first.update(bias_m=-.1,zncc=.9);second.update(bias_m=.2,zncc=.8)
  rows=add_ranks([first,second]);self.assertEqual([r['mean_rank'] for r in rows],[1.375,1.625])
  self.assertEqual(rounded(5.125,2),'5.13');self.assertEqual(rounded(-.00001,4),'0.0000')
 def test_wrong_or_duplicate_ids_fail_even_when_counts_match(self):
  for rows in ([{'tile_id':'a'},{'tile_id':'a'}],[{'tile_id':'a'},{'tile_id':'x'}]):
   with self.assertRaises(ValueError):validate_ids(rows,['a','b'])
  with self.assertRaises(ValueError):groups([{'sampler':'unexpected'}],('sampler',),[('plms',)])

 def test_full_export_validates_cohorts_and_equal_region_weighting(self):
  from scripts.export_paper_tables import build_tables,render,CONFIGS,INPUTS
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);ids={'pondinlet':['p0','p1'],'tuk':['t0'],'cambridge':['c0','c1','c2']}
   primary={r:{'test' if r=='cambridge' else 'validation':v} for r,v in ids.items()}
   split=root/'primary.json';split.write_text(json.dumps(primary))
   alternate=root/'alternate.json';alternate.write_text(json.dumps({'tuk':{'test':['t0','t1','t2','t3']}}))
   def row(region,pid,**kw):return dict(region=region,tile_id=pid,**{k:1.0 for k in METRICS},**kw)
   def save(key,rows):
    p=root/INPUTS[key];p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:
     writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
   validation=[];test=[];conditioning=[]
   for region,pids in ids.items():
    for schedule,sampler in CONFIGS:
     for pid in pids:
      r=row(region,pid,schedule=schedule,sampler=sampler,split='test' if region=='cambridge' else 'validation')
      r['rmse_m']=9 if region=='pondinlet' else 1
      (test if region=='cambridge' else validation).append(r)
    for mode in ('normal','shuffle','null'):
     conditioning += [row(region,pid,mode=mode,seed=42) for pid in pids]
   save('validation',validation);save('test',test);save('conditioning',conditioning)
   save('image_free',[row('cambridge',pid) for pid in ids['cambridge']])
   save('alternate',[row('tuk','t'+str(i),schedule='cosine',sampler='plms',split='test') for i in range(4)])
   with self.assertRaisesRegex(ValueError,'U-Net results required'):build_tables(root,split,alternate,expected_counts=(2,1,3,4))
   tables,meta=build_tables(root,split,alternate,allow_incomplete=True,expected_counts=(2,1,3,4))
   equal=[r for r in tables['main_table_1'] if r['panel']=='equal_region']
   self.assertEqual([r['rmse_m'] for r in equal],[5]*6)
   self.assertEqual([r['n'] for r in tables['supplementary_table_2']],[4,1,1])
   self.assertFalse(meta['complete']);self.assertIn('INCOMPLETE',render(tables,meta)['main_table_2.tex'])
   validation[0]['tile_id']='wrong';save('validation',validation)
   with self.assertRaisesRegex(ValueError,'patch IDs'):build_tables(root,split,alternate,allow_incomplete=True,expected_counts=(2,1,3,4))

class ArchiveTests(unittest.TestCase):
 def test_archive_ignores_generated_bytecode(self):
  from scripts.prepare_release import zip_tree
  import zipfile
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)/'source';root.mkdir();(root/'example.py').write_text('pass')
   (root/'__pycache__').mkdir();(root/'__pycache__/example.pyc').write_bytes(b'cache')
   target=Path(d)/'source.zip';zip_tree(root,target)
   with zipfile.ZipFile(target) as z:self.assertEqual(z.namelist(),['example.py'])
 def test_archive_excludes_raw_inputs_credentials_and_weights(self):
  verify_software_files([Path('src/model/unet.py'),Path('input_data/sample_lidar_patch/example.tif')])
  for p in ('.ssh/id_rsa','../secret','copernicus_login.py','models/weights.pth','input_data/lidar_patches_tuk/a.tif'):
   with self.assertRaises(ValueError):verify_software_files([Path(p)])
 def test_copy_does_not_overwrite_changed_historical_weights(self):
  with tempfile.TemporaryDirectory() as d:
   source=Path(d)/'source';target=Path(d)/'target';source.write_bytes(b'original');copy_checked(source,target)
   self.assertEqual(sha256_file(source),sha256_file(target));source.write_bytes(b'changed')
   with self.assertRaises(FileExistsError):copy_checked(source,target)
 def test_licenses_and_baseline_are_real_gates(self):
  with tempfile.TemporaryDirectory() as d:
   missing=release_gates(Path(d),{})
   self.assertTrue(any('code_license' in x for x in missing));self.assertTrue(any('unet_checkpoint' in x for x in missing))

if __name__=='__main__':unittest.main()
