#!/usr/bin/env python3
"""Prepare isolated, allowlisted software and weights/results archives. Never publishes."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
PAPER={
 'roughnet_cosine':dict(legacy='original_clean',fold='original_clean',study='noise_sampler_clean',schedule='cosine',epoch=166,sha256='53f42e1075573b4367b7cb7a12ee617fa0d43691ae0163a9305b858a0a5b047c',paper='Final model; Tables 1, 2, S1 and figures'),
 'roughnet_linear':dict(legacy='original_clean_linear',fold='original_clean',study='noise_sampler_clean',schedule='linear',epoch=93,sha256='00054cad9fc62edec4472e056d450ec99a8bbe5fa04e356e9ccdfc6263be46e3',paper='Linear rows in Tables 1 and S1'),
 'roughnet_cosine_tuk_heldout':dict(legacy='tuk_heldout_clean',fold='tuk_heldout_clean',study='tuk_heldout_clean_evaluation',schedule='cosine',epoch=134,sha256='610d97c9e9ed11c406593311afa01f488e24da096400721fe210835b59aa4ce7',paper='Geographic transfer; Table S2'),
}
SCRIPT_NAMES=('__init__.py','train_roughnet.py','train_roughnet.sh','prepare_spatial_splits.py',
 'evaluation.py','noise_sampler_ablation.py','run_noise_sampler_ablation.sh','conditioning_ablation.py',
 'prepare_corrected_conditioning.py','run_corrected_conditioning_ablation.sh','alternate_fold_evaluation.py',
 'run_alternate_fold_evaluation.sh','image_free_baseline.py','run_image_free_baseline.sh',
 'figure5_analysis.py','fig5_statistics.py','plot_figure5.py','run_figure5_analysis.sh',
 'export_paper_tables.py','prepare_release.py','verify_checkpoint.py')
FIGURES=('model_diagram.png','sample_patch.png','patch_recontructions_pdfs_clean.png','conditioning_ablation_examples.png',
 'tuk_mosaic_2d_demeaned.png','pondinlet_mosaic_2d_demeaned.png','cambridge_mosaic_2d_demeaned.png',
 'figure5/figure5_patch12556_corrected.png','figure5/figure5_extended_roughness_corrected.png','figure5/figure5_extended_structure_corrected.png')
STUDIES=('noise_sampler_clean','conditioning_ablation_clean','tuk_heldout_clean_evaluation','image_free_baseline_clean','fig5_extended_clean')


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()


def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 text=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
 if path.exists() and path.read_text()!=text:raise FileExistsError(f'Refusing to replace different release metadata: {path}')
 if not path.exists():path.write_text(text)


def copy_checked(source,dest,checksum=None):
 source,dest=Path(source),Path(dest)
 checksum=checksum or sha(source)
 if dest.exists():
  if sha(dest)!=checksum:raise FileExistsError(f'Refusing to replace {dest}')
 else:
  dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
 if sha(dest)!=checksum:raise ValueError(f'Checksum mismatch: {dest}')


def checkpoints(repo=ROOT):
 from src.utils.spatial_split import validate_manifest,portable_digest,digest
 repo=Path(repo);hashes={};manifests={}
 for fold in ('original_clean','tuk_heldout_clean'):
  legacy=json.loads((repo/'splits'/f'{fold}.json').read_text())
  # Bootstrap equivalence only against the original audited local inputs.
  validate_manifest(legacy,repo)
  portable=copy.deepcopy(legacy);portable['version']=2
  for i,row in enumerate(portable['patches']):
   for record in row['files']:
    path=repo/record['path']
    if record['path'] not in hashes:hashes[record['path']]=sha(path)
    record['sha256']=hashes[record['path']];record.pop('mtime_ns',None)
   if (i+1)%2000==0:print(f'{fold}: {i+1}/{len(portable["patches"])} content records',flush=True)
  validate_manifest(portable)
  write(repo/'splits'/f'{fold}_portable.json',portable)
  manifests[fold]=(digest(legacy),portable_digest(portable))
 records=[]
 for name,spec in PAPER.items():
  source=repo/'models'/spec['legacy']/(spec['legacy']+'_best.pth')
  old_manifest=json.loads((repo/'reconstructions'/spec['study']/'manifest.json').read_text())
  observed=old_manifest['checkpoints'][spec['schedule']]
  if observed['sha256']!=spec['sha256'] or observed['epoch']!=spec['epoch'] or sha(source)!=spec['sha256']:
   raise ValueError(f'Unexpected paper checkpoint: {source}')
  dest=repo/'models'/name/(name+'_best.pth');copy_checked(source,dest,spec['sha256'])
  records.append(dict(name=name,path=str(dest.relative_to(repo)),original_path=str(source.relative_to(repo)),
                      sha256=spec['sha256'],epoch=spec['epoch'],paper=spec['paper'],
                      legacy_split_digest=manifests[spec['fold']][0],portable_split_digest=manifests[spec['fold']][1],
                      manifest='splits/'+spec['fold']+'_portable.json'))
 write(repo/'release/checkpoints.json',dict(schema=1,checkpoints=records))
 print('Verified and copied three unchanged publication checkpoints.',flush=True)
 return records


def release_gates(repo,metadata):
 missing=[]
 for key,file in [('code_license','LICENSE'),('artifact_license','LICENSE.artifacts')]:
  if not metadata.get(key) or not (repo/file).is_file():missing.append(f'{key}: collaborator decision and {file}')
 for key in ('unet_checkpoint','unet_results','unet_config'):
  if not metadata.get(key) or not (repo/metadata[key]).is_file():missing.append(key+': colleague contribution')
 if not isinstance(metadata.get('unet_epoch'),int) or metadata['unet_epoch'] < 1:
  missing.append('unet_epoch: selected baseline checkpoint epoch')
 if not metadata.get('unet_code') or any(not (repo/p).is_file() for p in metadata.get('unet_code',[])):
  missing.append('unet_code: colleague implementation')
 if not (repo/'figures/lidar_map.pdf').is_file():missing.append('figures/lidar_map.pdf')
 if not metadata.get('final_panels_confirmed'):missing.append('final illustrative panel confirmation')
 return missing


def software_files(repo,metadata):
 files=[Path(x) for x in ('README.md','config.yaml','requirements.txt','requirements-notebooks.txt','requirements-dev.txt','CITATION.cff','src/requirements.txt','release/publication.json','release/checkpoints.json',
  'docs/baseline_data_split.md','docs/release_checklist.md','docs/code_availability.md','docs/validation.md','configs/figure5.yaml',
  '.github/workflows/tests.yml',
  'splits/train_val_split_ids.json','splits/tuk_heldout_clean_ids.json')]
 files += [Path('scripts')/n for n in SCRIPT_NAMES]
 files += list(Path('src').rglob('*.py')) if repo==Path.cwd() else [p.relative_to(repo) for p in (repo/'src').rglob('*.py')]
 files += [p.relative_to(repo) for p in (repo/'tests').glob('test_*.py') if p.name!='test_pondinlet_fold_evaluation.py']
 files += [p.relative_to(repo) for p in (repo/'environment').glob('*') if p.is_file()]
 files += [Path('splits')/(n+'.yaml') for n in PAPER]
 files += [Path('notebooks')/n for n in ('data_collocation.ipynb','patching.ipynb','patch_inference.ipynb')]
 for n in FIGURES:
  p=Path('figures/final_figures')/n;files.append(p)
  for suffix in ('.json','.pdf'):
   if (repo/p.with_suffix(suffix)).is_file():files.append(p.with_suffix(suffix))
 for folder in ('input_data/sample_lidar_patch','input_data/sample_s2_patch'):
  files += [p.relative_to(repo) for p in (repo/folder).rglob('*') if p.is_file() and not p.name.endswith('.aux.xml')]
 for name in ('LICENSE','LICENSE.artifacts','figures/lidar_map.pdf'):
  if (repo/name).is_file():files.append(Path(name))
 for name in metadata.get('unet_code') or []:files.append(Path(name))
 if metadata.get('unet_config'):files.append(Path(metadata['unet_config']))
 return sorted(set(files))


def verify_software_files(files):
 for p in files:
  if p.is_absolute() or '..' in p.parts or any(x in p.parts for x in ('.ssh','.git','.aws','__pycache__')):
   raise ValueError(f'Unsafe software path {p}')
  if p.name=='copernicus_login.py' or p.suffix in ('.pth','.npz'):raise ValueError(f'Unexpected artifact in software tree: {p}')
  if p.parts[0]=='input_data' and p.parts[1] not in ('sample_lidar_patch','sample_s2_patch'):raise ValueError('Full data cannot enter software archive')


def source_snapshot(repo,path,checksum):
 current=repo/path
 if current.is_file() and sha(current)==checksum:return current.read_bytes()
 revisions=subprocess.check_output(['git','log','--all','--format=%H','--',path],cwd=repo,text=True).splitlines()
 for rev in revisions:
  result=subprocess.run(['git','show',rev+':'+path],cwd=repo,capture_output=True)
  if result.returncode==0 and hashlib.sha256(result.stdout).hexdigest()==checksum:return result.stdout
 raise ValueError(f'Historical source not recoverable: {path} ({checksum})')


def archive_files(folder):
 return [p for p in sorted(folder.rglob('*')) if p.is_file()
         and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo')]


def zip_tree(folder,destination):
 with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED) as archive:
  for p in archive_files(folder):
    info=zipfile.ZipInfo(str(p.relative_to(folder)),date_time=(1980,1,1,0,0,0))
    info.compress_type=zipfile.ZIP_DEFLATED
    archive.writestr(info,p.read_bytes())


def build(repo=ROOT,out=None,allow_incomplete=False):
 from scripts.export_paper_tables import export,INPUTS
 repo=Path(repo).resolve();out=Path(out or repo/'build/release').resolve()
 if out==repo or repo.is_relative_to(out):raise ValueError('Release output must not contain the source checkout')
 metadata=json.loads((repo/'release/publication.json').read_text());missing=release_gates(repo,metadata)
 if missing and not allow_incomplete:raise ValueError('Publication gates incomplete: '+', '.join(missing))
 if (out/'software').exists() or (out/'artifacts').exists():raise FileExistsError('Use a new --out for a different release staging run')
 files=software_files(repo,metadata);verify_software_files(files)
 for p in files:
  if not (repo/p).is_file():raise FileNotFoundError(repo/p)
 registry=json.loads((repo/'release/checkpoints.json').read_text())
 out.mkdir(parents=True,exist_ok=True);software=out/'software';artifacts=out/'artifacts'
 for p in files:
  copy_checked(repo/p,software/p)
  if p.suffix=='.ipynb':
   n=json.loads((software/p).read_text())
   for c in n.get('cells',[]):
    if c.get('cell_type')=='code':c['outputs']=[];c['execution_count']=None
   (software/p).write_text(json.dumps(n,indent=1,ensure_ascii=False)+'\n')
 for record in registry['checkpoints']:
  copy_checked(repo/record['path'],artifacts/record['path'],record['sha256'])
  copy_checked(repo/record['manifest'],artifacts/record['manifest'])
 for p in ('release/checkpoints.json','release/publication.json','splits/train_val_split_ids.json','splits/tuk_heldout_clean_ids.json',*[f'splits/{n}.yaml' for n in PAPER]):copy_checked(repo/p,artifacts/p)
 for p in ('CITATION.cff','LICENSE.artifacts'):
  if (repo/p).is_file():copy_checked(repo/p,artifacts/p)
 write(artifacts/'artifact_metadata.json',dict(title='RoughNet: trained weights and derived evaluation results',
        creators=metadata['software_creators'],version=metadata['version'],license=metadata.get('artifact_license'),
        related_software_doi=metadata.get('software_doi')))
 for rel in [*INPUTS.values(),'image_free_baseline_clean/assignments.csv']:
  source=repo/'reconstructions'/rel
  # Measurement CSVs contain paths relative to input_data, never raw arrays.
  text=source.read_text()
  if str(repo) in text:raise ValueError(f'Local absolute path requires explicit export normalization: {rel}')
  copy_checked(source,artifacts/'results'/rel)
 lineage={};snapshots={}
 for study in STUDIES:
  source=repo/'reconstructions'/study/'manifest.json';m=json.loads(source.read_text())
  lineage[study]=dict(original_manifest_sha256=sha(source),source_hashes=m.get('sources',{}))
  for path,checksum in m.get('sources',{}).items():
   logical='scripts/'+path if study=='fig5_extended_clean' else path
   key=(logical,checksum)
   if key not in snapshots:snapshots[key]=source_snapshot(repo,logical,checksum)
  runtime=source.parent/'runtime.json'
  if runtime.is_file():copy_checked(runtime,artifacts/'provenance'/study/'runtime.json')
 for (path,checksum),contents in snapshots.items():
  destination=artifacts/'provenance/sources'/checksum/path;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(contents)
 write(artifacts/'provenance/lineage.json',lineage)
 analysis=repo/'reconstructions/fig5_extended_clean'
 for name in ('per_patch.csv','regional_summary.csv','regional_summary.json','regional_summary.md','regional_summary.tex','patch_representativeness.csv'):
  copy_checked(analysis/name,artifacts/'statistics'/name)
 for p in sorted((analysis/'patches').rglob('*.json')):
  original=json.loads(p.read_text());relative=p.relative_to(analysis)
  write(artifacts/'statistics'/relative,dict(row=original['row'],details=original['details'],original_record_sha256=sha(p)))
  arrays=p.with_suffix('.npz')
  with zipfile.ZipFile(arrays) as z:
   if set(z.namelist())!={'sv_gt.npy','sv_pred.npy','pair_counts.npy','hist_gt.npy','hist_pred.npy'}:raise ValueError('Unexpected non-statistical analysis array')
  copy_checked(arrays,(artifacts/'statistics'/relative).with_suffix('.npz'))
 if metadata.get('unet_results'):
  copy_checked(repo/metadata['unet_results'],artifacts/'results/unet.csv')
 if metadata.get('unet_checkpoint'):
  copy_checked(repo/metadata['unet_checkpoint'],artifacts/'models/roughnet_unet_baseline/roughnet_unet_baseline_best.pth')
 if metadata.get('unet_config'):copy_checked(repo/metadata['unet_config'],artifacts/'baseline/config.yaml')
 model_inventory=copy.deepcopy(registry['checkpoints'])
 if metadata.get('unet_checkpoint'):
  model_inventory.append(dict(name='roughnet_unet_baseline',path='models/roughnet_unet_baseline/roughnet_unet_baseline_best.pth',
                              sha256=sha(repo/metadata['unet_checkpoint']),epoch=metadata.get('unet_epoch'),paper='Deterministic baseline in main Table 2'))
 write(artifacts/'models/manifest.json',dict(checkpoints=model_inventory))
 tables,tablemeta=export(artifacts/'results',artifacts/'splits/train_val_split_ids.json',artifacts/'splits/tuk_heldout_clean_ids.json',artifacts/'tables',
                       artifacts/'results/unet.csv' if metadata.get('unet_results') else None,allow_incomplete)
 (artifacts/'README.md').write_text('RoughNet weights and derived results. No full prepared inputs or prediction rasters.\n'
  'Recreate tables with scripts/export_paper_tables.py --results-root results --split-ids splits/train_val_split_ids.json '
  '--alternate-ids splits/tuk_heldout_clean_ids.json --out new_tables'+(' --unet-results results/unet.csv' if metadata.get('unet_results') else ' --allow-incomplete')+'\n')
 for tree in (software,artifacts):
  checks={str(p.relative_to(tree)):sha(p) for p in archive_files(tree)}
  write(tree/'SHA256SUMS.json',checks)
  zip_tree(tree,out/(tree.name+'.zip'))
 report=dict(version=metadata['version'],ready_to_publish=not missing,missing_inputs=missing,tables_complete=tablemeta['complete'],
             software_creators=metadata['software_creators'],publication_steps_pending=['Enable Zenodo GitHub integration','Publish final GitHub release after gates pass','Upload and cross-link artifacts deposit','Verify version-specific software DOI and insert into manuscript'],
             journal_requirement_satisfied=False,archives={n:sha(out/n) for n in ('software.zip','artifacts.zip')})
 write(out/'readiness.json',report)
 print(json.dumps(report,indent=2));return report


def main():
 parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
 sub.add_parser('checkpoints',help='Bootstrap portable inventories from original audited inputs and copy paper weights')
 p=sub.add_parser('build');p.add_argument('--out',type=Path);p.add_argument('--allow-incomplete',action='store_true')
 args=parser.parse_args()
 try:
  if args.command=='checkpoints':checkpoints()
  else:build(out=args.out,allow_incomplete=args.allow_incomplete)
 except (ValueError,OSError,KeyError,subprocess.CalledProcessError) as e:parser.exit(2,str(e)+'\n')

if __name__=='__main__':main()
