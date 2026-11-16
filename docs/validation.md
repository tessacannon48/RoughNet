# Publication preparation validation

The release implementation was checked locally with:

- All 43 tests passing from a clean copy of the committed source in a newly
  created Python 3.13 environment with CPU PyTorch 2.7.0 and the consolidated
  development requirements. The full working checkout passed 46 tests in both
  the existing and new environments; its three additional local research tests
  and related third-fold changes were left out of the publication commits.
- `pip check`: no broken requirements in the new environment.
- `cffconvert --validate`: citation metadata valid against CFF 1.2.0.
- Original and publication checkpoint SHA-256 values identical for all three
  models, with selected epochs 166, 93 and 134 bound in the registry.
- All 24 main/supplementary ablation rows, ten available Table 2 rows and three
  transfer rows agreeing with the supplied manuscript at displayed precision,
  including all reported ablation mean ranks.
- A synthetic copied dataset with changed timestamps retaining its content
  identity; changed bytes, geometry, membership, or unbound checkpoint identity
  rejected. Resume signatures retain content identity while allowing root paths
  to move.
- Sample loading and bounded single-forward-pass CPU inference with the released
  architecture. These checks do not constitute full 1,000-step diffusion reruns.
- README links, example syntax, shell syntax, CLI argument parsing and release
  allowlist checks.
- Strict release preparation refusing to proceed without licenses, the baseline
  contribution and final figure confirmation.

CUDA is unavailable in this execution environment, so no new GPU inference or
full training was performed. The historical paper runtime is recorded in
`environment/paper-runtime.json`. All historical checkpoints, reconstruction
studies and original timestamp inventories remain unchanged. The standalone
paper-table exporter uses the retained measurements and does not recalculate
predictions or alter the paper's metric definitions.

The deterministic U-Net row remains explicitly incomplete until the colleague's
per-patch results are supplied. No code/artifact license, DOI, public GitHub
release or Zenodo deposit has been created or assumed.
