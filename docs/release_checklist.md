# RoughNet publication release

The journal requires the custom software to be archived in a recognised DOI
repository and cited from Methods or Code Availability. The release is prepared
locally first; the tools here never create a GitHub release or publish a deposit.

## Prepare locally

1. Install the runtime and run the tests (README). The historical environment is
   preserved under `environment/`; it is not the minimal installation recipe.
2. Prepare content-based inventories from the retained originals, or audit a
   relocated dataset using `--data-root`. Existing historical studies are read-only.
3. Run `python scripts/prepare_release.py checkpoints` to copy and verify the
   three unchanged paper checkpoints and bind them to the portable inventories.
4. Run `python scripts/export_paper_tables.py --allow-incomplete` while the U-Net
   contribution is pending. It produces explicitly incomplete draft tables; its
   default mode requires the baseline's per-patch results. The baseline CSV must
   contain Cambridge test IDs, `region`, `tile_id`, and the eight existing metric
   columns; optional `split` must be `test`.
5. Run `python scripts/prepare_release.py build --allow-incomplete` to stage an
   isolated, allowlisted software checkout and a separate weights/results archive
   under `build/release/`. It excludes local credentials, internal drafts, extra
   figures, full data, recovery checkpoints and ground-truth prediction caches.
6. Inspect `build/release/readiness.json` and the archive checksums. Fill pending
   publication metadata in `release/publication.json` only when confirmed.

All 11 final main and supplementary figures supplied from Overleaf are present
under `figures/final_figures/`. `release/figures.json` maps manuscript filenames
to repository assets and binds their checksums. Release preparation verifies
these files and includes them in the software archive. Additional historical
figures can remain in the development repository.

## Required before publication

- Collaborator-selected code and artifact licenses, with the corresponding
  license text and source-data terms. No permissive license is assumed.
- Weibin's baseline implementation integrated into the source tree, its selected
  checkpoint (`roughnet_unet_baseline_best.pth`), config, selected epoch and per-patch measurements.
- A final build without `--allow-incomplete`, passing tests, validated
  `CITATION.cff`, clean-checkout installation and inference checks.

See `docs/validation.md` for completed checks and their limits.

## Archive and cite

Use the staged `software/` tree as the contents of an isolated publication
branch, preserving this development checkout. Check that the branch's files
match the staged checksums; do not tag the unfiltered development tree.

Enable Zenodo integration for the GitHub repository **before** creating the
final `v1.0.0` GitHub release. Upload the allowlisted weights/results archive as
its own Zenodo deposit and link the two records. The three named software
creators are Tessa Cannon, Michel Tsamados and Weibin Chen; acknowledge the other
paper authors through the manuscript and their relevant contributions.

Verify both downloaded archive checksums, inspect the software record's actual
contents and confirm its version-specific DOI resolves. Record the DOIs in
publication metadata and use `docs/code_availability.md` to update the manuscript.
The software DOI is essential for the journal requirement; the separate weights/
results DOI is additional. No full prepared dataset is redistributed; the small
existing tracked sample is retained with source attribution.

Historical source snapshots and byte-identical original checkpoints remain
available for provenance. New runtime checks do not authorize resuming changed
source into historical studies. Exact numerical GPU reproduction requires the
recorded runtime and original inputs; archived table generation requires neither.
