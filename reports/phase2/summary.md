# Phase 2 data-pipeline completion record

Phase 2 was completed in commit `6af7da2` (`Build validated Phase 2 data pipeline`). The committed implementation covers MAF parsing and SNV filtering, GRCh38 sequence extraction, reference-allele and strand checks, COSMIC-CGC proxy labeling, deterministic gene-level splitting, and final split validation.

The run-specific Phase 2 JSON artifacts were written beneath `data/processed/phase2/` and were never committed because `data/` intentionally excludes private and derived cohort data. They are not present in this checkout. Recreating the exact historical JSON reports requires the original TCGA-BRCA MAF, GRCh38 FASTA files, liftover chain, COSMIC CGC file, and the saved `gene-split-v1` dataset. Those licensed/private inputs are not available locally, so exact regeneration is not cheap or currently possible.

Durable facts recoverable from committed project documentation are:

- Dataset size: 113,525 variants.
- Positive proxy labels: 8,387 variants.
- Split identifier: `gene-split-v1`.
- Gene overlap between train, validation, and test: zero.
- Test variants: 11,356.
- Test variants whose patients also occur in train: 11,356 (100%).
- Reverse-chain records with validated reference concordance: 455/455 (100%).

The pipeline can regenerate its operational reports when the required private inputs are restored:

- `parse_report.json`
- `reference_concordance.json`
- `sequence_report.json`
- `label_report.json`
- `split_report.json`
- `phase2_validation.json`

This retrospective record does not replace those run-specific artifacts and does not claim values that cannot be recovered from versioned evidence.
