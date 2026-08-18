# Project Specification: Genomic Variant Effect Predictor

**Project codename:** `variantfx` (working name, can be changed)

**One-line description:** A reproducible pipeline that fine-tunes a genomic language model to predict whether somatic cancer mutations are functionally impactful ("driver-like") or not, and outputs a ranked, batch-scored report from a VCF/MAF input file.

**Audience for this doc:** This document is meant to be handed to an AI coding assistant (Claude Code, Codex, or similar) as full project context. It should be sufficient, on its own, for an assistant to understand what to build, in what order, and what mistakes to avoid — without needing to re-derive any of the reasoning from scratch.

**How to use this doc during the build:** Work through the phases in order. Each phase has a concrete deliverable and a "done when" checkpoint. Do not start a phase until the previous one's checkpoint is met. After each step, the human will review before moving to the next — do not batch multiple phases into one code dump unless explicitly asked.

---

## 1. Project Goal & Motivation

Every cancer genome sequenced turns up thousands of somatic mutations, but only a small fraction are biologically meaningful ("driver" mutations that contribute to cancer progression) — the rest are inert "passenger" mutations. Manually triaging which mutations matter is slow. This project builds a tool that:

1. Takes a cohort's somatic mutation calls (MAF/VCF format) as input
2. Extracts DNA sequence context around each variant
3. Uses a fine-tuned genomic language model to predict functional impact
4. Outputs a ranked, human-readable report a researcher could actually use to prioritize follow-up

This is explicitly scoped as a **fine-tuning + productionization** project, not a from-scratch model architecture project. The value is in doing the full pipeline — data handling, fine-tuning, evaluation, packaging, reproducibility — correctly and rigorously, not in inventing a new model architecture.

---

## 2. Scope Decisions (already made — do not relitigate these without flagging to the human first)

- **Variant types:** SNVs (single nucleotide variants) only for v1. No indels initially — they shift sequence length and complicate windowed extraction. Add later if time allows.
- **Label source:** Driver vs. passenger classification, using COSMIC Cancer Gene Census (CGC) gene membership as the (weak/proxy) label source, applied to somatic mutations from a single cancer cohort.
- **Cancer cohort:** Start with **one** TCGA cancer type via cBioPortal (e.g., breast or lung — pick whichever has a well-documented, clean public study). Do not pull multiple cancer types in v1 — see Edge Cases section on tissue-specific driver labels.
- **Genome build:** GRCh38. This must be used consistently everywhere — reference FASTA, coordinate systems, liftover if needed. This single decision must be documented prominently in the README and verified with an automated sanity check (see Section 6).
- **Model:** Nucleotide Transformer (InstaDeep, available via HuggingFace). Start with the **smallest available parameter variant** (e.g., 50M), not the largest. Scale up only after the full pipeline works end-to-end.
- **Data volume:** Deliberately small and scoped — one cancer type, possibly one or a few chromosomes to start, likely low thousands of labeled variants after cleaning. This is intentional, not a limitation to apologize for.

---

## 3. Tech Stack

| Layer | Tool | Notes |
|---|---|---|
| Environment management | conda/mamba | Required, not optional — genomics tooling (htslib-based packages) is fragile under pure pip/venv |
| Sequence/VCF/MAF handling | `pysam`, `pyfaidx` | Use these for parsing, not hand-rolled parsers |
| Coordinate liftover (if needed) | `pyliftover` or UCSC liftOver | Only needed if source data isn't already GRCh38 |
| Modeling | HuggingFace `transformers`, PyTorch | Nucleotide Transformer checkpoint |
| Baseline model | scikit-learn | Logistic regression / random forest on hand-crafted features |
| Experiment tracking | MLflow or Weights & Biases | Set up from the start, not retrofitted |
| Pipeline orchestration | Snakemake | Reproducible, config-driven DAG of pipeline steps |
| Packaging | Docker | Pin all dependency versions inside the image |
| CLI | `click` or `argparse` | e.g. `variantfx score --input cohort.vcf --output report/` |
| Version control | Git, with disciplined commits per working checkpoint | Commit after each phase's "done when" checkpoint is met |

---

## 4. Data Sources

| Data | Source | Access notes |
|---|---|---|
| Somatic mutation calls (MAF) | cBioPortal, one TCGA study | Public, open access, no approval needed. Use their API or bulk-download the single study (not the full TCGA archive). |
| Reference genome (GRCh38 FASTA) | UCSC or Ensembl | Only download the specific chromosome(s) present in your chosen cohort's variants — not the whole genome. |
| Cancer Gene Census (driver gene list) | COSMIC | **License caution:** free for academic use, but do not bundle/redistribute this file in the repo. The pipeline should fetch it at runtime using the user's own COSMIC credentials, or document that the user must supply their own copy. |
| Pretrained model checkpoint | HuggingFace, Nucleotide Transformer | One-time download (few hundred MB to ~1-2GB for smallest variant). Test that your network allows this download before depending on it mid-pipeline. |
| (Optional, for evaluation) CADD scores | CADD website | Used only to sanity-check that your model's rankings are in a reasonable ballpark vs. an established tool — not reimplemented, just referenced. |

**Data use agreement note:** TCGA/cBioPortal open-access data has a data use agreement even at the public tier. The README should reflect that users are expected to source their own data under these terms rather than the repo redistributing a derived dataset.

---

## 5. Pipeline Architecture

```
Raw MAF (cBioPortal)
        ↓
[Snakemake stage] Parse + clean MAF (dedupe, filter to SNVs)
        ↓
[Snakemake stage] Extract sequence windows (ref + alt) from GRCh38 FASTA
        ↓         (with strand-orientation + ref-allele sanity checks)
[Snakemake stage] Label variants (driver/passenger via COSMIC CGC)
        ↓
[Snakemake stage] Gene-level train/val/test split (saved once, reused everywhere)
        ↓
Baseline model (Phase 2) ──┐
                            ├──→ Evaluation comparison (same test set)
Fine-tuned genomic LM ──────┘
        ↓
Batch scoring CLI → ranked report (CSV/HTML)
        ↓
Docker package + docs
```

---

## 6. Build Phases (work through these in order)

### Phase 0: Environment setup
- Set up conda/mamba environment, pin all dependency versions in `environment.yml` from the first commit.
- Verify HuggingFace model download works on your network before relying on it later.
- **Done when:** environment reproducible on a clean machine/container, all core packages import successfully.

### Phase 1: Data acquisition (small-scale)
- Download one TCGA study's MAF from cBioPortal (public API or single-study download).
- Download only the relevant chromosome(s) of the GRCh38 reference FASTA.
- Download COSMIC CGC gene list (do not commit this file to the repo).
- **Done when:** all three data sources are present locally and load without error.

### Phase 2: Data pipeline + sanity checks (this is the highest-risk phase — do not skip the checks)
- Parse MAF: dedupe variant calls, filter to SNVs only.
- Extract a sequence window (e.g., 512bp) centered on each variant from the reference FASTA.
- **Sanity check (mandatory, automated, not manual):** for every variant, confirm the reference base at that genomic position matches the MAF's listed reference allele. Mismatches indicate a genome-build or coordinate error — do not proceed until this passes for the large majority of variants.
- **Strand check (mandatory):** confirm variant orientation matches gene/transcript strand where applicable; flag or correct reverse-complement mismatches.
- Construct the "alternate" sequence (reference window with ref allele swapped for alt allele at the center position).
- Label each variant as driver/passenger using COSMIC CGC gene membership (single cancer type only — see Edge Cases).
- Perform gene-level train/val/test split (all variants from a gene go entirely into one split — never split within a gene). Save this split as a versioned artifact; reuse it for every later experiment, do not regenerate it per run.
- **Done when:** a clean, labeled dataset of (ref sequence, alt sequence, label) tuples exists, split by gene, with all sanity checks passing and logged.

### Phase 3: Baseline model
- Build a simple baseline (logistic regression or random forest) using hand-crafted features (e.g., GC content, COSMIC CGC membership, conservation score if available).
- Evaluate on the held-out gene-split test set using AUPRC and precision/recall — not accuracy (severe class imbalance expected between drivers and passengers).
- **Done when:** baseline model trains, evaluates, and produces a documented AUPRC/precision-recall number to beat later.

### Phase 4: Fine-tune the genomic language model
- Load smallest Nucleotide Transformer checkpoint, add classification head.
- Fine-tune on (ref, alt) sequence pairs using the same gene-level split as the baseline.
- Log all experiments via MLflow/W&B from the first run.
- Check tokenizer behavior explicitly (k-mer boundary alignment relative to variant position) before trusting results.
- If compute-constrained: freeze most of the model, fine-tune only a small head/adapter layer rather than the full model.
- **Done when:** fine-tuned model trains successfully and produces evaluation metrics on the same test set as the baseline.

### Phase 5: Evaluation
- Compare fine-tuned model vs. baseline on identical test set (AUPRC, precision/recall curves).
- Optional: compare rankings against CADD scores for the same variants as an external sanity check.
- Document results honestly — a modest improvement over baseline is a legitimate, expected outcome given likely small sample size; do not overstate results.
- **Done when:** a documented comparison table/plot exists showing baseline vs. fine-tuned performance.

### Phase 6: Batch scoring CLI
- Build `variantfx score --input cohort.vcf --output report/` (or similar).
- Output a ranked table: mutation, predicted impact, confidence, affected gene, known cancer association (from COSMIC), priority rank.
- Handle malformed real-world VCFs gracefully (multi-allelic sites, missing genotype fields, `chr1` vs `1` naming) — use `pysam` VCF handling, not hand-rolled parsing. Write test cases with deliberately malformed VCFs to confirm graceful failure with clear error messages.
- **Done when:** CLI runs end-to-end on a fresh VCF and produces a correct, readable report, including on at least one deliberately malformed test input (handled gracefully, not a crash).

### Phase 7: Packaging + documentation
- Dockerize the full pipeline; pin all dependency versions inside the image.
- Version model checkpoints explicitly; log which model version produced each report.
- Write README assuming zero context from the author: state the GRCh38 assumption prominently, document data sourcing steps (including COSMIC/TCGA licensing notes), and include a quickstart example.
- **Done when:** a stranger could clone the repo, follow the README, and run the tool on their own cohort without contacting you.

### Phase 8 (optional, time-permitting): Interpretability
- Add attention-weight visualization over the sequence window as a diagnostic overlay.
- Frame this explicitly and honestly in documentation as a diagnostic aid, not a definitive causal explanation — sequence attribution for genomic transformers is an unsettled research area; overclaiming here undermines credibility with informed reviewers.
- **Done when:** visualization renders for a sample prediction and documentation correctly scopes its limitations.

---

## 7. Edge Cases & Pain Points (and how to resolve them)

| # | Issue | Why it matters | Resolution |
|---|---|---|---|
| 1 | Genome build mismatch (GRCh37 vs GRCh38) | Silently produces wrong sequence context with no error — model trains on garbage biology | Automated sanity check: confirm reference base at variant position matches MAF ref allele before training. Liftover if needed. |
| 2 | Severe class imbalance (few true drivers vs. many passengers) | Naive model achieves high accuracy while being useless | Use class-weighted loss; evaluate with AUPRC/precision-recall, not accuracy |
| 3 | Data leakage via gene/patient overlap in train/test split | Inflated, non-generalizing performance | Split by gene (all variants from a gene stay together); split by patient/sample too if applicable |
| 4 | Tokenization boundary misalignment | Variant position may fall mid-token in k-mer tokenizers, weakening signal around the mutation | Explicitly test tokenizer behavior on sample sequences before trusting pipeline output |
| 5 | Compute limits for LM fine-tuning | Full fine-tuning may be infeasible without a GPU | Start with smallest model variant; freeze most layers and fine-tune only a head/adapter if needed |
| 6 | Duplicate variant calls across sequencing pipelines in MAF files | Double-counts some mutations, biases class balance | Deduplicate variants during Phase 2 parsing |
| 7 | Tissue-specific driver gene labels | A gene may be a driver in one cancer type but not another; blanket labeling across multiple cancer types mislabels passengers as drivers | Restrict to one cancer type in v1; use tissue-specific driver annotations if expanding later |
| 8 | Strand orientation mismatches | Sequence may be extracted as reverse-complement by accident for some variants | Build strand check into the same Phase 2 sanity test as the ref-allele check |
| 9 | Malformed real-world VCFs (multi-allelic sites, missing fields, chromosome naming inconsistencies) | Naive parsers crash or silently mis-parse | Use `pysam` VCF handling; write deliberately malformed test VCFs to confirm graceful error handling |
| 10 | Fragile bioinformatics Python environments | pip-only installs of htslib-dependent packages often fail or behave inconsistently | Use conda/mamba from the start; pin versions immediately |
| 11 | COSMIC/TCGA licensing | Redistributing COSMIC's gene list or TCGA-derived data may violate usage terms | Fetch COSMIC data at runtime with user's own credentials; do not bundle in repo; document data use agreements in README |
| 12 | Interpretability overclaiming | Sequence attribution for genomic transformers is an unsettled research area; overclaiming undermines credibility | Frame Phase 8 outputs explicitly as diagnostic aids, not definitive explanations |
| 13 | Scope creep / never reaching packaging phase | Common failure mode: polishing modeling indefinitely, shortchanging docs/packaging | Written "definition of done" (Section 8) locked in before starting; treat Phases 6-7 as non-negotiable |

---

## 8. Definition of Done (v1)

The project is considered complete when:

1. Pipeline runs end-to-end on one TCGA cancer type's cohort, from raw MAF to ranked report.
2. Fine-tuned model is evaluated against the baseline on an identical, gene-level-split, held-out test set, using AUPRC/precision-recall.
3. CLI (`variantfx score`) takes a VCF/MAF as input and outputs a ranked report (mutation, predicted impact, confidence, affected gene, known cancer association, priority rank).
4. Pipeline is Dockerized with pinned dependencies; model checkpoint is versioned.
5. README is complete: states the GRCh38 assumption prominently, documents data sourcing and licensing, includes a working quickstart example.
6. Automated sanity checks (ref-allele match, strand orientation, gene-split leakage) exist as actual test code, not manual one-off checks.

Interpretability (Phase 8) is a stretch goal, not required for v1 completion.

---

## 9. Suggested Repo Structure

```
variantfx/
├── environment.yml
├── Dockerfile
├── Snakefile
├── config/
│   └── config.yaml          # cancer type, genome build, model variant, paths — all config-driven, not hardcoded
├── data/                     # gitignored — user-supplied data lives here
├── src/
│   ├── pipeline/
│   │   ├── parse_maf.py
│   │   ├── extract_sequences.py
│   │   ├── sanity_checks.py  # ref-allele match, strand check
│   │   └── label_variants.py
│   ├── models/
│   │   ├── baseline.py
│   │   └── finetune_lm.py
│   ├── evaluation/
│   │   └── evaluate.py
│   └── cli/
│       └── score.py
├── tests/
│   ├── test_sanity_checks.py
│   ├── test_malformed_vcf.py
│   └── test_split_leakage.py
├── notebooks/                # exploratory only — no logic that matters lives only here
└── README.md
```

---

## 10. Working Style Notes (for the AI assistant building this)

- Work through Section 6's phases strictly in order. Do not skip ahead or combine multiple phases into a single code drop.
- After completing each phase's "done when" checkpoint, stop and present the result for review before continuing.
- If a phase's sanity checks fail, stop and surface the failure clearly rather than proceeding with known-bad data.
- Prefer small, working, ugly code at each phase over incomplete sophisticated code — get the boring version working first, then improve.
- Flag to the human immediately if any Section 2 scope decision needs to change (e.g., if COSMIC access turns out to be harder than expected, or if data volume is too small to train meaningfully) — do not silently improvise a different approach.
