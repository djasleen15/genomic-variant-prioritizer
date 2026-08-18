# Genomic Variant Prioritizer

Rank somatic single-nucleotide variants (SNVs) by relative driver-like sequence signal using a fine-tuned 50M-parameter Nucleotide Transformer.

> **Genome-build requirement: all CLI inputs and reference FASTA files must use GRCh38 coordinates.** The TCGA-BRCA mutation source used to develop this project was GRCh37 and was explicitly lifted to GRCh38 before sequence extraction. The pipeline then required reference-allele concordance, including 100% concordance for all 455 reverse-chain records, to validate the coordinate and strand conversion.

The command-line tool accepts a VCF, extracts 512-base reference and alternate windows, scores each valid SNV with the approved fine-tuned model, adds Cancer Gene Census annotations from a user-supplied COSMIC file, and writes a ranked CSV. Raw model outputs are deliberately named `Ranking_Score_Uncalibrated`: they support ranking, not calibrated clinical probabilities.

This is a research prioritization tool, not a clinical diagnostic or medical decision system.

## What is and is not included

The repository contains the pipeline, model code, tests, reports, Docker packaging, and MIT-licensed source code. It does **not** contain:

- TCGA/cBioPortal mutation data or derived patient-level datasets;
- COSMIC Cancer Gene Census data;
- GRCh38 FASTA files or liftover chains; or
- the trained `best_model.pt` checkpoint.

Users must obtain and mount those artifacts under their applicable terms. The MIT license applies to the code only, not to any third-party data or model artifacts.

## Requirements

- Git
- Conda or Mamba for a local installation, or Docker for a containerized run
- A GRCh38 VCF containing SNVs
- Per-chromosome GRCh38 FASTA files such as `chr1.fa` and `chr22.fa`
- A COSMIC Cancer Gene Census CSV obtained through the user's own account
- A compatible Phase 4 `best_model.pt` checkpoint

The first model run also needs network access to download `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species` from Hugging Face. Hugging Face caches it for later runs.

## Quickstart: local Conda environment

The following sequence starts from a clean directory. Replace the four paths under “User-supplied files” with real files before running the final command.

```bash
git clone https://github.com/djasleen15/genomic-variant-prioritizer.git
cd genomic-variant-prioritizer

conda env create -f environment.yml
conda activate variantfx
python -m pip install --no-deps -e .

# User-supplied files:
INPUT_VCF=/absolute/path/to/cohort.grch38.vcf
REFERENCE_DIR=/absolute/path/to/grch38-per-chromosome-fasta
COSMIC_CGC=/absolute/path/to/cancer_gene_census.csv
CHECKPOINT=/absolute/path/to/best_model.pt

variant-prioritizer score \
  --input "$INPUT_VCF" \
  --output report \
  --reference-dir "$REFERENCE_DIR" \
  --cosmic "$COSMIC_CGC" \
  --checkpoint "$CHECKPOINT"

column -s, -t < report/ranked_variants.csv | less -S
cat report/run_metadata.json
```

Use `--device cuda` only in an environment with a compatible CUDA-enabled PyTorch installation. The pinned default environment is CPU-compatible.

### Input conventions

- Input must be VCF/BCF readable by `pysam`; the CLI does not hand-parse VCF text.
- Coordinates and alleles must correspond to GRCh38.
- Chromosomes `1`–`22`, `X`, and `Y` are supported with or without a `chr` prefix.
- Multi-allelic SNV records are expanded and scored allele-by-allele.
- Genotype/sample columns are optional; site-only VCFs are accepted.
- Non-SNV alleles, unsupported contigs, reference mismatches, and unusable sequence windows are written to `rejected_variants.csv`.
- An optional `GENE` INFO field supplies the affected-gene label and enables COSMIC matching. Without it, the output reports `not_provided`.

The reference directory should contain one uncompressed FASTA per chromosome used by the VCF. Filenames may be `chr22.fa` or `22.fa`; `pyfaidx` creates an index when required.

The COSMIC CSV must contain these columns:

```text
Gene Symbol,Tumour Types(Somatic),Role in Cancer,Tier
```

## Docker

Build the pinned image:

```bash
docker build -t genomic-variant-prioritizer:0.1.0 .
```

Place or link all user-supplied inputs under one host directory. The output subdirectory must be writable:

```text
run/
├── cohort.grch38.vcf
├── cancer_gene_census.csv
├── best_model.pt
└── reference/
    ├── chr1.fa
    └── ...
```

Run the container without copying any private or large artifacts into the image:

```bash
mkdir -p run/report

docker run --rm \
  -v "$PWD/run:/work" \
  genomic-variant-prioritizer:0.1.0 \
  score \
  --input /work/cohort.grch38.vcf \
  --output /work/report \
  --reference-dir /work/reference \
  --cosmic /work/cancer_gene_census.csv \
  --checkpoint /work/best_model.pt \
  --device cpu
```

The image contains code and pinned dependencies only. `.dockerignore` excludes data, reports, checkpoints, MLflow runs, and Git history from the build context.

## Outputs and checkpoint provenance

Each scoring run creates:

- `ranked_variants.csv`: ranked accepted SNVs;
- `rejected_variants.csv`: rejected records or alleles with explicit reasons; and
- `run_metadata.json`: model, checkpoint version, counts, caveats, and output paths.

Every real CLI run computes the complete SHA-256 digest of the supplied checkpoint and records it as:

```json
{
  "model_checkpoint_version": "phase4-best-model-sha256:<64-hex-character digest>"
}
```

This makes a report traceable even if multiple files are named `best_model.pt`. Checkpoints (`*.pt`, `*.ckpt`, `*.bin`, and `checkpoints/`) are ignored by Git.

### Obtaining or retraining the checkpoint

No trained checkpoint is distributed from this repository. To reproduce it:

1. Obtain the source data, GRCh38 references, liftover chain, and COSMIC CGC as described below.
2. Configure paths in `config/config.yaml`.
3. Run the Snakemake Phase 2 pipeline to create the versioned `gene-split-v1` dataset.
4. Open `notebooks/phase4_colab.ipynb` on a GPU runtime, upload the labeled split dataset as instructed, and run the notebook.
5. Copy the resulting `best_model.pt` from the Phase 4 artifacts directory to private storage.

The checkpoint architecture must match `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species` and the paired-sequence classifier in `src/models/finetune_lm.py`.

## Data sourcing and licensing

### TCGA-BRCA somatic mutations

Development used the TCGA Breast Invasive Carcinoma cohort obtained through cBioPortal. Download a single study from the [cBioPortal data downloads page](https://www.cbioportal.org/datasets) or its study page rather than redistributing it through this repository. The source mutation coordinates used here were GRCh37 and required the tested `scripts/liftover_maf_snvs.py` GRCh37→GRCh38 conversion before Phase 2.

TCGA open-access data remains subject to the applicable [NCI Genomic Data Commons data policies](https://gdc.cancer.gov/access-data/data-access-processes-and-tools). Users are responsible for confirming and following current data-use requirements. Do not commit raw or derived cohort data.

### GRCh38 reference FASTA

Download only the chromosomes represented in the cohort from the [UCSC hg38 per-chromosome directory](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/), for example `chr22.fa.gz`, then decompress it into the configured reference directory. Do not mix hg19/GRCh37 and hg38/GRCh38 references.

### COSMIC Cancer Gene Census

Obtain the Cancer Gene Census through the [COSMIC download portal](https://cancer.sanger.ac.uk/cosmic/download) using your own eligible account. COSMIC licensing does not permit this project to bundle or redistribute the downloaded CGC file. Keep it outside version control and pass its path with `--cosmic`.

## Reproducible training pipeline

The Snakemake workflow performs the Phase 2 data preparation:

```text
GRCh37 TCGA-BRCA MAF
  → tested GRCh37→GRCh38 SNV liftover
  → parse, deduplicate, and filter SNVs
  → extract 512 bp GRCh38 reference/alternate windows
  → enforce reference-allele concordance
  → apply COSMIC-CGC proxy labels
  → create and validate gene-split-v1
```

Run it after configuring private inputs in `config/config.yaml`:

```bash
conda activate variantfx
snakemake --cores 1
python -m src.pipeline.validate_phase2 \
  --dataset data/processed/phase2/labeled_split_dataset.tsv \
  --report data/processed/phase2/phase2_validation.json
```

Material model and pipeline logic lives under `src/`; the notebook only orchestrates GPU training.

## Model and evaluation results

The approved model uses full fine-tuning of the 50M Nucleotide Transformer with a shared encoder over paired 512-base reference and alternate windows. A class-weighted loss addresses label imbalance. Evaluation uses the identical held-out `gene-split-v1` rows for both models and AUPRC rather than accuracy.

| Split | Sequence-only logistic baseline | Fine-tuned model | Paired difference (95% bootstrap CI) |
|---|---:|---:|---:|
| Validation | 0.0873 | 0.0906 | +0.0033 [-0.0071, 0.0128] |
| Test | 0.0753 | 0.0860 | +0.0107 [0.00066, 0.02263] |

The test improvement is statistically supported but modest and marginal: its confidence interval excludes zero, while the lower bound lies very close to zero. The validation interval crosses zero, so improvement was not statistically confirmed on validation. This should not be summarized as strong or consistently established superiority.

The detailed bootstrap comparison, precision–recall curves, and calibration histogram are in `reports/phase5/`.

## Limitations

- **Weak proxy labels:** COSMIC CGC gene membership labels every mutation in a CGC gene as driver-like. It is not variant-level biological ground truth.
- **One cancer type:** development and evaluation used TCGA-BRCA only. Tissue-specific effects in other cancers are not captured.
- **SNVs only:** indels, structural variants, non-primary contigs, and mitochondrial variants are outside v1 scope.
- **Small positive class:** the deliberately scoped dataset contains 8,387 driver-labeled variants among 113,525 total variants. Statistical power remains limited despite the total row count.
- **Gene-level split, patient overlap:** `gene-split-v1` has zero gene overlap between train, validation, and test. The gene–patient graph was one connected component across all 1,009 patients, making simultaneous patient-disjoint and gene-disjoint splitting impractical without discarding most data. Consequently, 100% (11,356/11,356) of test variants come from patients also represented in train. The architecture receives only local sequence pairs—no sample identifier or patient feature—so this is not direct patient-identity leakage, but it remains a cohort-dependence limitation.
- **Uncalibrated scores:** class weighting improved ranking but elevated and compressed raw outputs. On test, probabilities ranged from 0.3029 to 0.9670 with mean 0.4127 and median 0.4075, despite 6.82% positive prevalence. Treat `Ranking_Score_Uncalibrated` only as a relative ranking score.
- **Cohort-relative tiers:** `Priority_Tier` is based on rank percentiles within the submitted VCF. Tiers are unstable and less meaningful for very small cohorts; for an eight-variant input, “top 5%” effectively means one row.
- **Sequence-only context:** the model does not use expression, chromatin, conservation, protein consequence, clonality, or clinical evidence. A high rank is not proof of pathogenicity.
- **External comparison omitted:** CADD comparison was optional and was not performed because matching local CADD scores were unavailable.

## Repository layout

```text
.
├── Dockerfile                 # pinned container packaging
├── environment.yml            # reproducible local Conda environment
├── requirements-docker.txt    # pinned scoring-runtime subset of the Conda environment
├── pyproject.toml             # installable CLI entry point
├── Snakefile                  # Phase 2 data-preparation DAG
├── config/config.yaml         # build, model, and private input paths
├── scripts/                   # liftover and access utilities
├── src/
│   ├── pipeline/              # parsing, sequence extraction, labels, split, validation
│   ├── models/                # baseline, fine-tuning, and prediction
│   ├── evaluation/            # paired bootstrap and diagnostics
│   └── cli/                   # VCF scoring command
├── tests/                     # liftover, sanity, leakage, model, evaluation, CLI tests
├── notebooks/                 # orchestration only; no unique production logic
└── reports/                   # committed aggregate evaluation reports and plots
```

This sensibly extends the original suggested layout with dedicated `scripts/`, `evaluation/`, and aggregate `reports/` directories. User data, checkpoints, and run-specific outputs remain ignored.

## Tests

```bash
conda activate variantfx
python -m pytest -q
```

The suite covers liftover coordinate and strand logic, reference-allele concordance, label and split integrity, tokenizer/model behavior, bootstrap evaluation, malformed VCF handling, multi-allelic expansion, chromosome normalization, and end-to-end report generation.

## License

Source code is licensed under the [MIT License](LICENSE). That license does not grant rights to TCGA/cBioPortal data, COSMIC content, genome references, pretrained model files, or trained checkpoints. Consult each provider's current terms.
