# genomic-variant-prioritizer

A reproducible pipeline that fine-tunes a genomic language model (Nucleotide Transformer) to predict whether somatic breast cancer mutations are functionally impactful ("driver-like") or not. It outputs a ranked, batch-scored report from a VCF input file.

> **Genome build: GRCh38.** All coordinates, reference sequences, and inputs to this tool assume GRCh38. The source TCGA-BRCA mutation data is distributed in **GRCh37** coordinates and was lifted over to GRCh38 as part of this pipeline (see the Liftover section below). If you supply your own VCF, it must use GRCh38 coordinates.

---

## What this does

Every sequenced cancer genome turns up thousands of somatic mutations, but only a small fraction are biologically meaningful ("driver" mutations that contribute to cancer progression). The rest are inert "passenger" mutations. This tool:

1. Takes a cohort's somatic mutation calls (VCF) as input
2. Extracts DNA sequence context around each variant
3. Uses a fine-tuned genomic language model to predict functional impact
4. Outputs a ranked report a researcher can use to prioritize follow-up

This is a fine-tuning and productionization project, not a from-scratch model architecture project. The emphasis is on doing the full pipeline (data handling, fine-tuning, evaluation, packaging, reproducibility) rigorously and honestly, rather than maximizing a benchmark number.

---

## Scope (v1)

- **Variant types:** SNVs only. Indels are out of scope for v1, since they shift sequence length and complicate windowed extraction.
- **Cohort:** TCGA-BRCA (breast cancer), the `brca_tcga_pan_can_atlas_2018` study via cBioPortal.
- **Labels:** Driver vs. passenger, using COSMIC Cancer Gene Census (CGC) gene membership as a weak, proxy label. This is a real limitation, discussed below.
- **Model:** Nucleotide Transformer v2, 50M parameter variant (`InstaDeepAI/nucleotide-transformer-v2-50m-multi-species`).
- **Genome build:** GRCh38 throughout.

---

## Data sourcing

You need to supply three data sources yourself. None are bundled in this repo (see Licensing below).

**1. TCGA-BRCA mutation data.** Public, open access, via cBioPortal's DataHub. *(Codex: confirm and document the exact working retrieval method here. The original raw S3 archive URL returned HTTP 403 during this project. The working retrieval path used cBioPortal DataHub's Git LFS backend with SHA-256 verification against the official object ID. Document the actual working command and steps.)*

**2. GRCh38 reference FASTA.** Only the chromosomes actually represented in your variant data are needed. Download per chromosome from UCSC or Ensembl, not the whole genome. This repo's dataset spans chromosomes 1 through 22, X, and Y.

**3. COSMIC Cancer Gene Census.** Requires a free academic COSMIC account. Register at cancer.sanger.ac.uk/cosmic, download the Cancer Gene Census CSV export, and place it locally per the licensing note below. This file must never be committed to a fork of this repo.

**Data use agreement note:** TCGA and cBioPortal data carry a data use agreement even at the public tier. You are responsible for sourcing your own copy under those terms. This repo does not redistribute a derived dataset.

---

## GRCh37 to GRCh38 liftover

The TCGA-BRCA MAF data is distributed in GRCh37 coordinates, not GRCh38. This was caught early by an automated build check confirming that the reference base at each variant's position matched the MAF's listed reference allele, rather than being discovered downstream. A mismatch here would silently train the model on the wrong sequence context with no error.

An auditable liftover utility (`scripts/liftover_maf_snvs.py`) handles the conversion. It:
- Uses the official UCSC `hg19ToHg38.over.chain.gz`
- Correctly converts between MAF's 1based and pyliftover's 0based coordinates
- Is scoped to SNVs only, matching this project's v1 scope
- Rejects ambiguous mappings (more than one candidate target location), unmapped coordinates, and mappings to alternate or random contigs explicitly, with reasons, not silently
- Correctly complements alleles for reverse strand chain mappings (455 records in this dataset)
- Preserves rejected records in a separate audit file

**Validation:** 16 unit tests cover this script, including two tests cross checked against an independent source, Ensembl's own coordinate mapping tool, not just internal consistency with the UCSC chain file the script itself uses. This matters because a shared misunderstanding of the chain file format could otherwise pass both the script and a test built on the same source.

**Results on the full dataset:**
- Input: 126,252 original MAF records
- Successfully lifted to GRCh38 SNVs: 113,526
- Non-SNV records excluded: 12,657
- Alternate or random contig mappings rejected: 57
- Unmapped: 12

---

## Data pipeline and sanity checks

For each variant, a 512bp reference sequence window is extracted, with the variant centered at index 256. An "alternate" window is constructed by substituting the alt allele at the center position.

**Mandatory reference allele concordance check.** This is the single most important check in the pipeline. It validates both the genome build and the liftover step itself. For every variant, the extracted reference base is compared against the MAF's listed reference allele.

- Overall concordance: 113,525 out of 113,526 (99.9991%)
- Forward chain subset: 113,070 out of 113,071 (99.9991%)
- Reverse chain subset: 455 out of 455 (100%)
- The single mismatch was investigated and found to be a genuine reference allele change between genome assembly versions at that position, not a pipeline error. It was excluded.

The 100% concordance on the reverse chain subset specifically confirms the liftover's allele complementation logic held up on the full real dataset, not just the synthetic test cases.

**Labeling:** COSMIC CGC gene membership, restricted to this single cancer cohort. Tissue specific driver effects mean a blanket multi cancer type label would be unreliable (see Limitations).

**Class balance:** 8,387 driver labeled variants (7.39%) and 105,138 passenger labeled variants (92.61%) out of 113,525 final labeled variants. This is a significant imbalance, handled throughout via class weighted loss and AUPRC based evaluation rather than accuracy.

**Train, validation, and test split** (`gene-split-v1`, seed 42): split at the gene level. Every variant from a given gene is entirely within one split, never divided across splits, to prevent the model from learning "this gene equals driver" as a shortcut rather than learning real sequence signal.

| Split | Variants | Genes | Drivers | Passengers |
|---|---|---|---|---|
| Train | 91,308 | 14,977 | 6,829 | 84,479 |
| Validation | 10,861 | 1,872 | 784 | 10,077 |
| Test | 11,356 | 1,872 | 774 | 10,582 |

Gene overlap between splits: zero.

**Patient overlap:** 100% of test set variants come from patients who also appear in the training set. This is a structural property of the cohort, not a pipeline flaw. The gene patient graph is one fully connected component spanning all 1,009 patients and 18,721 genes (a handful of frequently mutated genes, such as TP53, link nearly the entire cohort together). Enforcing patient disjoint splits alongside gene disjoint splits would require discarding the large majority of the dataset. This is not treated as a meaningful leakage risk for this model specifically, because the model's only input is DNA sequence context around a variant. It has no patient ID or patient level covariate as a feature, so there is no mechanism by which the model could exploit patient identity.

---

## Baseline model

A class weighted logistic regression trained on hand crafted sequence features, used as the number the fine tuned model needs to beat:

- Reference and alternate window GC content, and the GC content delta
- One hot encoded reference and alternate allele
- Reference trinucleotide context centered on the variant
- Transition indicator (A to G or C to T)
- CpG context indicator

COSMIC CGC membership was deliberately excluded from this feature set. Since CGC membership is the label source itself, including it would make the "baseline" trivially reconstruct the label. This was confirmed via a separate circularity diagnostic run, which unsurprisingly scored AUPRC 1.0 and was excluded from consideration as a real baseline.

| | Validation AUPRC | Test AUPRC |
|---|---|---|
| Prevalence (chance) | 0.0722 | 0.0682 |
| Baseline (logistic regression) | 0.0873 | 0.0753 |

A random forest was also trained for comparison and underperformed logistic regression on both splits, plausibly because the feature set is small and low signal enough that a simpler model generalizes better.

---

## Fine-tuned model

**Architecture:** `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species`, fully fine tuned for 3 epochs with class weighted cross entropy loss. Reference and alternate sequences are encoded separately with a shared encoder. The contextual embedding at the mutation containing token is taken from each, along with their directional difference (alt minus ref), concatenated and passed through a small MLP classification head. This mutation token approach, rather than mean pooling the full window, was chosen because the ref/alt difference affects only one token, and whole window pooling risked diluting that signal.

**Tokenizer check (verified before trusting any results):** the model tokenizes DNA in 6mers, not single bases, meaning the variant position does not necessarily fall on a token boundary. This was verified on 100 real variants. Every ref/alt pair produced identically length (88 token) sequences with matching boundary alignment, and the variant consistently fell at offset 4 within token index 43 (covering sequence positions 252 to 258). No length or boundary mismatch was found between ref/alt pairs.

**Result:**

| | Validation AUPRC | Test AUPRC |
|---|---|---|
| Baseline | 0.0873 | 0.0753 |
| Fine-tuned | 0.0906 | 0.0860 |

**Statistical significance** (paired, class stratified bootstrap, 10,000 iterations, 95% confidence intervals):

- Validation: difference +0.0033, 95% CI [ 0.0071, 0.0128]. This interval crosses zero, so the validation improvement is not statistically confirmed.
- Test: difference +0.0107, 95% CI [0.00066, 0.0226]. This interval excludes zero, but the lower bound sits very close to it. This should be read as a modest, marginally significant improvement, not a strong or unequivocal one.

**Honest summary:** the fine tuned model outperforms the baseline directionally on both splits, with statistical support on the test set specifically, though that support is not strong. The result should not be described as a large or definitively established improvement. It is a real but modest signal, consistent with the label source (gene level CGC membership) being a noisy proxy for true per variant functional impact.

---

## Calibration

The model's raw output probabilities are not well calibrated and should not be read as literal confidence. On the test set:

| Statistic | Value |
|---|---|
| Minimum | 0.303 |
| Mean | 0.413 |
| Median | 0.407 |
| Maximum | 0.967 |
| True prevalence | 6.8% |

Predicted probabilities are compressed into a narrow band well above the true prevalence. This is a known side effect of class weighted training, which corrects ranking behavior without preserving a calibrated probability scale. This is why the CLI (below) does not present raw probabilities as "confidence."

---

## Batch scoring CLI

```
variant-prioritizer score \
  --input cohort.vcf \
  --output report/ \
  --reference-dir data/raw/reference \
  --cosmic data/raw/cosmic/cancer_gene_census.csv \
  --checkpoint path/to/best_model.pt
```

*(Codex: confirm this matches the actual installed command signature and add any flags or options omitted here, such as `--batch-size` or `--device`.)*

**Output (`report/ranked_variants.csv`):** `Priority_Rank`, `Mutation`, `Affected_Gene`, `Predicted_Impact` (labeled `relative_driver_likeness`, not a clinical claim), `Ranking_Score_Uncalibrated`, `Priority_Tier`, `Known_Cancer_Association`.

Given the calibration finding above, raw model output is deliberately not exposed as "confidence." Instead:
- `Ranking_Score_Uncalibrated` is explicitly labeled as suitable for relative ordering only.
- `Priority_Tier` provides cohort relative percentile bands (`top_5_percent`, `top_20_percent`, `remaining`) computed within the submitted cohort. This is a more defensible presentation than treating the compressed 0.30 to 0.97 raw range as literal probability. Note that these tiers become less meaningful on very small input cohorts, since percentile bands need a reasonably sized sample to mean much.

Also produced: `report/rejected_variants.csv` (with explicit rejection reasons) and `report/run_metadata.json` (including the exact SHA256 of the model checkpoint used, so every report is traceable to the specific trained model that produced it).

**Real world VCF handling:** uses `pysam`, not a hand rolled parser. It handles multi allelic sites (expanded into one scored row per alternate allele), missing genotype fields (site only VCFs are accepted, since genotype data isn't required for scoring), and `chr1` versus `1` chromosome naming inconsistencies. Malformed input produces a clear error message, not a crash. This was verified with deliberately malformed test VCFs.

---

## Reproducibility

- **Docker:** a digest pinned image (`Dockerfile`) runs the CLI end to end with pinned dependencies matching `environment.yml`. No data, reference FASTA, COSMIC file, or model checkpoint is bundled in the image. These are mounted by the user at runtime.
- **Verified end to end:** the same approved model checkpoint was run against the same 8 real test variants in both the original Colab environment and the packaged Docker container. Scores matched to within floating point noise (maximum absolute difference: 1.19 times 10 to the negative 7). This confirms the packaged artifact reproduces the validated result, not just that it runs without error.
- **Model checkpoint versioning:** every CLI run computes and logs the SHA256 digest of the checkpoint used, so any report can be traced to the exact model that produced it.
- **Experiment tracking:** all training and evaluation runs are logged via MLflow.

---

## Interpretability

As a diagnostic addition, attention weights from the fine tuned model were extracted and visualized for a sample of test set predictions, examining which regions of the 512bp window the model attends to most strongly relative to the mutation containing token (index 43, covering sequence positions 252 to 258).

Visualizations cover 8 representative examples: 3 correctly classified drivers, 3 correctly classified passengers, and 2 false positives, included deliberately rather than only showing successes. These are available in `reports/phase8/`.

*(Codex: embed 1 to 2 example plots here as images, per the existing Phase 8 output.)*

**Important caveat:** this is included as a diagnostic aid only, not a definitive causal explanation. Sequence attribution for genomic transformer models is a genuinely unsettled area of research. Attention weights show what the model attended to during inference, not necessarily what biologically or causally drove the prediction. Read these as a starting point for qualitative inspection, not validated biological insight.

---

## Model improvement exploration (not completed)

A model and training improvement track was scoped to explore whether performance could be pushed further while holding the dataset, labels, and split fixed. Planned work included scaling to larger Nucleotide Transformer variants (100M and 250M), hyperparameter tuning with early stopping on validation AUPRC, a pooling architecture ablation comparing mutation token pooling, mean pooling, and learned attention pooling, integrating conservation scores (phyloP or phastCons) as an auxiliary feature, and reverse complement sequence augmentation.

The supporting code and unit tests were implemented and validated (`src/models/phase9.py`, `tests/test_phase9.py`), but the experiments were not run to completion due to time constraints. No results from this exploration are reported or used anywhere in this document. All performance numbers above are the original, fully validated results.

A label quality upgrade, replacing gene level COSMIC CGC membership with mutation level annotations such as OncoKB or Cancer Hotspots, is identified as the most promising direction for a future v2. This directly targets the likely ceiling on current performance: a gene level label cannot distinguish a true oncogenic hotspot mutation from a near neutral passenger mutation elsewhere in the same driver gene.

---

## Limitations

- **Label quality:** COSMIC CGC gene membership is a weak, gene level proxy for driver status, not a per variant ground truth. This is likely the dominant ceiling on model performance.
- **Single cancer type:** trained and evaluated on TCGA-BRCA only. Driver effects are often tissue specific, so this model should not be assumed to generalize to other cancer types.
- **SNVs only:** indels are out of scope for v1.
- **Small effective label size:** 8,387 driver labeled variants. This is a deliberate, disclosed scope choice per the project spec, not an oversight.
- **Patient overlap:** see the data pipeline section above. It is structurally near total, but not a meaningful leakage risk for this sequence only architecture specifically.
- **Uncalibrated outputs:** raw model probabilities are compressed and systematically elevated relative to true prevalence. Use `Ranking_Score_Uncalibrated` for relative ordering and `Priority_Tier` for cohort relative context, not as calibrated confidence.
- **Small cohort tier instability:** `Priority_Tier` percentiles are less meaningful on very small input VCFs.
- **No external validation against an established tool:** an optional comparison against CADD scores was scoped but not completed due to time constraints.
- **Interpretability caveat:** see the Interpretability section above. Attention visualization is diagnostic only, not causal explanation.

---

## Repository structure

*(Codex: fill in with the actual current structure via `tree` or equivalent, confirming it matches or documenting deviations from the original spec's suggested layout.)*

## Local setup (Conda)

*(Codex: verify and document the exact, currently working commands: `conda env create -f environment.yml`, activation, and the HuggingFace access check script.)*

## Docker quickstart

*(Codex: verify and document the exact, currently working build and run commands, including the volume mount pattern for data, checkpoint, and COSMIC files.)*

## License

Code is licensed under MIT (see `LICENSE`). This license covers code only. It does not extend to any third party data (TCGA, COSMIC) or the trained model checkpoint, which carry their own separate usage terms as described above.
