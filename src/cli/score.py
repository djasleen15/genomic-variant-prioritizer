"""Batch-score VCF SNVs with the approved fine-tuned model."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd
import pysam
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.models.finetune_lm import MODEL_NAME, PairCollator, PairedSequenceClassifier, VariantPairDataset
from src.pipeline.extract_sequences import VARIANT_INDEX, WINDOW_SIZE
from src.pipeline.sanity_checks import ChromosomeReferences


PRIMARY_CHROMOSOMES = {str(number) for number in range(1, 23)} | {"X", "Y"}
OUTPUT_COLUMNS = [
    "Priority_Rank", "Mutation", "Affected_Gene", "Predicted_Impact",
    "Ranking_Score_Uncalibrated", "Priority_Tier", "Known_Cancer_Association",
]


class ScoringError(ValueError):
    """User-facing VCF or scoring validation error."""


class SequenceScorer(Protocol):
    def predict(self, references: list[str], alternates: list[str]) -> list[float]: ...


@dataclass
class Variant:
    chromosome: str
    position: int
    reference: str
    alternate: str
    gene: str
    reference_sequence: str = ""
    alternate_sequence: str = ""

    @property
    def mutation(self) -> str:
        return f"chr{self.chromosome}:{self.position} {self.reference}>{self.alternate}"


def normalize_chromosome(value: str) -> str:
    chromosome = value.removeprefix("chr").removeprefix("CHR")
    chromosome = chromosome.upper() if chromosome.upper() in {"X", "Y"} else chromosome
    if chromosome not in PRIMARY_CHROMOSOMES:
        raise ScoringError(f"unsupported_chromosome:{value}")
    return chromosome


def record_gene(record: pysam.VariantRecord) -> str:
    if "GENE" not in record.header.info:
        return ""
    value = record.info.get("GENE", "")
    if isinstance(value, tuple):
        return ",".join(str(item) for item in value)
    return str(value or "")


def parse_vcf(input_path: Path) -> tuple[list[Variant], list[dict[str, str]]]:
    variants, rejected = [], []
    try:
        handle = pysam.VariantFile(str(input_path))
    except (OSError, ValueError) as error:
        raise ScoringError(f"Could not open VCF {input_path}: {error}") from error
    try:
        for record in handle:
            try:
                chromosome = normalize_chromosome(record.contig)
            except ScoringError as error:
                rejected.append({"record": str(record).strip(), "reason": str(error)})
                continue
            if not record.alts:
                rejected.append({"record": str(record).strip(), "reason": "missing_alternate_allele"})
                continue
            for alternate in record.alts:
                identity = f"{record.contig}:{record.pos}:{record.ref}>{alternate}"
                if len(record.ref) != 1 or len(alternate) != 1 or record.ref.upper() not in "ACGT" or alternate.upper() not in "ACGT":
                    rejected.append({"record": identity, "reason": "non_snv_or_invalid_allele"})
                    continue
                variants.append(Variant(chromosome, record.pos, record.ref.upper(), alternate.upper(), record_gene(record)))
    except (OSError, ValueError) as error:
        raise ScoringError(f"Malformed VCF record in {input_path}: {error}") from error
    finally:
        handle.close()
    return variants, rejected


def add_sequences(variants: list[Variant], reference_dir: Path) -> tuple[list[Variant], list[dict[str, str]]]:
    references = ChromosomeReferences(reference_dir)
    retained, rejected = [], []
    try:
        for variant in variants:
            start = variant.position - 1 - VARIANT_INDEX
            end = start + WINDOW_SIZE
            try:
                if start < 0:
                    raise ScoringError("window_crosses_chromosome_start")
                sequence = references.sequence(variant.chromosome, start, end)
                if len(sequence) != WINDOW_SIZE:
                    raise ScoringError("window_crosses_chromosome_end")
                if sequence[VARIANT_INDEX] != variant.reference:
                    raise ScoringError(f"reference_mismatch:expected_{variant.reference}:observed_{sequence[VARIANT_INDEX]}")
            except (FileNotFoundError, KeyError, ScoringError) as error:
                rejected.append({"record": variant.mutation, "reason": str(error)})
                continue
            variant.reference_sequence = sequence
            variant.alternate_sequence = sequence[:VARIANT_INDEX] + variant.alternate + sequence[VARIANT_INDEX + 1 :]
            retained.append(variant)
    finally:
        references.close()
    return retained, rejected


class FineTunedScorer:
    def __init__(self, checkpoint_path: Path, batch_size: int = 32, device: str | None = None):
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = file_sha256(checkpoint_path)
        self.model_version = f"phase4-best-model-sha256:{self.checkpoint_sha256}"
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        pretrained = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
        self.model = PairedSequenceClassifier(pretrained.base_model, pretrained.config.hidden_size)
        saved = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(saved["model_state_dict"]); self.model.to(self.device); self.model.eval()

    @torch.no_grad()
    def predict(self, references: list[str], alternates: list[str]) -> list[float]:
        frame = pd.DataFrame({"Reference_Sequence": references, "Alternate_Sequence": alternates, "Label": [0] * len(references)})
        loader = DataLoader(VariantPairDataset(frame), batch_size=self.batch_size, shuffle=False, collate_fn=PairCollator(self.tokenizer))
        probabilities = []
        for batch in loader:
            batch = {key: value.to(self.device) for key, value in batch.items()}
            probabilities.extend(torch.softmax(self.model(batch), dim=-1)[:, 1].cpu().tolist())
        return probabilities


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a stable checkpoint fingerprint without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cosmic_associations(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"Gene Symbol", "Tumour Types(Somatic)", "Role in Cancer", "Tier"}
    if not required.issubset(frame.columns):
        raise ScoringError(f"COSMIC CGC file is missing columns: {sorted(required - set(frame.columns))}")
    associations = {}
    for _, row in frame.iterrows():
        details = [f"CGC tier {row['Tier']}"]
        if row["Role in Cancer"]: details.append(str(row["Role in Cancer"]))
        if row["Tumour Types(Somatic)"]: details.append(str(row["Tumour Types(Somatic)"]))
        associations[str(row["Gene Symbol"]).upper()] = "; ".join(details)
    return associations


def priority_tier(rank: int, total: int) -> str:
    percentile = 0.0 if total == 1 else (rank - 1) / (total - 1)
    if percentile <= 0.05: return "top_5_percent"
    if percentile <= 0.20: return "top_20_percent"
    return "remaining"


def build_ranked_rows(variants: list[Variant], scores: list[float], associations: dict[str, str]) -> list[dict]:
    if len(variants) != len(scores):
        raise ScoringError("Model returned a different number of scores than input variants")
    ranked = sorted(zip(variants, scores), key=lambda item: item[1], reverse=True)
    rows = []
    for rank, (variant, score) in enumerate(ranked, start=1):
        gene_associations = [associations[gene.strip().upper()] for gene in variant.gene.split(",") if gene.strip().upper() in associations]
        rows.append({
            "Priority_Rank": rank, "Mutation": variant.mutation,
            "Affected_Gene": variant.gene or "not_provided",
            "Predicted_Impact": "relative_driver_likeness",
            "Ranking_Score_Uncalibrated": score,
            "Priority_Tier": priority_tier(rank, len(ranked)),
            "Known_Cancer_Association": " | ".join(gene_associations) if gene_associations else "none_in_supplied_CGC",
        })
    return rows


def write_rejections(path: Path, rejected: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["record", "reason"]); writer.writeheader(); writer.writerows(rejected)


def run_score(input_path: Path, output_dir: Path, reference_dir: Path, cosmic_path: Path, scorer: SequenceScorer) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    variants, rejected = parse_vcf(input_path)
    variants, sequence_rejections = add_sequences(variants, reference_dir); rejected.extend(sequence_rejections)
    write_rejections(output_dir / "rejected_variants.csv", rejected)
    if not variants:
        raise ScoringError(f"No scoreable SNVs remained; see {output_dir / 'rejected_variants.csv'}")
    scores = scorer.predict([v.reference_sequence for v in variants], [v.alternate_sequence for v in variants])
    rows = build_ranked_rows(variants, scores, load_cosmic_associations(cosmic_path))
    pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(output_dir / "ranked_variants.csv", index=False)
    report = {
        "input_vcf": str(input_path), "model": MODEL_NAME, "model_usage": "fine_tuned_model_only",
        "model_checkpoint_version": getattr(scorer, "model_version", "unversioned"),
        "scored_variants": len(rows), "rejected_alleles_or_records": len(rejected),
        "ranking_score_caveat": "Ranking_Score_Uncalibrated is suitable for ordering variants, not as calibrated confidence. Priority_Tier is cohort-relative.",
        "outputs": {"ranked": str(output_dir / "ranked_variants.csv"), "rejected": str(output_dir / "rejected_variants.csv")},
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
