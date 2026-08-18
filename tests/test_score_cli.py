import csv
from pathlib import Path

import pandas as pd
import pytest

from src.cli.score import ScoringError, file_sha256, parse_vcf, run_score


class FakeScorer:
    model_version = "test-fixture-v1"

    def predict(self, references, alternates):
        return [0.9 if alternate[256] == "T" else 0.4 for alternate in alternates]


def write_reference(directory: Path):
    directory.mkdir()
    sequence = "A" * 1000
    (directory / "chr1.fa").write_text(">chr1\n" + sequence + "\n")


def write_cosmic(path: Path):
    pd.DataFrame(
        [{"Gene Symbol": "TP53", "Tumour Types(Somatic)": "breast", "Role in Cancer": "TSG", "Tier": 1}]
    ).to_csv(path, index=False)


def write_vcf(path: Path, records: list[str], with_samples: bool = False):
    header = [
        "##fileformat=VCFv4.2",
        "##contig=<ID=chr1,length=1000>",
        "##contig=<ID=1,length=1000>",
        '##INFO=<ID=GENE,Number=1,Type=String,Description="Gene symbol">',
    ]
    columns = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO"
    if with_samples:
        header.append('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
        columns += "\tFORMAT\tSAMPLE"
    path.write_text("\n".join([*header, columns, *records]) + "\n")


def test_parse_vcf_expands_multiallelic_snvs_and_normalizes_chromosome(tmp_path):
    vcf = tmp_path / "multi.vcf"
    write_vcf(vcf, ["chr1\t500\t.\tA\tC,T\t.\tPASS\tGENE=TP53"])
    variants, rejected = parse_vcf(vcf)
    assert [(v.chromosome, v.alternate) for v in variants] == [("1", "C"), ("1", "T")]
    assert rejected == []


def test_site_only_vcf_without_genotypes_is_accepted(tmp_path):
    vcf = tmp_path / "site-only.vcf"
    write_vcf(vcf, ["1\t500\t.\tA\tT\t.\tPASS\tGENE=TP53"])
    variants, rejected = parse_vcf(vcf)
    assert len(variants) == 1
    assert rejected == []


def test_non_snv_is_rejected_with_explicit_reason(tmp_path):
    vcf = tmp_path / "indel.vcf"
    write_vcf(vcf, ["1\t500\t.\tA\tAT\t.\tPASS\tGENE=TP53"])
    variants, rejected = parse_vcf(vcf)
    assert variants == []
    assert rejected[0]["reason"] == "non_snv_or_invalid_allele"


def test_malformed_vcf_fails_with_clear_message(tmp_path):
    vcf = tmp_path / "broken.vcf"
    vcf.write_text("this is not a VCF\n")
    with pytest.raises(ScoringError, match="Could not open VCF|Malformed VCF"):
        parse_vcf(vcf)


def test_checkpoint_fingerprint_is_stable_sha256(tmp_path):
    checkpoint = tmp_path / "best_model.pt"
    checkpoint.write_bytes(b"versioned checkpoint fixture")
    assert file_sha256(checkpoint) == "e5a7f63026f358d410e29959f3b0a92cdfd824a11d4a13ed71a2ebb8d787708e"


def test_score_runs_end_to_end_and_writes_ranked_and_rejected_reports(tmp_path):
    reference = tmp_path / "reference"; write_reference(reference)
    cosmic = tmp_path / "cosmic.csv"; write_cosmic(cosmic)
    vcf = tmp_path / "cohort.vcf"
    write_vcf(
        vcf,
        [
            "chr1\t500\t.\tA\tC,T\t.\tPASS\tGENE=TP53",
            "1\t600\t.\tA\tAT\t.\tPASS\tGENE=UNKNOWN",
            "1\t700\t.\tC\tG\t.\tPASS\tGENE=TP53",
        ],
    )
    output = tmp_path / "report"
    report = run_score(vcf, output, reference, cosmic, FakeScorer())
    ranked = pd.read_csv(output / "ranked_variants.csv")
    with (output / "rejected_variants.csv").open() as handle:
        rejected = list(csv.DictReader(handle))

    assert report["scored_variants"] == 2
    assert report["rejected_alleles_or_records"] == 2
    assert report["model_checkpoint_version"] == "test-fixture-v1"
    assert ranked.Mutation.tolist() == ["chr1:500 A>T", "chr1:500 A>C"]
    assert ranked.Priority_Tier.iloc[0] == "top_5_percent"
    assert "Uncalibrated" in ranked.columns[4]
    assert "CGC tier 1" in ranked.Known_Cancer_Association.iloc[0]
    assert {row["reason"] for row in rejected} == {"non_snv_or_invalid_allele", "reference_mismatch:expected_C:observed_A"}
