import csv
from pathlib import Path

import pytest
from pyliftover import LiftOver

from scripts.liftover_maf_snvs import (
    LIFTOVER_STRAND_COLUMN,
    convert_snv_record,
    process_maf,
)


FIELDNAMES = [
    "NCBI_Build",
    "Chromosome",
    "Start_Position",
    "End_Position",
    "Strand",
    "Variant_Type",
    "Reference_Allele",
    "Tumor_Seq_Allele1",
    "Tumor_Seq_Allele2",
]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRCH37_TO_GRCH38_CHAIN = (
    PROJECT_ROOT / "data/raw/reference/hg19ToHg38.over.chain.gz"
)
requires_local_chain = pytest.mark.skipif(
    not GRCH37_TO_GRCH38_CHAIN.exists(),
    reason="official UCSC hg19ToHg38 chain is a local, gitignored data dependency",
)


class FakeLiftOver:
    def __init__(self, mappings):
        self.mappings = mappings
        self.calls = []

    def convert_coordinate(self, chromosome, position):
        self.calls.append((chromosome, position))
        return self.mappings


def maf_row(**overrides):
    row = {
        "NCBI_Build": "GRCh37",
        "Chromosome": "7",
        "Start_Position": "101",
        "End_Position": "101",
        "Strand": "+",
        "Variant_Type": "SNP",
        "Reference_Allele": "A",
        "Tumor_Seq_Allele1": "A",
        "Tumor_Seq_Allele2": "C",
    }
    row.update(overrides)
    return row


def test_forward_mapping_converts_between_maf_and_pyliftover_coordinates():
    lifter = FakeLiftOver([("chr7", 200, "+", 1)])

    converted, reason, reverse_strand = convert_snv_record(maf_row(), lifter)

    assert lifter.calls == [("chr7", 100)]
    assert reason is None
    assert reverse_strand is False
    assert converted["NCBI_Build"] == "GRCh38"
    assert converted["Chromosome"] == "7"
    assert converted["Start_Position"] == "201"
    assert converted["End_Position"] == "201"
    assert converted["Reference_Allele"] == "A"
    assert converted["Tumor_Seq_Allele1"] == "A"
    assert converted["Tumor_Seq_Allele2"] == "C"
    assert converted["Strand"] == "+"
    assert converted[LIFTOVER_STRAND_COLUMN] == "+"


def test_reverse_mapping_complements_every_allele_and_emits_forward_strand():
    lifter = FakeLiftOver([("chr2", 300, "-", 1)])

    converted, reason, reverse_strand = convert_snv_record(maf_row(), lifter)

    assert reason is None
    assert reverse_strand is True
    assert converted["Chromosome"] == "2"
    assert converted["Start_Position"] == "301"
    assert converted["End_Position"] == "301"
    assert converted["Reference_Allele"] == "T"
    assert converted["Tumor_Seq_Allele1"] == "T"
    assert converted["Tumor_Seq_Allele2"] == "G"
    assert converted["Strand"] == "+"
    assert converted[LIFTOVER_STRAND_COLUMN] == "-"


@pytest.mark.parametrize(
    ("query_strand", "query_start", "query_end", "expected_strand"),
    [
        ("+", 300, 301, "+"),
        ("-", 699, 700, "-"),
    ],
)
def test_real_pyliftover_chain_semantics(
    tmp_path, query_strand, query_start, query_end, expected_strand
):
    """Exercise pyliftover itself with a one-base synthetic UCSC chain."""
    chain_path = tmp_path / "synthetic.chain"
    chain_path.write_text(
        "chain 1 chr7 1000 + 100 101 "
        f"chr2 1000 {query_strand} {query_start} {query_end} 1\n"
        "1\n\n"
    )

    mappings = LiftOver(str(chain_path)).convert_coordinate("chr7", 100)

    assert mappings == [("chr2", 300, expected_strand, 1)]


@requires_local_chain
def test_real_forward_coordinate_matches_independent_ensembl_mapping():
    # Independent result retrieved from Ensembl's assembly-mapping REST service:
    # https://grch37.rest.ensembl.org/map/human/GRCh37/11:66082467..66082467:1/GRCh38
    lifter = LiftOver(str(GRCH37_TO_GRCH38_CHAIN))
    source = maf_row(
        Chromosome="11",
        Start_Position="66082467",
        End_Position="66082467",
        Reference_Allele="C",
        Tumor_Seq_Allele1="C",
        Tumor_Seq_Allele2="T",
    )

    converted, reason, reverse_strand = convert_snv_record(source, lifter)

    assert reason is None
    assert reverse_strand is False
    assert converted["Chromosome"] == "11"
    assert converted["Start_Position"] == "66314996"
    assert converted["End_Position"] == "66314996"
    assert converted["Reference_Allele"] == "C"
    assert converted["Tumor_Seq_Allele2"] == "T"
    assert converted[LIFTOVER_STRAND_COLUMN] == "+"


@requires_local_chain
def test_real_reverse_coordinate_matches_independent_ensembl_mapping():
    # Independent result retrieved from Ensembl's assembly-mapping REST service:
    # https://grch37.rest.ensembl.org/map/human/GRCh37/11:51516077..51516077:1/GRCh38
    lifter = LiftOver(str(GRCH37_TO_GRCH38_CHAIN))
    source = maf_row(
        Chromosome="11",
        Start_Position="51516077",
        End_Position="51516077",
        Reference_Allele="G",
        Tumor_Seq_Allele1="G",
        Tumor_Seq_Allele2="C",
    )

    converted, reason, reverse_strand = convert_snv_record(source, lifter)

    assert reason is None
    assert reverse_strand is True
    assert converted["Chromosome"] == "11"
    assert converted["Start_Position"] == "54603203"
    assert converted["End_Position"] == "54603203"
    assert converted["Reference_Allele"] == "C"
    assert converted["Tumor_Seq_Allele1"] == "C"
    assert converted["Tumor_Seq_Allele2"] == "G"
    assert converted["Strand"] == "+"
    assert converted[LIFTOVER_STRAND_COLUMN] == "-"


@pytest.mark.parametrize(
    ("overrides", "mappings", "expected_reason"),
    [
        ({"NCBI_Build": "GRCh38"}, [], "unexpected_source_build"),
        ({"Variant_Type": "DEL"}, [], "not_snv"),
        ({"Start_Position": "abc"}, [], "invalid_position"),
        ({"Start_Position": "0", "End_Position": "0"}, [], "invalid_position"),
        ({"End_Position": "102"}, [], "snv_interval_not_one_base"),
        ({}, [], "unmapped"),
        (
            {},
            [("chr7", 200, "+", 1), ("chr7", 201, "+", 2)],
            "multiple_mappings",
        ),
        ({}, [("chr7", 200, "?", 1)], "invalid_target_strand"),
        (
            {},
            [("chr7_KI270803v1_alt", 200, "+", 1)],
            "non_primary_target_contig",
        ),
    ],
)
def test_rejection_reasons_are_explicit(overrides, mappings, expected_reason):
    converted, reason, reverse_strand = convert_snv_record(
        maf_row(**overrides), FakeLiftOver(mappings)
    )

    assert converted is None
    assert reason == expected_reason
    assert reverse_strand is False


def write_maf(path: Path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=FIELDNAMES, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def test_process_maf_accounts_for_every_input_record_once(tmp_path):
    input_path = tmp_path / "input.maf"
    output_path = tmp_path / "output.maf"
    rejected_path = tmp_path / "rejected.tsv"
    rows = [
        maf_row(),
        maf_row(Variant_Type="DEL"),
        maf_row(Start_Position="bad"),
        maf_row(NCBI_Build="GRCh38"),
    ]
    write_maf(input_path, rows)

    counts = process_maf(
        input_path,
        output_path,
        rejected_path,
        FakeLiftOver([("chr7", 200, "+", 1)]),
    )

    terminal_outcomes = (
        counts["lifted"]
        + counts["not_snv"]
        + counts["invalid_position"]
        + counts["unexpected_source_build"]
    )
    assert counts["input"] == len(rows) == terminal_outcomes
    assert sum(1 for _ in csv.DictReader(output_path.open(), delimiter="\t")) == 1
    rejected_rows = list(csv.DictReader(rejected_path.open(), delimiter="\t"))
    assert len(rejected_rows) == 3
    assert {row["Liftover_Rejection_Reason"] for row in rejected_rows} == {
        "not_snv",
        "invalid_position",
        "unexpected_source_build",
    }
