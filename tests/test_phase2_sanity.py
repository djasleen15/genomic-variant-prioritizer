import csv

from src.pipeline.parse_maf import clean_rows
from src.pipeline.sanity_checks import run as run_concordance
from src.pipeline.extract_sequences import (
    VARIANT_INDEX,
    ChromosomeReferences,
    extract_window,
    transcript_strand_check,
)


def variant_row(**overrides):
    row = {
        "Hugo_Symbol": "GENE1",
        "NCBI_Build": "GRCh38",
        "Chromosome": "1",
        "Start_Position": "2",
        "End_Position": "2",
        "Variant_Type": "SNP",
        "Reference_Allele": "C",
        "Tumor_Seq_Allele2": "T",
        "Tumor_Sample_Barcode": "SAMPLE1",
        "Liftover_Strand": "+",
    }
    row.update(overrides)
    return row


def test_clean_rows_deduplicates_by_sample_position_and_alleles():
    retained, rejected, counts = clean_rows(
        [variant_row(), variant_row(), variant_row(Tumor_Sample_Barcode="SAMPLE2")]
    )

    assert len(retained) == 2
    assert len(rejected) == 1
    assert rejected[0][1] == "duplicate_variant_call"
    assert counts == {"input": 3, "retained": 2, "duplicate_variant_call": 1}


def test_clean_rows_reverifies_true_snv_alleles():
    retained, rejected, counts = clean_rows(
        [
            variant_row(Reference_Allele="CC"),
            variant_row(Tumor_Seq_Allele2="N"),
            variant_row(Tumor_Seq_Allele2="C"),
            variant_row(Start_Position="2", End_Position="3"),
        ]
    )

    assert retained == []
    assert [reason for _, reason in rejected] == [
        "invalid_reference_allele",
        "invalid_alternate_allele",
        "reference_equals_alternate",
        "snv_interval_not_one_base",
    ]
    assert counts["input"] == 4


def test_reference_concordance_reports_forward_and_reverse_separately(tmp_path):
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    (reference_dir / "chr1.fa").write_text(">chr1\nACGT\n")
    input_path = tmp_path / "variants.tsv"
    rows = [
        variant_row(),
        variant_row(
            Start_Position="3",
            End_Position="3",
            Reference_Allele="G",
            Tumor_Seq_Allele2="A",
            Tumor_Sample_Barcode="SAMPLE2",
            Liftover_Strand="-",
        ),
        variant_row(
            Start_Position="4",
            End_Position="4",
            Reference_Allele="A",
            Tumor_Seq_Allele2="C",
            Tumor_Sample_Barcode="SAMPLE3",
        ),
    ]
    with input_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=rows[0], delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    report = run_concordance(
        input_path,
        reference_dir,
        tmp_path / "concordant.tsv",
        tmp_path / "mismatches.tsv",
        tmp_path / "report.json",
    )

    assert report["overall"] == {
        "total": 3,
        "matched": 2,
        "mismatched": 1,
        "rate": 2 / 3,
    }
    assert report["forward"]["rate"] == 0.5
    assert report["reverse"]["rate"] == 1.0
    mismatches = list(
        csv.DictReader((tmp_path / "mismatches.tsv").open(), delimiter="\t")
    )
    assert len(mismatches) == 1
    assert mismatches[0]["Observed_Reference_Base"] == "T"
    concordant = list(
        csv.DictReader((tmp_path / "concordant.tsv").open(), delimiter="\t")
    )
    assert len(concordant) == 2


def test_transcript_strand_check_uses_hgvsc_alleles():
    assert transcript_strand_check(variant_row(HGVSc="ENST1:c.10C>T")) == "plus"
    assert transcript_strand_check(variant_row(HGVSc="ENST1:c.10G>A")) == "minus"
    assert transcript_strand_check(variant_row(HGVSc="")) == "not_applicable"
    assert transcript_strand_check(variant_row(HGVSc="ENST1:c.10A>C")) == "mismatch"


def test_extract_window_places_variant_at_fixed_center(tmp_path):
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    sequence = "A" * 256 + "C" + "G" * 255
    (reference_dir / "chr1.fa").write_text(f">chr1\n{sequence}\n")
    references = ChromosomeReferences(reference_dir)
    row = variant_row(Start_Position="257", End_Position="257")

    reference, alternate = extract_window(row, references)
    references.close()

    assert len(reference) == len(alternate) == 512
    assert reference[VARIANT_INDEX] == "C"
    assert alternate[VARIANT_INDEX] == "T"
    assert reference[:VARIANT_INDEX] == alternate[:VARIANT_INDEX]
    assert reference[VARIANT_INDEX + 1 :] == alternate[VARIANT_INDEX + 1 :]
