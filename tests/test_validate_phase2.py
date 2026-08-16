import csv

import pytest

from src.pipeline.validate_phase2 import run


def row(**overrides):
    reference = "A" * 256 + "C" + "A" * 255
    alternate = "A" * 256 + "T" + "A" * 255
    result = {
        "Variant_ID": "v1",
        "Reference_Allele": "C",
        "Alternate_Allele": "T",
        "Reference_Sequence": reference,
        "Alternate_Sequence": alternate,
        "Hugo_Symbol": "GENE1",
        "Split": "train",
        "Label_Name": "passenger",
        "Transcript_Strand_Check": "plus",
    }
    result.update(overrides)
    return result


def write_rows(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=rows[0], delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def test_final_validator_accepts_valid_sequence_pairs(tmp_path):
    input_path = tmp_path / "dataset.tsv"
    write_rows(input_path, [row()])

    report = run(input_path, tmp_path / "report.json")

    assert report["unique_variant_ids"] == 1
    assert report["gene_split_leakage_count"] == 0
    assert report["variant_index"] == 256


def test_final_validator_rejects_gene_split_leakage(tmp_path):
    input_path = tmp_path / "dataset.tsv"
    write_rows(
        input_path,
        [row(), row(Variant_ID="v2", Split="test")],
    )

    with pytest.raises(ValueError, match="Gene split leakage"):
        run(input_path, tmp_path / "report.json")
