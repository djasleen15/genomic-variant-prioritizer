import pandas as pd
import pytest
import torch

from src.models.phase9 import (
    ConservationFeatureSpec,
    enforce_fixed_split,
    fit_conservation_features,
    reverse_complement,
    reverse_complement_training_rows,
    split_assignment_fingerprint,
    transform_conservation_features,
    write_experiment_report,
)


def split_frame():
    return pd.DataFrame(
        {
            "Chromosome": ["1", "1", "2"],
            "Position": [10, 20, 30],
            "Reference_Allele": ["A", "C", "G"],
            "Alternate_Allele": ["T", "G", "A"],
            "Gene": ["TRAIN", "VALID", "TEST"],
            "Split": ["train", "validation", "test"],
            "Reference_Sequence": ["AACG", "CCAA", "GGTT"],
            "Alternate_Sequence": ["AATG", "CGAA", "GATT"],
            "Label": [0, 1, 0],
        }
    )


def test_reverse_complement_handles_case_and_ambiguous_base():
    assert reverse_complement("AaCGN") == "NCGtT"


def test_reverse_complement_rejects_unsupported_bases():
    with pytest.raises(ValueError, match="unsupported"):
        reverse_complement("ACGU")


def test_reverse_complement_augmentation_only_duplicates_training_rows():
    result = reverse_complement_training_rows(split_frame())
    assert len(result) == 4
    assert result.Split.value_counts().to_dict() == {"train": 2, "validation": 1, "test": 1}
    added = result.loc[result.Reverse_Complement_Augmented]
    assert added.Reference_Sequence.tolist() == ["CGTT"]
    assert added.Alternate_Sequence.tolist() == ["CATT"]


def test_split_fingerprint_is_independent_of_row_order():
    frame = split_frame()
    assert split_assignment_fingerprint(frame) == split_assignment_fingerprint(
        frame.sample(frac=1, random_state=7)
    )


def test_fixed_split_accepts_gene_split_v1_and_matching_manifest():
    frame = split_frame()
    fingerprint = split_assignment_fingerprint(frame)
    assert enforce_fixed_split(frame, expected_fingerprint=fingerprint) == fingerprint


def test_fixed_split_rejects_assignment_changes():
    frame = split_frame()
    fingerprint = split_assignment_fingerprint(frame)
    changed = frame.copy()
    changed.loc[0, "Split"] = "validation"
    with pytest.raises(ValueError, match="Expected splits|leakage|fingerprint"):
        enforce_fixed_split(changed, expected_fingerprint=fingerprint)


def test_fixed_split_rejects_other_split_version():
    with pytest.raises(ValueError, match="gene-split-v1"):
        enforce_fixed_split(split_frame(), split_version="gene-split-v2")


def test_conservation_fit_uses_training_medians():
    frame = pd.DataFrame({"phyloP": [1.0, None, 5.0], "phastCons": [0.2, 0.8, None]})
    assert fit_conservation_features(frame, ["phyloP", "phastCons"]) == ConservationFeatureSpec(
        ("phyloP", "phastCons"), (3.0, 0.5)
    )


def test_conservation_transform_adds_explicit_missingness_flags():
    frame = pd.DataFrame({"phyloP": [None, 4.0], "phastCons": [0.25, None]})
    spec = ConservationFeatureSpec(("phyloP", "phastCons"), (3.0, 0.5))
    result = transform_conservation_features(frame, spec)
    assert result.dtype == torch.float32
    assert result.tolist() == [[3.0, 0.25, 1.0, 0.0], [4.0, 0.5, 0.0, 1.0]]


def test_conservation_fit_rejects_columns_with_no_valid_training_scores():
    with pytest.raises(ValueError, match="No valid training scores"):
        fit_conservation_features(pd.DataFrame({"phyloP": [None, None]}), ["phyloP"])


def test_experiment_report_requires_validation_auprc(tmp_path):
    with pytest.raises(ValueError, match="validation_auprc"):
        write_experiment_report(tmp_path / "report.json", {"model": "50m"})


def test_experiment_report_rejects_test_metrics_during_selection(tmp_path):
    with pytest.raises(ValueError, match="test metrics"):
        write_experiment_report(
            tmp_path / "report.json", {"validation_auprc": 0.1, "test_auprc": 0.2}
        )


def test_experiment_report_writes_durable_json(tmp_path):
    path = tmp_path / "report.json"
    write_experiment_report(path, {"validation_auprc": 0.1, "model": "50m"})
    assert path.read_text().endswith("\n")
