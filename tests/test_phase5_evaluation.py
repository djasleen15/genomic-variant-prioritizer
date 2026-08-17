import numpy as np
import pandas as pd
import pytest

from src.evaluation.phase5 import (
    probability_summary,
    stratified_bootstrap_difference,
    validate_and_merge,
)


def test_paired_bootstrap_detects_clear_improvement():
    labels = np.array([0] * 80 + [1] * 20)
    baseline = np.linspace(0.8, 0.2, 100)
    fine_tuned = np.concatenate([np.linspace(0.0, 0.4, 80), np.linspace(0.6, 1.0, 20)])
    result = stratified_bootstrap_difference(labels, baseline, fine_tuned, iterations=200, seed=7)
    assert result["difference"] > 0
    assert result["difference_ci_95"][0] > 0
    assert result["difference_ci_excludes_zero"] is True


def test_paired_bootstrap_is_deterministic():
    labels = np.array([0, 0, 0, 1, 1, 1])
    baseline = np.array([0.1, 0.4, 0.3, 0.6, 0.5, 0.9])
    fine_tuned = np.array([0.2, 0.3, 0.1, 0.7, 0.8, 0.9])
    first = stratified_bootstrap_difference(labels, baseline, fine_tuned, 100, 42)
    second = stratified_bootstrap_difference(labels, baseline, fine_tuned, 100, 42)
    assert first == second


def test_merge_rejects_unpaired_rows(tmp_path):
    baseline = pd.DataFrame({"dataset_row": [1, 2], "split": ["test"] * 2, "label": [0, 1], "baseline_probability": [0.1, 0.8]})
    fine = pd.DataFrame({"dataset_row": [1, 3], "split": ["test"] * 2, "label": [0, 1], "fine_tuned_probability": [0.2, 0.9]})
    baseline_path, fine_path = tmp_path / "baseline.tsv", tmp_path / "fine.tsv"
    baseline.to_csv(baseline_path, sep="\t", index=False); fine.to_csv(fine_path, sep="\t", index=False)
    with pytest.raises(ValueError, match="identical rows"):
        validate_and_merge(baseline_path, fine_path)


def test_probability_summary_reports_distribution():
    summary = probability_summary(pd.Series([0.1, 0.2, 0.3, 0.4]))
    assert summary["min"] == 0.1
    assert summary["max"] == 0.4
    assert summary["mean"] == pytest.approx(0.25)
    assert summary["median"] == pytest.approx(0.25)
