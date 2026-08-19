import pandas as pd
import pytest

from src.evaluation.phase8 import LIMITATION, select_representative_examples


def test_representative_selection_is_deterministic_and_balanced():
    predictions = pd.DataFrame(
        {
            "dataset_row": range(12),
            "label": [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1],
            "fine_tuned_probability": [
                0.55, 0.65, 0.75, 0.95, 0.05, 0.15, 0.25, 0.45, 0.9, 0.1, 0.8, 0.2
            ],
        }
    )
    first = select_representative_examples(predictions, 3, 2)
    second = select_representative_examples(predictions, 3, 2)
    pd.testing.assert_frame_equal(first, second)
    assert (first.category == "correct_driver").sum() == 3
    assert (first.category == "correct_passenger").sum() == 3
    assert (first.category.str.startswith("false_")).sum() == 2


def test_representative_selection_validates_columns():
    with pytest.raises(ValueError, match="Missing prediction columns"):
        select_representative_examples(pd.DataFrame({"label": [0]}))


def test_limitation_is_explicitly_noncausal():
    lower = LIMITATION.lower()
    assert "not a definitive causal explanation" in lower
    assert "not establish" in lower
    assert "unsettled research area" in lower
