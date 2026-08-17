import pandas as pd

from src.models.baseline import (
    VARIANT_INDEX,
    build_features,
    build_model,
    evaluate,
    feature_row,
)


def dataset_row(**overrides):
    reference = "A" * 255 + "GCG" + "A" * 254
    alternate = reference[:VARIANT_INDEX] + "T" + reference[VARIANT_INDEX + 1 :]
    row = {
        "Reference_Sequence": reference,
        "Alternate_Sequence": alternate,
        "Reference_Allele": "C",
        "Alternate_Allele": "T",
        "CGC_Tier": "1",
        "Label": 1,
        "Split": "train",
    }
    row.update(overrides)
    return row


def test_feature_row_extracts_interpretable_sequence_features():
    features = feature_row(pd.Series(dataset_row()))

    assert features["Reference_Base"] == "C"
    assert features["Alternate_Base"] == "T"
    assert features["Trinucleotide_Context"] == "GCG"
    assert features["Is_Transition"] == 1
    assert features["Is_CpG_Context"] == 1
    assert features["CGC_Member"] == 1
    assert features["GC_Content_Delta"] < 0


def test_sequence_only_model_excludes_cgc_feature_and_uses_balanced_weights():
    model = build_model(include_cgc=False)
    numeric_columns = model.named_steps["features"].transformers[0][2]

    assert "CGC_Member" not in numeric_columns
    assert model.named_steps["classifier"].class_weight == "balanced"


def test_random_forest_uses_balanced_subsample_weights():
    model = build_model(include_cgc=False, model_kind="random_forest")

    assert model.named_steps["classifier"].class_weight == "balanced_subsample"
    assert model.named_steps["classifier"].random_state == 42


def test_baseline_fits_and_evaluates_on_synthetic_data():
    rows = []
    for index in range(20):
        label = index % 2
        reference = "C" if label else "A"
        alternate = "T" if label else "G"
        sequence = "A" * VARIANT_INDEX + reference + "A" * 255
        rows.append(
            dataset_row(
                Reference_Sequence=sequence,
                Alternate_Sequence=(
                    sequence[:VARIANT_INDEX]
                    + alternate
                    + sequence[VARIANT_INDEX + 1 :]
                ),
                Reference_Allele=reference,
                Alternate_Allele=alternate,
                CGC_Tier="1" if label else "",
                Label=label,
            )
        )
    dataset = pd.DataFrame(rows)
    features = build_features(dataset)
    model = build_model(include_cgc=False)
    model.fit(features, dataset["Label"])

    metrics = evaluate(model, features, dataset["Label"])

    assert 0 <= metrics["auprc"] <= 1
    assert set(metrics["thresholds"]) == {"0.25", "0.5", "0.75"}
