"""Train and evaluate interpretable class-weighted logistic baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


VARIANT_INDEX = 256
RANDOM_SEED = 42
THRESHOLDS = (0.25, 0.5, 0.75)
SEQUENCE_NUMERIC_FEATURES = [
    "Reference_GC_Content",
    "Alternate_GC_Content",
    "GC_Content_Delta",
    "Is_Transition",
    "Is_CpG_Context",
]
SEQUENCE_CATEGORICAL_FEATURES = [
    "Reference_Base",
    "Alternate_Base",
    "Trinucleotide_Context",
]
CGC_FEATURE = "CGC_Member"
INPUT_COLUMNS = [
    "Reference_Sequence",
    "Alternate_Sequence",
    "Reference_Allele",
    "Alternate_Allele",
    "CGC_Tier",
    "Label",
    "Split",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mlflow-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def gc_content(sequence: str) -> float:
    return (sequence.count("G") + sequence.count("C")) / len(sequence)


def is_transition(reference: str, alternate: str) -> int:
    return int((reference, alternate) in {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")})


def feature_row(row: pd.Series) -> dict[str, float | int | str]:
    reference_sequence = row["Reference_Sequence"]
    alternate_sequence = row["Alternate_Sequence"]
    reference = row["Reference_Allele"]
    alternate = row["Alternate_Allele"]
    reference_gc = gc_content(reference_sequence)
    alternate_gc = gc_content(alternate_sequence)
    trinucleotide = reference_sequence[VARIANT_INDEX - 1 : VARIANT_INDEX + 2]
    is_cpg = int(
        (reference == "C" and reference_sequence[VARIANT_INDEX + 1] == "G")
        or (reference == "G" and reference_sequence[VARIANT_INDEX - 1] == "C")
    )
    return {
        "Reference_GC_Content": reference_gc,
        "Alternate_GC_Content": alternate_gc,
        "GC_Content_Delta": alternate_gc - reference_gc,
        "Is_Transition": is_transition(reference, alternate),
        "Is_CpG_Context": is_cpg,
        "Reference_Base": reference,
        "Alternate_Base": alternate,
        "Trinucleotide_Context": trinucleotide,
        CGC_FEATURE: int(bool(row["CGC_Tier"])),
    }


def build_features(dataset: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        (feature_row(row) for _, row in dataset.iterrows()), index=dataset.index
    )


def build_model(include_cgc: bool, model_kind: str = "logistic_regression") -> Pipeline:
    numeric_features = [*SEQUENCE_NUMERIC_FEATURES]
    if include_cgc:
        numeric_features.append(CGC_FEATURE)
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric_features),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                SEQUENCE_CATEGORICAL_FEATURES,
            ),
        ]
    )
    if model_kind == "logistic_regression":
        classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_SEED,
            solver="liblinear",
        )
    elif model_kind == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model kind: {model_kind}")
    return Pipeline([("features", preprocessor), ("classifier", classifier)])


def evaluate(model: Pipeline, features: pd.DataFrame, labels: pd.Series) -> dict:
    probabilities = model.predict_proba(features)[:, 1]
    metrics: dict[str, object] = {
        "auprc": average_precision_score(labels, probabilities),
        "positive_rate": float(labels.mean()),
        "thresholds": {},
    }
    for threshold in THRESHOLDS:
        predictions = probabilities >= threshold
        metrics["thresholds"][str(threshold)] = {
            "precision": precision_score(labels, predictions, zero_division=0),
            "recall": recall_score(labels, predictions, zero_division=0),
            "predicted_positive": int(predictions.sum()),
        }
    return metrics


def flatten_metrics(prefix: str, metrics: dict) -> dict[str, float]:
    flattened = {
        f"{prefix}_auprc": metrics["auprc"],
        f"{prefix}_positive_rate": metrics["positive_rate"],
    }
    for threshold, values in metrics["thresholds"].items():
        label = threshold.replace(".", "_")
        flattened[f"{prefix}_precision_at_{label}"] = values["precision"]
        flattened[f"{prefix}_recall_at_{label}"] = values["recall"]
    return flattened


def train_run(
    name: str,
    include_cgc: bool,
    model_kind: str,
    features: pd.DataFrame,
    labels: pd.Series,
    splits: pd.Series,
    output_dir: Path,
) -> tuple[dict, str]:
    train_mask = splits == "train"
    validation_mask = splits == "validation"
    test_mask = splits == "test"
    model = build_model(include_cgc, model_kind)

    with mlflow.start_run(run_name=name) as active_run:
        mlflow.log_params(
            {
                "model": model_kind,
                "class_weight": (
                    "balanced_subsample" if model_kind == "random_forest" else "balanced"
                ),
                "include_cgc_membership": include_cgc,
                "split_version": "gene-split-v1",
                "random_seed": RANDOM_SEED,
                "feature_count_numeric": len(SEQUENCE_NUMERIC_FEATURES)
                + int(include_cgc),
                "feature_count_categorical": len(SEQUENCE_CATEGORICAL_FEATURES),
            }
        )
        model.fit(features.loc[train_mask], labels.loc[train_mask])
        validation_metrics = evaluate(
            model, features.loc[validation_mask], labels.loc[validation_mask]
        )
        test_metrics = evaluate(model, features.loc[test_mask], labels.loc[test_mask])
        mlflow.log_metrics(
            {
                **flatten_metrics("validation", validation_metrics),
                **flatten_metrics("test", test_metrics),
            }
        )
        model_path = output_dir / f"{name}.joblib"
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="model_artifacts")
        mlflow.sklearn.log_model(model, artifact_path="sklearn_model")
        run_id = active_run.info.run_id

    result = {
        "include_cgc_membership": include_cgc,
        "model": model_kind,
        "class_weight": (
            "balanced_subsample" if model_kind == "random_forest" else "balanced"
        ),
        "validation": validation_metrics,
        "test": test_metrics,
        "model_path": str(model_path),
        "mlflow_run_id": run_id,
    }
    return result, run_id


def run(
    input_path: Path, output_dir: Path, mlflow_dir: Path, report_path: Path
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    mlflow_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_csv(
        input_path,
        sep="\t",
        usecols=INPUT_COLUMNS,
        dtype={"Label": "int8"},
        keep_default_na=False,
    )
    features = build_features(dataset)
    labels = dataset["Label"]
    splits = dataset["Split"]

    mlflow.set_tracking_uri(mlflow_dir.resolve().as_uri())
    mlflow.set_experiment("phase3-baseline")
    sequence_only, _ = train_run(
        "sequence_only_logistic",
        False,
        "logistic_regression",
        features,
        labels,
        splits,
        output_dir,
    )
    sequence_forest, _ = train_run(
        "sequence_only_random_forest",
        False,
        "random_forest",
        features,
        labels,
        splits,
        output_dir,
    )
    cgc_diagnostic, _ = train_run(
        "cgc_inclusive_diagnostic_logistic",
        True,
        "logistic_regression",
        features,
        labels,
        splits,
        output_dir,
    )
    sequence_models = {
        "sequence_only_logistic": sequence_only,
        "sequence_only_random_forest": sequence_forest,
    }
    baseline_name = max(
        sequence_models,
        key=lambda model_name: sequence_models[model_name]["validation"]["auprc"],
    )
    report = {
        "baseline_selection_rule": "highest validation AUPRC among sequence-only models",
        "baseline_to_beat": baseline_name,
        "baseline_to_beat_validation_auprc": sequence_models[baseline_name]["validation"][
            "auprc"
        ],
        "baseline_to_beat_test_auprc": sequence_models[baseline_name]["test"]["auprc"],
        "feature_set": {
            "numeric": SEQUENCE_NUMERIC_FEATURES,
            "categorical": SEQUENCE_CATEGORICAL_FEATURES,
            "diagnostic_only": [CGC_FEATURE],
        },
        "cgc_circularity_caveat": (
            "CGC membership is the label source. The CGC-inclusive result is a "
            "circular diagnostic, not the meaningful Phase 4 baseline."
        ),
        "conservation_score_omission": (
            "Conservation was omitted because no local score track was available; "
            "acquiring one is optional and outside the scoped baseline."
        ),
        "models": {
            **sequence_models,
            "cgc_inclusive_diagnostic_logistic": cgc_diagnostic,
        },
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    args = parse_args()
    report = run(args.input, args.output_dir, args.mlflow_dir, args.report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
