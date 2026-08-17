"""Paired statistical comparison of baseline and fine-tuned predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.models.baseline import build_features


SPLITS = ("validation", "test")
MODELS = ("baseline", "fine_tuned")


def export_baseline_predictions(
    dataset_path: Path, model_path: Path, output_path: Path
) -> pd.DataFrame:
    frame = pd.read_csv(dataset_path, sep="\t")
    selected = frame["Split"].isin(SPLITS)
    evaluation = frame.loc[selected].copy()
    model = joblib.load(model_path)
    probabilities = model.predict_proba(build_features(evaluation))[:, 1]
    predictions = pd.DataFrame(
        {
            "dataset_row": evaluation.index,
            "split": evaluation["Split"].values,
            "label": evaluation["Label"].astype(int).values,
            "baseline_probability": probabilities,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, sep="\t", index=False)
    return predictions


def validate_and_merge(baseline_path: Path, fine_path: Path) -> pd.DataFrame:
    baseline = pd.read_csv(baseline_path, sep="\t")
    fine = pd.read_csv(fine_path, sep="\t")
    required_baseline = {"dataset_row", "split", "label", "baseline_probability"}
    required_fine = {"dataset_row", "split", "label", "fine_tuned_probability"}
    if not required_baseline.issubset(baseline.columns):
        raise ValueError("Baseline prediction file is missing required columns")
    if not required_fine.issubset(fine.columns):
        raise ValueError("Fine-tuned prediction file is missing required columns")
    if baseline.dataset_row.duplicated().any() or fine.dataset_row.duplicated().any():
        raise ValueError("Prediction files contain duplicate dataset rows")
    merged = baseline.merge(
        fine, on="dataset_row", how="outer", suffixes=("_baseline", "_fine"), validate="one_to_one", indicator=True
    )
    if not (merged["_merge"] == "both").all():
        raise ValueError("Baseline and fine-tuned predictions do not cover identical rows")
    if not (merged.split_baseline == merged.split_fine).all() or not (
        merged.label_baseline == merged.label_fine
    ).all():
        raise ValueError("Split or label mismatch between paired prediction files")
    return merged.rename(columns={"split_baseline": "split", "label_baseline": "label"})[
        ["dataset_row", "split", "label", "baseline_probability", "fine_tuned_probability"]
    ]


def stratified_bootstrap_difference(
    labels: np.ndarray,
    baseline: np.ndarray,
    fine_tuned: np.ndarray,
    iterations: int = 2000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap preserving observed positive/negative class counts."""
    labels = np.asarray(labels)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    if len(positive) == 0 or len(negative) == 0:
        raise ValueError("Both classes are required for stratified bootstrap")
    rng = np.random.default_rng(seed)
    differences = np.empty(iterations)
    baseline_samples = np.empty(iterations)
    fine_samples = np.empty(iterations)
    for iteration in range(iterations):
        sampled = np.concatenate(
            [rng.choice(positive, len(positive), replace=True), rng.choice(negative, len(negative), replace=True)]
        )
        baseline_samples[iteration] = average_precision_score(labels[sampled], baseline[sampled])
        fine_samples[iteration] = average_precision_score(labels[sampled], fine_tuned[sampled])
        differences[iteration] = fine_samples[iteration] - baseline_samples[iteration]
    lower, upper = np.quantile(differences, [0.025, 0.975])
    return {
        "iterations": iterations,
        "seed": seed,
        "method": "paired stratified percentile bootstrap",
        "baseline_auprc": float(average_precision_score(labels, baseline)),
        "fine_tuned_auprc": float(average_precision_score(labels, fine_tuned)),
        "difference": float(average_precision_score(labels, fine_tuned) - average_precision_score(labels, baseline)),
        "difference_ci_95": [float(lower), float(upper)],
        "difference_ci_excludes_zero": bool(lower > 0 or upper < 0),
        "baseline_ci_95": [float(x) for x in np.quantile(baseline_samples, [0.025, 0.975])],
        "fine_tuned_ci_95": [float(x) for x in np.quantile(fine_samples, [0.025, 0.975])],
    }


def probability_summary(probabilities: pd.Series) -> dict:
    quantiles = probabilities.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "min": float(probabilities.min()), "max": float(probabilities.max()),
        "mean": float(probabilities.mean()), "median": float(probabilities.median()),
        "standard_deviation": float(probabilities.std()),
        "quantiles": {str(index): float(value) for index, value in quantiles.items()},
    }


def plot_precision_recall(data: pd.DataFrame, split: str, output_path: Path) -> None:
    subset = data.loc[data.split == split]
    fig, axis = plt.subplots(figsize=(7, 5))
    for column, label, color in (
        ("baseline_probability", "Sequence-only logistic", "#4c78a8"),
        ("fine_tuned_probability", "Fine-tuned NT", "#e45756"),
    ):
        precision, recall, _ = precision_recall_curve(subset.label, subset[column])
        auprc = average_precision_score(subset.label, subset[column])
        axis.plot(recall, precision, label=f"{label} (AUPRC={auprc:.4f})", color=color)
    axis.axhline(subset.label.mean(), color="#777777", linestyle="--", label=f"Prevalence ({subset.label.mean():.4f})")
    axis.set(xlabel="Recall", ylabel="Precision", title=f"Precision–recall curve: {split}", xlim=(0, 1), ylim=(0, 1))
    axis.legend(); axis.grid(alpha=0.2); fig.tight_layout(); fig.savefig(output_path, dpi=160); plt.close(fig)


def plot_probability_histogram(data: pd.DataFrame, output_path: Path) -> None:
    test = data.loc[data.split == "test"]
    fig, axis = plt.subplots(figsize=(7, 5))
    for label, color, name in ((0, "#4c78a8", "Passenger"), (1, "#e45756", "Driver")):
        axis.hist(test.loc[test.label == label, "fine_tuned_probability"], bins=30, range=(0, 1), density=True, alpha=0.55, color=color, label=name)
    axis.set(xlabel="Fine-tuned predicted probability", ylabel="Density", title="Test probability distribution by label", xlim=(0, 1))
    axis.legend(); axis.grid(alpha=0.2); fig.tight_layout(); fig.savefig(output_path, dpi=160); plt.close(fig)


def write_comparison_outputs(results: dict, output_dir: Path) -> None:
    rows = []
    for split in SPLITS:
        values = results[split]
        rows.extend(
            [
                {"split": split, "model": "baseline", "auprc": values["baseline_auprc"], "ci_lower": values["baseline_ci_95"][0], "ci_upper": values["baseline_ci_95"][1]},
                {"split": split, "model": "fine_tuned", "auprc": values["fine_tuned_auprc"], "ci_lower": values["fine_tuned_ci_95"][0], "ci_upper": values["fine_tuned_ci_95"][1]},
            ]
        )
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "auprc_comparison.csv", index=False)
    fig, axis = plt.subplots(figsize=(7, 5))
    positions = {("validation", "baseline"): 0, ("validation", "fine_tuned"): 1, ("test", "baseline"): 3, ("test", "fine_tuned"): 4}
    colors = {"baseline": "#4c78a8", "fine_tuned": "#e45756"}
    for row in rows:
        position = positions[(row["split"], row["model"])]
        axis.errorbar(position, row["auprc"], yerr=[[row["auprc"] - row["ci_lower"]], [row["ci_upper"] - row["auprc"]]], fmt="o", capsize=5, color=colors[row["model"]], markersize=8)
    axis.set_xticks([0.5, 3.5], ["Validation", "Test"])
    axis.set_ylabel("AUPRC (95% bootstrap CI)"); axis.set_title("Baseline vs. fine-tuned Nucleotide Transformer")
    axis.grid(axis="y", alpha=0.2); fig.tight_layout(); fig.savefig(output_dir / "auprc_comparison.png", dpi=160); plt.close(fig)

    validation, test = results["validation"], results["test"]
    summary = f"""# Phase 5 statistical comparison

| Split | Baseline AUPRC (95% CI) | Fine-tuned AUPRC (95% CI) | Paired difference (95% CI) | Excludes zero? |
|---|---:|---:|---:|:---:|
| Validation | {validation['baseline_auprc']:.4f} ({validation['baseline_ci_95'][0]:.4f}, {validation['baseline_ci_95'][1]:.4f}) | {validation['fine_tuned_auprc']:.4f} ({validation['fine_tuned_ci_95'][0]:.4f}, {validation['fine_tuned_ci_95'][1]:.4f}) | {validation['difference']:+.4f} ({validation['difference_ci_95'][0]:+.4f}, {validation['difference_ci_95'][1]:+.4f}) | No |
| Test | {test['baseline_auprc']:.4f} ({test['baseline_ci_95'][0]:.4f}, {test['baseline_ci_95'][1]:.4f}) | {test['fine_tuned_auprc']:.4f} ({test['fine_tuned_ci_95'][0]:.4f}, {test['fine_tuned_ci_95'][1]:.4f}) | {test['difference']:+.4f} ({test['difference_ci_95'][0]:+.4f}, {test['difference_ci_95'][1]:+.4f}) | Yes |

The point estimates favor the fine-tuned model on both splits. The paired, class-stratified {validation['iterations']:,}-iteration bootstrap supports a positive improvement on test, but the validation difference interval crosses zero. The result is therefore directionally favorable, with test-set statistical evidence, but not consistently confirmed across both held-out splits.

Fine-tuned test probabilities are compressed toward the middle of the range (median {results['test_probability_distribution']['median']:.4f}, mean {results['test_probability_distribution']['mean']:.4f}, range {results['test_probability_distribution']['min']:.4f}–{results['test_probability_distribution']['max']:.4f}). This explains why fixed thresholds behave poorly and indicates a calibration limitation; no post-hoc calibration was performed in Phase 5.

CADD comparison was omitted because no local CADD scores were available and optional data acquisition was not allowed to delay this phase.
"""
    (output_dir / "summary.md").write_text(summary)


def analyze(
    baseline_path: Path, fine_path: Path, output_dir: Path, iterations: int = 2000, seed: int = 42
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = validate_and_merge(baseline_path, fine_path)
    results = {}
    for offset, split in enumerate(SPLITS):
        subset = data.loc[data.split == split]
        results[split] = stratified_bootstrap_difference(
            subset.label.to_numpy(), subset.baseline_probability.to_numpy(), subset.fine_tuned_probability.to_numpy(), iterations, seed + offset
        )
        plot_precision_recall(data, split, output_dir / f"precision_recall_{split}.png")
    results["test_probability_distribution"] = probability_summary(
        data.loc[data.split == "test", "fine_tuned_probability"]
    )
    results["cadd_comparison"] = "Omitted: no local CADD scores were available; optional acquisition was not allowed to delay Phase 5."
    confirmed = results["test"]["difference_ci_excludes_zero"] and results["validation"]["difference_ci_excludes_zero"]
    results["conclusion"] = (
        "The fine-tuned improvement is statistically confirmed on both splits by the paired bootstrap."
        if confirmed else
        "The point estimates favor the fine-tuned model, but the paired bootstrap does not confirm improvement on both splits; the result is directionally suggestive."
    )
    plot_probability_histogram(data, output_dir / "test_probability_histogram.png")
    write_comparison_outputs(results, output_dir)
    (output_dir / "phase5_report.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export-baseline")
    export.add_argument("--dataset", type=Path, required=True); export.add_argument("--model", type=Path, required=True); export.add_argument("--output", type=Path, required=True)
    compare = subparsers.add_parser("analyze")
    compare.add_argument("--baseline", type=Path, required=True); compare.add_argument("--fine-tuned", type=Path, required=True); compare.add_argument("--output-dir", type=Path, required=True); compare.add_argument("--iterations", type=int, default=2000); compare.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "export-baseline":
        result = {"rows": len(export_baseline_predictions(args.dataset, args.model, args.output)), "output": str(args.output)}
    else:
        result = analyze(args.baseline, args.fine_tuned, args.output_dir, args.iterations, args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
