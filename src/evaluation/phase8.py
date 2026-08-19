"""Phase 8 attention diagnostics for the approved Phase 4 classifier.

Attention is summarized from the final encoder layer for the mutation-token
query, matching the token representation consumed by the Phase 4 classifier.
The resulting weights are diagnostic observations, not causal attributions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.models.finetune_lm import (
    MODEL_NAME,
    VARIANT_INDEX,
    VARIANT_TOKEN_INDEX,
    PairCollator,
    PairedSequenceClassifier,
    VariantPairDataset,
    token_spans,
)


LIMITATION = (
    "Attention weights are a diagnostic aid, not a definitive causal explanation. "
    "They show where the model allocated attention, but do not establish which bases "
    "caused the prediction or prove biological understanding; sequence attribution "
    "for genomic transformers remains an unsettled research area."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantile_rows(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    """Choose probability-quantile-spaced rows with deterministic tie-breaking."""
    if frame.empty or count <= 0:
        return frame.iloc[0:0]
    ordered = frame.sort_values(["fine_tuned_probability", "dataset_row"])
    positions = np.linspace(0, len(ordered) - 1, min(count, len(ordered)))
    indices = sorted({int(round(position)) for position in positions})
    return ordered.iloc[indices]


def select_representative_examples(
    predictions: pd.DataFrame,
    per_correct_class: int = 3,
    include_errors: int = 2,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Select examples by a declared rule rather than subjective inspection."""
    required = {"dataset_row", "label", "fine_tuned_probability"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")

    work = predictions.copy()
    work["predicted_label"] = (work.fine_tuned_probability >= threshold).astype(int)
    selected: list[pd.DataFrame] = []
    categories = (
        ("correct_driver", (work.label == 1) & (work.predicted_label == 1)),
        ("correct_passenger", (work.label == 0) & (work.predicted_label == 0)),
    )
    for category, mask in categories:
        rows = _quantile_rows(work.loc[mask], per_correct_class).copy()
        rows["category"] = category
        selected.append(rows)

    errors = work.loc[work.label != work.predicted_label].copy()
    if include_errors and not errors.empty:
        errors["error_confidence"] = np.where(
            errors.predicted_label == 1,
            errors.fine_tuned_probability,
            1 - errors.fine_tuned_probability,
        )
        errors = errors.sort_values(
            ["error_confidence", "dataset_row"], ascending=[False, True]
        ).head(include_errors)
        errors["category"] = np.where(
            errors.predicted_label == 1, "false_positive", "false_negative"
        )
        selected.append(errors)

    result = pd.concat(selected, ignore_index=True) if selected else work.iloc[0:0]
    result["selection_rule"] = (
        "probability-quantile-spaced within each correct class; "
        "highest-confidence errors"
    )
    return result.drop(columns=["error_confidence"], errors="ignore")


@torch.no_grad()
def score_test_set(
    model: PairedSequenceClassifier,
    tokenizer: Any,
    test_frame: pd.DataFrame,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    loader = DataLoader(
        VariantPairDataset(test_frame),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=PairCollator(tokenizer),
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    probabilities: list[float] = []
    model.eval()
    for batch_number, batch in enumerate(loader, start=1):
        batch = {key: value.to(device) for key, value in batch.items()}
        probabilities.extend(torch.softmax(model(batch), dim=-1)[:, 1].cpu().tolist())
        if batch_number % 100 == 0:
            done = min(batch_number * batch_size, len(test_frame))
            print(f"Scored {done:,}/{len(test_frame):,} test variants", flush=True)
    return pd.DataFrame(
        {
            "dataset_row": test_frame.index.astype(int),
            "label": test_frame.Label.astype(int).values,
            "fine_tuned_probability": probabilities,
        }
    )


@torch.no_grad()
def mutation_query_attention(
    model: PairedSequenceClassifier,
    tokenizer: Any,
    sequence: str,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Return mean-head, final-layer attention from mutation query to DNA tokens."""
    encoded = tokenizer(sequence, add_special_tokens=True, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    outputs = model.encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True,
        return_dict=True,
    )
    if not outputs.attentions:
        raise RuntimeError("Encoder did not return attention weights")
    final_layer = outputs.attentions[-1]
    weights = final_layer[0, :, VARIANT_TOKEN_INDEX, :].mean(dim=0).cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
    spans = token_spans(tokens)
    dna_total = sum(float(weights[span["token_index"]]) for span in spans)
    if dna_total <= 0:
        raise RuntimeError("DNA-token attention weights sum to zero")
    records = []
    for span in spans:
        weight = float(weights[span["token_index"]])
        records.append(
            {
                **span,
                "relative_start": span["start"] - VARIANT_INDEX,
                "relative_end": span["end"] - VARIANT_INDEX,
                "relative_center": (span["start"] + span["end"]) / 2 - VARIANT_INDEX,
                "attention_weight_raw": weight,
                "attention_weight_dna_normalized": weight / dna_total,
                "is_mutation_token": span["token_index"] == VARIANT_TOKEN_INDEX,
            }
        )
    return records


def plot_attention(example: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.2))
    colors = {"reference": "#3366cc", "alternate": "#d64b3c"}
    for sequence_type, group in example.groupby("sequence_type", sort=False):
        ax.plot(
            group.relative_center,
            group.attention_weight_dna_normalized,
            marker="o",
            markersize=2.8,
            linewidth=1.2,
            label=sequence_type.capitalize(),
            color=colors[sequence_type],
        )
    mutation = example.loc[example.is_mutation_token]
    ax.axvspan(
        mutation.relative_start.min(),
        mutation.relative_end.max(),
        color="#f2c94c",
        alpha=0.35,
        label=f"Mutation-containing token (index {VARIANT_TOKEN_INDEX})",
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set(xlabel="Base position relative to mutation", ylabel="DNA-normalized attention weight")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.18)
    fig.text(0.01, 0.01, "Diagnostic attention, not causal attribution.", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def load_approved_model(checkpoint_path: Path, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    pretrained = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = PairedSequenceClassifier(pretrained.base_model, pretrained.config.hidden_size)
    saved = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(saved["model_state_dict"])
    model.to(device).eval()
    return model, tokenizer


def run(
    dataset_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    batch_size: int = 32,
    predictions_path: Path | None = None,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 8 inference requires a CUDA GPU")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    frame = pd.read_csv(dataset_path, sep="\t")
    test = frame.loc[frame.Split == "test"].copy()
    if test.empty:
        raise ValueError("Dataset contains no test rows")
    model, tokenizer = load_approved_model(checkpoint_path, device)
    if predictions_path is None:
        predictions = score_test_set(model, tokenizer, test, device, batch_size)
        prediction_source = "scored by Phase 8"
    else:
        archived = pd.read_csv(predictions_path, sep="\t")
        predictions = archived.loc[archived.split == "test", [
            "dataset_row", "label", "fine_tuned_probability"
        ]].copy()
        if predictions.empty:
            raise ValueError("Archived predictions contain no test rows")
        if not set(predictions.dataset_row).issubset(set(test.index)):
            raise ValueError("Archived predictions do not match dataset test rows")
        prediction_source = "approved Phase 5 fine_tuned_predictions.tsv"
    selected = select_representative_examples(predictions)
    selected.to_csv(output_dir / "selected_examples.csv", index=False)

    all_attention: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    for ordinal, selected_row in selected.reset_index(drop=True).iterrows():
        dataset_row = int(selected_row.dataset_row)
        source = frame.loc[dataset_row]
        example_id = f"{ordinal + 1:02d}_{selected_row.category}_row_{dataset_row}"
        rows: list[dict[str, Any]] = []
        for sequence_type, column in (
            ("reference", "Reference_Sequence"),
            ("alternate", "Alternate_Sequence"),
        ):
            for record in mutation_query_attention(
                model, tokenizer, source[column], device
            ):
                # Do not persist DNA token strings from the private cohort dataset.
                # Positions and weights are sufficient to reproduce the plots.
                record = {key: value for key, value in record.items() if key != "token"}
                rows.append(
                    {
                        "example_id": example_id,
                        "dataset_row": dataset_row,
                        "category": selected_row.category,
                        "label": int(selected_row.label),
                        "predicted_label": int(selected_row.predicted_label),
                        "fine_tuned_probability": float(
                            selected_row.fine_tuned_probability
                        ),
                        "sequence_type": sequence_type,
                        **record,
                    }
                )
        example_frame = pd.DataFrame(rows)
        all_attention.extend(rows)
        plot_name = f"attention_{example_id}.png"
        plot_attention(
            example_frame,
            output_dir / plot_name,
            (
                f"{selected_row.category.replace('_', ' ').title()} — "
                f"row {dataset_row}, label={int(selected_row.label)}, "
                f"score={selected_row.fine_tuned_probability:.3f}"
            ),
        )
        canonical = None
        if selected_row.category == "correct_driver" and not any(
            item.get("canonical") == "true_driver_example.png" for item in examples
        ):
            canonical = "true_driver_example.png"
        elif selected_row.category == "correct_passenger" and not any(
            item.get("canonical") == "true_passenger_example.png" for item in examples
        ):
            canonical = "true_passenger_example.png"
        if canonical:
            shutil.copyfile(output_dir / plot_name, output_dir / canonical)
        examples.append(
            {
                "example_id": example_id,
                "dataset_row": dataset_row,
                "category": selected_row.category,
                "label": int(selected_row.label),
                "predicted_label": int(selected_row.predicted_label),
                "fine_tuned_probability": float(selected_row.fine_tuned_probability),
                "plot": plot_name,
                "canonical": canonical,
            }
        )

    pd.DataFrame(all_attention).to_csv(output_dir / "attention_weights.csv", index=False)
    report = {
        "phase": 8,
        "status": "complete",
        "model": "approved Phase 4 checkpoint",
        "model_name": MODEL_NAME,
        "checkpoint_sha256": sha256(checkpoint_path),
        "split": "gene-split-v1 test only",
        "prediction_source": prediction_source,
        "attention_summary": (
            "Final encoder layer, mutation-token query (token index 43), mean across heads; "
            "reference and alternate sequences shown separately."
        ),
        "selection_policy": (
            "Three probability-quantile-spaced correct examples per class and up to two "
            "highest-confidence errors, selected deterministically before attention inspection."
        ),
        "examples": examples,
        "limitation": LIMITATION,
    }
    (output_dir / "phase8_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/phase8"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Optional approved Phase 5 prediction export; avoids rescoring the test set",
    )
    args = parser.parse_args()
    report = run(
        args.dataset,
        args.checkpoint,
        args.output_dir,
        args.batch_size,
        args.predictions,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
