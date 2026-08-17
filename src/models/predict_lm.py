"""Export per-variant probabilities from a trained Phase 4 checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.models.finetune_lm import MODEL_NAME, PairCollator, PairedSequenceClassifier, VariantPairDataset


@torch.no_grad()
def export_predictions(dataset_path: Path, checkpoint_path: Path, output_path: Path, batch_size: int = 32) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Fine-tuned inference requires a CUDA GPU; run this export in Colab")
    frame = pd.read_csv(dataset_path, sep="\t", usecols=["Reference_Sequence", "Alternate_Sequence", "Label", "Split"])
    selected = frame.Split.isin(["validation", "test"])
    evaluation = frame.loc[selected].copy()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    pretrained = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = PairedSequenceClassifier(pretrained.base_model, pretrained.config.hidden_size)
    saved = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(saved["model_state_dict"]); model.to(device); model.eval()
    loader = DataLoader(VariantPairDataset(evaluation), batch_size=batch_size, shuffle=False, collate_fn=PairCollator(tokenizer), num_workers=2, pin_memory=True)
    probabilities = []
    for batch_index, batch in enumerate(loader, start=1):
        batch = {key: value.to(device) for key, value in batch.items()}
        probabilities.extend(torch.softmax(model(batch), dim=-1)[:, 1].cpu().tolist())
        if batch_index % 100 == 0:
            print(f"Scored {min(batch_index * batch_size, len(evaluation)):,}/{len(evaluation):,} variants", flush=True)
    predictions = pd.DataFrame({
        "dataset_row": evaluation.index, "split": evaluation.Split.values,
        "label": evaluation.Label.astype(int).values, "fine_tuned_probability": probabilities,
    })
    output_path.parent.mkdir(parents=True, exist_ok=True); predictions.to_csv(output_path, sep="\t", index=False)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True); parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(); predictions = export_predictions(args.dataset, args.checkpoint, args.output, args.batch_size)
    print(f"Saved {len(predictions):,} predictions to {args.output}")


if __name__ == "__main__":
    main()
