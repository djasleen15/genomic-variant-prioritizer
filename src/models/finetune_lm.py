"""Fine-tune Nucleotide Transformer on paired reference/alternate sequences."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer


MODEL_NAME = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
VARIANT_INDEX = 256
VARIANT_TOKEN_INDEX = 43
THRESHOLDS = (0.25, 0.5, 0.75)
BASELINE_VALIDATION_AUPRC = 0.08727184295835033
BASELINE_TEST_AUPRC = 0.07525126312375156
SPECIAL_TOKENS = {"<cls>", "<pad>", "<mask>", "<unk>", "<eos>", "<bos>"}


@dataclass
class TrainingConfig:
    model_name: str = MODEL_NAME
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 3
    patience: int = 1
    seed: int = 42
    freeze_encoder: bool = False
    audit_samples: int = 100
    pooling: str = "mutation"
    conservation_feature_count: int = 0


class VariantPairDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.references = frame["Reference_Sequence"].tolist()
        self.alternates = frame["Alternate_Sequence"].tolist()
        self.labels = frame["Label"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[str, str, int]:
        return self.references[index], self.alternates[index], self.labels[index]


class PairCollator:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __call__(self, batch: list[tuple[str, str, int]]) -> dict[str, torch.Tensor]:
        references, alternates, labels = zip(*batch)
        kwargs = {"padding": True, "truncation": True, "return_tensors": "pt"}
        ref = self.tokenizer(list(references), **kwargs)
        alt = self.tokenizer(list(alternates), **kwargs)
        return {
            "ref_input_ids": ref["input_ids"],
            "ref_attention_mask": ref["attention_mask"],
            "alt_input_ids": alt["input_ids"],
            "alt_attention_mask": alt["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class PairedSequenceClassifier(nn.Module):
    """Shared encoder using the contextualized mutation-containing token."""

    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int,
        dropout: float = 0.1,
        pooling: str = "mutation",
        conservation_feature_count: int = 0,
    ):
        super().__init__()
        if pooling not in {"mutation", "mean", "attention"}:
            raise ValueError(f"Unknown pooling strategy: {pooling}")
        if conservation_feature_count < 0:
            raise ValueError("conservation_feature_count cannot be negative")
        self.encoder = encoder
        self.pooling = pooling
        self.conservation_feature_count = conservation_feature_count
        self.attention_pooler = nn.Linear(hidden_size, 1) if pooling == "attention" else None
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 3 + conservation_feature_count, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_size, 2)
        )

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        mask = attention_mask.bool()
        if self.pooling == "mutation":
            if hidden.shape[1] <= VARIANT_TOKEN_INDEX:
                raise ValueError("Tokenized sequence is too short to contain the variant token")
            if not mask[:, VARIANT_TOKEN_INDEX].all():
                raise ValueError("Variant token is masked for at least one sequence")
            return hidden[:, VARIANT_TOKEN_INDEX, :]
        if self.pooling == "mean":
            weights = mask.unsqueeze(-1).to(hidden.dtype)
            return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        scores = self.attention_pooler(hidden).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (hidden * weights).sum(dim=1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        ref = self.encode(batch["ref_input_ids"], batch["ref_attention_mask"])
        alt = self.encode(batch["alt_input_ids"], batch["alt_attention_mask"])
        features = [ref, alt, alt - ref]
        if self.conservation_feature_count:
            conservation = batch.get("conservation_features")
            if conservation is None:
                raise ValueError("conservation_features are required by this model")
            if conservation.shape[-1] != self.conservation_feature_count:
                raise ValueError("Unexpected conservation feature width")
            features.append(conservation.to(ref.dtype))
        return self.classifier(torch.cat(features, dim=-1))


def token_spans(tokens: list[str]) -> list[dict[str, Any]]:
    """Reconstruct base spans for non-overlapping DNA k-mer tokens."""
    cursor = 0
    spans = []
    for token_index, token in enumerate(tokens):
        if token in SPECIAL_TOKENS:
            continue
        if not token or set(token.upper()) - set("ACGTN"):
            raise ValueError(f"Cannot reconstruct DNA span for token {token!r}")
        start, end = cursor, cursor + len(token)
        spans.append({"token_index": token_index, "token": token, "start": start, "end": end})
        cursor = end
    return spans


def covering_token(tokenizer: Any, sequence: str, position: int) -> tuple[dict, int]:
    encoded = tokenizer(sequence, add_special_tokens=True)
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    spans = token_spans(tokens)
    if sum(span["end"] - span["start"] for span in spans) != len(sequence):
        raise ValueError("Tokenizer DNA tokens do not reconstruct the full sequence")
    matches = [span for span in spans if span["start"] <= position < span["end"]]
    if len(matches) != 1:
        raise ValueError(f"Expected one token covering base {position}, found {len(matches)}")
    return matches[0], len(tokens)


def audit_tokenizer(
    frame: pd.DataFrame, tokenizer: Any, sample_size: int = 100, seed: int = 42
) -> dict:
    sample = frame.sample(n=min(sample_size, len(frame)), random_state=seed)
    records = []
    for index, row in sample.iterrows():
        ref_token, ref_length = covering_token(tokenizer, row.Reference_Sequence, VARIANT_INDEX)
        alt_token, alt_length = covering_token(tokenizer, row.Alternate_Sequence, VARIANT_INDEX)
        aligned = (ref_token["start"], ref_token["end"], ref_token["token_index"]) == (
            alt_token["start"], alt_token["end"], alt_token["token_index"]
        )
        records.append({
            "row_index": int(index), "ref_token": ref_token, "alt_token": alt_token,
            "ref_token_count": ref_length, "alt_token_count": alt_length,
            "same_token_count": ref_length == alt_length, "same_boundary_alignment": aligned,
            "variant_offset_within_token": VARIANT_INDEX - ref_token["start"],
        })
    if not all(r["same_token_count"] and r["same_boundary_alignment"] for r in records):
        raise ValueError("Reference/alternate tokenizer alignment mismatch detected")
    return {
        "sample_size": len(records), "variant_index_zero_based": VARIANT_INDEX,
        "all_pairs_same_token_count": True, "all_pairs_same_boundary_alignment": True,
        "unique_covering_spans": sorted({
            f"{r['ref_token']['start']}:{r['ref_token']['end']}" for r in records
        }),
        "unique_variant_offsets_within_token": sorted({
            r["variant_offset_within_token"] for r in records
        }),
        "records": records,
    }


def class_weights(labels: pd.Series) -> torch.Tensor:
    counts = labels.value_counts().sort_index()
    if set(counts.index) != {0, 1}:
        raise ValueError("Training labels must contain both classes 0 and 1")
    total = len(labels)
    return torch.tensor([total / (2 * counts[0]), total / (2 * counts[1])], dtype=torch.float32)


def metrics(labels: list[int], probabilities: list[float]) -> dict:
    y, p = np.asarray(labels), np.asarray(probabilities)
    result = {"auprc": float(average_precision_score(y, p)), "positive_rate": float(y.mean()), "thresholds": {}}
    for threshold in THRESHOLDS:
        predictions = p >= threshold
        result["thresholds"][str(threshold)] = {
            "precision": float(precision_score(y, predictions, zero_division=0)),
            "recall": float(recall_score(y, predictions, zero_division=0)),
            "predicted_positive": int(predictions.sum()),
        }
    return result


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval(); labels, probabilities = [], []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        logits = model(batch)
        probabilities.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())
        labels.extend(batch["labels"].cpu().tolist())
    return metrics(labels, probabilities)


def train_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train(); total_loss = 0.0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(batch), batch["labels"])
        loss.backward(); optimizer.step()
        total_loss += loss.item() * len(batch["labels"])
    return total_loss / len(loader.dataset)


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def flatten(prefix: str, values: dict) -> dict[str, float]:
    output = {f"{prefix}_auprc": values["auprc"], f"{prefix}_positive_rate": values["positive_rate"]}
    for threshold, scores in values["thresholds"].items():
        suffix = threshold.replace(".", "_")
        output[f"{prefix}_precision_at_{suffix}"] = scores["precision"]
        output[f"{prefix}_recall_at_{suffix}"] = scores["recall"]
    return output


def run(input_path: Path, output_dir: Path, mlflow_dir: Path, config: TrainingConfig) -> dict:
    import mlflow

    seed_everything(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True); mlflow_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(input_path, sep="\t", usecols=["Reference_Sequence", "Alternate_Sequence", "Label", "Split"])
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    audit = audit_tokenizer(frame, tokenizer, config.audit_samples, config.seed)
    (output_dir / "tokenizer_audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    pretrained = AutoModelForMaskedLM.from_pretrained(
        config.model_name, trust_remote_code=True
    )
    # This checkpoint's remote EsmConfig is registered for masked-LM loading,
    # not generic AutoModel loading. Retain only its underlying EsmModel encoder.
    encoder = pretrained.base_model
    hidden_size = pretrained.config.hidden_size
    model = PairedSequenceClassifier(
        encoder,
        hidden_size,
        pooling=config.pooling,
        conservation_feature_count=config.conservation_feature_count,
    )
    if config.freeze_encoder:
        for parameter in model.encoder.parameters(): parameter.requires_grad = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Phase 4 training requires a CUDA GPU; use the supplied Colab runner")
    model.to(device)

    subsets = {name: frame.loc[frame.Split == name].reset_index(drop=True) for name in ("train", "validation", "test")}
    collator = PairCollator(tokenizer)
    loaders = {
        name: DataLoader(VariantPairDataset(data), batch_size=config.batch_size, shuffle=name == "train", collate_fn=collator, num_workers=2, pin_memory=True)
        for name, data in subsets.items()
    }
    weights = class_weights(subsets["train"].Label).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    checkpoint = output_dir / "best_model.pt"
    mlflow.set_tracking_uri(mlflow_dir.resolve().as_uri()); mlflow.set_experiment("phase4-nucleotide-transformer")
    best_auprc, stale_epochs, history = -1.0, 0, []
    with mlflow.start_run(run_name="paired_nt_full" if not config.freeze_encoder else "paired_nt_frozen") as active_run:
        mlflow.log_params({**asdict(config), "architecture": "shared_encoder_variant_token_concat_ref_alt_delta", "variant_token_index": VARIANT_TOKEN_INDEX, "split_version": "gene-split-v1", "device": str(device), "class_weights": weights.cpu().tolist()})
        for epoch in range(1, config.epochs + 1):
            loss = train_epoch(model, loaders["train"], optimizer, criterion, device)
            validation = evaluate(model, loaders["validation"], device)
            history.append({"epoch": epoch, "train_loss": loss, "validation": validation})
            mlflow.log_metrics({"train_loss": loss, **flatten("validation", validation)}, step=epoch)
            if validation["auprc"] > best_auprc:
                best_auprc, stale_epochs = validation["auprc"], 0
                torch.save({"model_state_dict": model.state_dict(), "config": asdict(config), "hidden_size": hidden_size}, checkpoint)
            else:
                stale_epochs += 1
                if stale_epochs > config.patience: break
        saved = torch.load(checkpoint, map_location=device)
        model.load_state_dict(saved["model_state_dict"])
        validation, test = evaluate(model, loaders["validation"], device), evaluate(model, loaders["test"], device)
        final_metrics = {**flatten("final_validation", validation), **flatten("test", test)}
        mlflow.log_metrics(final_metrics); mlflow.log_artifact(str(checkpoint), artifact_path="checkpoints"); mlflow.log_artifact(str(output_dir / "tokenizer_audit.json"), artifact_path="audits")
        run_id = active_run.info.run_id
    report = {
        "model_name": config.model_name, "architecture": "shared encoder; contextual embeddings at mutation-containing token 43 for ref, alt, and alt-minus-ref concatenated into an MLP classifier",
        "training_approach": "frozen_encoder_plus_head" if config.freeze_encoder else "full_fine_tuning",
        "split_version": "gene-split-v1", "class_weighted_loss": weights.cpu().tolist(),
        "tokenizer_audit": audit, "history": history, "validation": validation, "test": test,
        "baseline": {"validation_auprc": BASELINE_VALIDATION_AUPRC, "test_auprc": BASELINE_TEST_AUPRC},
        "improvement": {"validation_auprc": validation["auprc"] - BASELINE_VALIDATION_AUPRC, "test_auprc": test["auprc"] - BASELINE_TEST_AUPRC},
        "mlflow_run_id": run_id, "checkpoint": str(checkpoint),
    }
    (output_dir / "phase4_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def run_audit(input_path: Path, output_path: Path, sample_size: int = 100) -> dict:
    frame = pd.read_csv(
        input_path, sep="\t", usecols=["Reference_Sequence", "Alternate_Sequence"]
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    audit = audit_tokenizer(frame, tokenizer, sample_size=sample_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--mlflow-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16); parser.add_argument("--learning-rate", type=float, default=2e-5); parser.add_argument("--epochs", type=int, default=3); parser.add_argument("--patience", type=int, default=1); parser.add_argument("--audit-samples", type=int, default=100); parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--audit-only", action="store_true", help="Run tokenizer audit without loading the model or requiring a GPU")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.audit_only:
        result = run_audit(args.input, args.output_dir / "tokenizer_audit.json", args.audit_samples)
    else:
        config = TrainingConfig(batch_size=args.batch_size, learning_rate=args.learning_rate, epochs=args.epochs, patience=args.patience, freeze_encoder=args.freeze_encoder, audit_samples=args.audit_samples)
        result = run(args.input, args.output_dir, args.mlflow_dir, config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
