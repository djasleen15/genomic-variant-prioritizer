from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch import nn

from src.models.finetune_lm import (
    PairCollator,
    PairedSequenceClassifier,
    audit_tokenizer,
    class_weights,
    covering_token,
    metrics,
)


class SixMerTokenizer:
    @staticmethod
    def dna_tokens(sequence):
        full_length = len(sequence) - (len(sequence) % 6)
        tokens = [sequence[i : i + 6] for i in range(0, full_length, 6)]
        tokens.extend(sequence[full_length:])
        return tokens

    def __call__(self, sequences, add_special_tokens=True, **kwargs):
        if isinstance(sequences, str):
            tokens = ["<cls>"] + self.dna_tokens(sequences)
            return {"input_ids": list(range(len(tokens)))}
        width = max(1 + len(self.dna_tokens(sequence)) for sequence in sequences)
        ids, masks = [], []
        for sequence in sequences:
            length = 1 + len(self.dna_tokens(sequence))
            ids.append(list(range(length)) + [0] * (width - length))
            masks.append([1] * length + [0] * (width - length))
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)}

    def convert_ids_to_tokens(self, input_ids):
        sequence = "A" * 512
        return ["<cls>"] + self.dna_tokens(sequence)


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__(); self.embedding = nn.Embedding(100, 4)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def test_variant_position_is_inside_expected_nonoverlapping_sixmer():
    token, token_count = covering_token(SixMerTokenizer(), "A" * 512, 256)
    assert token == {"token_index": 43, "token": "AAAAAA", "start": 252, "end": 258}
    assert token_count == 88


def test_tokenizer_audit_requires_ref_alt_alignment():
    frame = pd.DataFrame({"Reference_Sequence": ["A" * 512], "Alternate_Sequence": ["A" * 256 + "C" + "A" * 255]})
    result = audit_tokenizer(frame, SixMerTokenizer(), sample_size=1)
    assert result["all_pairs_same_token_count"] is True
    assert result["all_pairs_same_boundary_alignment"] is True
    assert result["unique_covering_spans"] == ["252:258"]
    assert result["unique_variant_offsets_within_token"] == [4]


def test_pair_collator_keeps_ref_alt_and_labels_separate():
    batch = PairCollator(SixMerTokenizer())([("AAAAAA", "AAAACA", 1), ("CCCCCC", "CCCTCC", 0)])
    assert batch["ref_input_ids"].shape == batch["alt_input_ids"].shape == (2, 2)
    assert batch["labels"].tolist() == [1, 0]


def test_classifier_combines_shared_ref_alt_embeddings():
    model = PairedSequenceClassifier(TinyEncoder(), hidden_size=4, dropout=0)
    token_ids = torch.arange(44).repeat(2, 1)
    attention_mask = torch.ones((2, 44), dtype=torch.long)
    batch = {key: token_ids for key in ("ref_input_ids", "alt_input_ids")}
    batch.update({key: attention_mask for key in ("ref_attention_mask", "alt_attention_mask")})
    assert model(batch).shape == (2, 2)


def test_class_weights_balance_binary_training_labels():
    weights = class_weights(pd.Series([0] * 9 + [1]))
    assert weights.tolist() == pytest.approx([10 / 18, 5.0])


def test_metrics_match_phase3_threshold_contract():
    result = metrics([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9])
    assert result["auprc"] == 1.0
    assert set(result["thresholds"]) == {"0.25", "0.5", "0.75"}
