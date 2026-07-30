"""
Phase 0 sanity check: confirm we can reach HuggingFace Hub and pull the
smallest Nucleotide Transformer checkpoint before depending on it later.

Run this BEFORE starting Phase 1. If this fails, fix your network/access
situation first (e.g. corporate/university firewall blocking HF, or you
need to run `huggingface-cli login` if the model is gated).

Usage:
    python scripts/check_hf_access.py
"""

from transformers import AutoTokenizer, AutoModelForMaskedLM

MODEL_NAME = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"


def main():
    print(f"Attempting to download tokenizer + model: {MODEL_NAME}")
    print("(smallest Nucleotide Transformer v2 variant, ~50M params, "
          "expect a few hundred MB download)\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print("Tokenizer downloaded OK.")

    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print("Model downloaded OK.")

    # Smoke test: tokenize and embed a dummy DNA sequence
    dummy_seq = "ATCGATCGATCGATCGATCGATCG"
    tokens = tokenizer(dummy_seq, return_tensors="pt")
    print(f"\nTokenized dummy sequence into {tokens['input_ids'].shape[1]} tokens.")
    print("Token IDs:", tokens["input_ids"])

    outputs = model(**tokens, output_hidden_states=True)
    print(f"\nForward pass OK. Last hidden state shape: "
          f"{outputs.hidden_states[-1].shape}")

    print("\n✅ Phase 0 HuggingFace check PASSED. Safe to proceed to Phase 1.")


if __name__ == "__main__":
    main()
