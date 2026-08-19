# Phase 8: attention diagnostics

This directory records a small interpretability diagnostic for the original,
approved Phase 4 `best_model.pt` checkpoint. It is deliberately scoped to a
handful of deterministically selected `gene-split-v1` test examples: three
correct drivers, three correct passengers, and up to two incorrect predictions.
Correct examples are spaced across prediction-score quantiles within each class;
errors are selected by confidence. Selection occurs before attention inspection
to reduce the opportunity for visual cherry-picking.

Each plot shows final-layer attention from the mutation-containing query token
(token index 43, covering the mutation at zero-based base index 256) to the DNA
tokens in the 512 bp reference and alternate windows, averaged across attention
heads. The yellow region marks the mutation-containing token.

## Essential limitation

**These attention weights are a diagnostic aid, not a definitive causal
explanation.** They show where the model allocated attention, but do not
establish which bases caused the prediction or prove that the model understands
the underlying biology. Sequence attribution for genomic transformers remains
an unsettled research area. Attention patterns can be useful for debugging and
hypothesis generation, but they must not be presented as mechanistic evidence.

The generated artifacts are:

- `phase8_report.json`: method, checkpoint digest, selection policy, examples,
  and limitation statement;
- `selected_examples.csv`: selected row IDs, labels, predictions, and categories;
- `attention_weights.csv`: compact token-position/weight data for every plot
  (DNA token strings are deliberately omitted to avoid redistributing derived
  cohort sequence); and
- `attention_*.png`: static diagnostics, including canonical driver and passenger
  examples linked from the main README.

Regenerate on a CUDA runtime with the private fixed dataset and approved Phase 4
checkpoint:

```bash
python -m src.evaluation.phase8 \
  --dataset /private/path/labeled_split_dataset.tsv \
  --checkpoint /private/path/best_model.pt \
  --predictions /private/path/fine_tuned_predictions.tsv \
  --output-dir reports/phase8
```
