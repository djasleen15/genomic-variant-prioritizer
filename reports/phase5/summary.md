# Phase 5 statistical comparison

| Split | Baseline AUPRC (95% CI) | Fine-tuned AUPRC (95% CI) | Paired difference (95% CI) | Excludes zero? |
|---|---:|---:|---:|:---:|
| Validation | 0.0873 (0.0804, 0.0971) | 0.0906 (0.0841, 0.0992) | +0.0033 (-0.0071, +0.0128) | No |
| Test | 0.0753 (0.0704, 0.0825) | 0.0860 (0.0778, 0.0979) | +0.0107 (+0.0007, +0.0226) | Yes |

The point estimates favor the fine-tuned model on both splits. The paired, class-stratified 10,000-iteration bootstrap supports a positive improvement on test, but the validation difference interval crosses zero. The result is therefore directionally favorable, with test-set statistical evidence, but not consistently confirmed across both held-out splits.

Fine-tuned test probabilities are compressed toward the middle of the range (median 0.4075, mean 0.4127, range 0.3029–0.9670). This explains why fixed thresholds behave poorly and indicates a calibration limitation; no post-hoc calibration was performed in Phase 5.

CADD comparison was omitted because no local CADD scores were available and optional data acquisition was not allowed to delay this phase.
