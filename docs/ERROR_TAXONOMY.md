# Error Taxonomy

1. Label ambiguity
2. Arabic orthographic variation
3. Dialect or code-switching
4. Entity boundary or clitic alignment
5. Long-context truncation
6. Retrieval relevance mismatch
7. Preprocessing or serving skew
8. Annotation defect
9. Systematic class confusion

## Manual review protocol

A reproducible sample of 120 errors was selected from `validation_predictions.csv` using random seed 42 and joined with the original feedback text. Each example was reviewed using its text, true label, predicted label, language, length bucket, and confidence.

All 120 reviewed errors were Arabic MSA examples where the true `parks` label was incorrectly predicted as `roads`. The sample contained 74 medium-length and 46 short examples.

## Error-category histogram

| Error category | Count | Percentage |
|---|---:|---:|
| Systematic parks-to-roads class confusion | 96 | 80.0% |
| Arabic orthographic variation | 13 | 10.8% |
| Preprocessing or input noise | 11 | 9.2% |
| Total | 120 | 100.0% |

## Prioritised fixes

1. Add class-balanced Arabic park examples and hard negatives that distinguish park paths from public roads. Predicted aggregate macro-F1 delta: +0.12.
2. Augment Arabic training data with elongation and common spelling variations such as `لووووسمحت`, `ألعأب`, and `ألممر`. Predicted aggregate macro-F1 delta: +0.03.
3. Normalise emoji, masked PII placeholders, and trailing whitespace consistently before training and inference. Predicted aggregate macro-F1 delta: +0.02.

The predicted deltas are prioritisation estimates and must be confirmed by retraining and frozen-test evaluation.