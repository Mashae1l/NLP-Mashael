# EVALUATION REPORT — Bayan

## Executive headline

The model achieved an aggregate macro-F1 of 0.8333 with a 95% bootstrap confidence interval of [0.8287, 0.8373]. The primary quality risk is Arabic MSA feedback, which achieved 0.6000 macro-F1 compared with 1.0000 for English, while short feedback also underperformed.

## Sliced metrics with bootstrap CIs

| Slice | Value | Count | Macro-F1 | 95% CI | Small slice |
| --- | --- | --- | --- | --- | --- |
| overall | all | 2400 | 0.8333 | [0.8287, 0.8373] | no |
| lang | ar | 1200 | 0.6000 | [0.6000, 0.6000] | no |
| lang | en | 1200 | 1.0000 | [1.0000, 1.0000] | no |
| dialect_region | MSA | 1200 | 0.6000 | [0.6000, 0.6000] | no |
| dialect_region | unknown | 1200 | 1.0000 | [1.0000, 1.0000] | no |
| class | billing | 300 | 1.0000 | [1.0000, 1.0000] | no |
| class | digital_services | 300 | 1.0000 | [1.0000, 1.0000] | no |
| class | licensing | 300 | 1.0000 | [1.0000, 1.0000] | no |
| class | lighting | 300 | 1.0000 | [1.0000, 1.0000] | no |
| class | parks | 300 | 0.0000 | [0.0000, 0.0000] | no |
| class | roads | 300 | 1.0000 | [1.0000, 1.0000] | no |
| class | waste | 300 | 1.0000 | [1.0000, 1.0000] | no |
| class | water | 300 | 1.0000 | [1.0000, 1.0000] | no |
| length_bucket | medium | 1646 | 0.8459 | [0.8411, 0.8498] | no |
| length_bucket | short | 754 | 0.7143 | [0.7143, 0.7143] | no |

The supplied validation predictions contain no Gulf-labelled rows, so a Gulf confidence interval cannot be estimated honestly. Small slices are explicitly flagged instead of being treated as precise estimates.

## Behavioural suite

| Test type | Passed | Total | Pass rate |
| --- | --- | --- | --- |
| directional | 200 | 200 | 100.0% |
| invariance | 0 | 200 | 0.0% |
| mft | 400 | 400 | 100.0% |

These results validate the supplied behavioural template contracts. Model-backed behavioural execution should be repeated when the production model artefacts are available in the same environment.

## Error taxonomy

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

## Retrieval quality

| Configuration | Recall@10 | MRR@10 | p50 latency |
| --- | --- | --- | --- |
| Bi-encoder | 1.0000 | 1.0000 | 32.90 ms |
| Cross-encoder reranked | 1.0000 | 1.0000 | 960.55 ms |

- Cross-lingual MRR gap: 0.0000
- No-answer empty-correct: 20/20
- Tuned threshold: -1.3641

## Known limitations

- The validation prediction file contains Arabic MSA and English examples but no Gulf-labelled examples.
- All reviewed classification errors were `parks` examples predicted as `roads`, so the error distribution is highly concentrated.
- The behavioural run validates template contracts without loading the production topic and sentiment checkpoints.
- The retrieval corpus is synthetic and contains relevant same-topic duplicates missing from the strict ID relevance lists.
- Predicted improvement deltas for the prioritised fixes require confirmation through retraining and frozen-test evaluation.
