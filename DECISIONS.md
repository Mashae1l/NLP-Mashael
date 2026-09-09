# Decision Records

## tokenizer
- Chosen checkpoint(s): `xlm-roberta-base` (XLM-R).
- Arabic fertility evidence: XLM-R achieved an Arabic fertility of 1.667, with an Arabic p95 sequence length of 21 tokens.
- English fertility evidence: XLM-R achieved an English fertility of 1.433, with an English p95 sequence length of 23 tokens.
- p95 length evidence: XLM-R produced balanced sequence lengths across both languages: 21 tokens for Arabic and 23 tokens for English.
- Operational trade-off / rationale: XLM-R was selected because it provides the best bilingual balance for Bayan. CAMeLBERT performed slightly better on Arabic but was considerably less efficient on English, while DistilBERT performed well on English but poorly on Arabic. XLM-R also achieved a 0.00% Arabic unknown-token rate.

## arabic-model

- Incumbent: CAMeLBERT-mix.
- Candidate: CAMeLBERT-DA.
- All/Gulf/MSA evidence: Both checkpoints achieved 1.0000 macro-F1 on all, Gulf, and MSA held-out slices.
- Gulf delta: +0.0000 macro-F1 for CAMeLBERT-DA over CAMeLBERT-mix.
- Verdict: Retain CAMeLBERT-mix because the dialect-aware candidate produced no measurable quality improvement; keep CAMeLBERT-DA as an evaluated alternative for Lab 6 sliced reporting.
- Ceiling effect: The incumbent already scored 1.0000 on the Gulf slice, making the requested +4-point improvement impossible on this dataset.
- Segmentation contract: CAMeL Tools d3tok was applied consistently; LOCATION recall remained 1.0000 before and after segmentation.

## search-min-score
- Threshold:
- No-answer evidence:
- False-positive / false-negative trade-off:

## quantisation-split
- Topic artefact:
- NER artefact:
- Latency evidence:
- Paired quality-tax evidence:
- Rollback artefact retained:

## architecture
- Encoder/decoder rationale by task:
- Multilingual vs Arabic-centric rationale:
- Evidence used:
