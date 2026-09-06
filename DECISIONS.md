# Decision Records

## tokenizer
- Chosen checkpoint(s): `xlm-roberta-base` (XLM-R).
- Arabic fertility evidence: XLM-R achieved an Arabic fertility of 1.667, with an Arabic p95 sequence length of 21 tokens.
- English fertility evidence: XLM-R achieved an English fertility of 1.433, with an English p95 sequence length of 23 tokens.
- p95 length evidence: XLM-R produced balanced sequence lengths across both languages: 21 tokens for Arabic and 23 tokens for English.
- Operational trade-off / rationale: XLM-R was selected because it provides the best bilingual balance for Bayan. CAMeLBERT performed slightly better on Arabic but was considerably less efficient on English, while DistilBERT performed well on English but poorly on Arabic. XLM-R also achieved a 0.00% Arabic unknown-token rate.

## arabic-model
- Incumbent:
- Candidate:
- All/Gulf/MSA evidence:
- CI-backed verdict:
- Segmentation contract:

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
