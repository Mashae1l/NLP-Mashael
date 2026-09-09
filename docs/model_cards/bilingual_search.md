# Model Card — Bayan Bilingual Semantic Search

## Intended use

Retrieve and rerank similar historical Arabic and English service cases.

## Artefact / data versions

- Model/checkpoint: paraphrase-multilingual-MiniLM-L12-v2 + mmarco-mMiniLMv2-L12-H384-v1
- Preprocessing version: 1.2.0
- Data version/snapshot: 20k synthetic Bayan case corpus

## Metrics

| Configuration | Recall@10 | MRR@10 | p50 latency |
| --- | --- | --- | --- |
| Bi-encoder | 1.0000 | 1.0000 | 32.90 ms |
| Cross-encoder reranked | 1.0000 | 1.0000 | 960.55 ms |

## Slice metrics

Arabic and English topic-level Recall@10 and MRR@10 were 1.0000; cross-lingual gap was 0.0000.

## Behavioural tests

No-answer threshold evaluation: 20/20 empty-correct.

## Known limitations

The corpus contains synthetic duplicates and incomplete strict-ID relevance labels. Cross-encoder reranking did not improve MRR and added substantial latency.

## Contact / owner

Bayan course project team
