# BENCHMARKS

> Fill these tables from **your own runs**. Do not copy course reference numbers.

## Lab 1 — Tokenizer audit

| Tokenizer | AR fertility | EN fertility | AR p95 len | EN p95 len | AR UNK rate |
|---|---:|---:|---:|---:|---:|
| mBERT | 2.143 | 1.509 | 27.0 | 25.0 | 0.46% |
| XLM-R | 1.667 | 1.433 | 21.0 | 23.0 | 0.00% |
| CAMeLBERT | 1.400 | 2.704 | 20.0 | 38.0 | 0.81% |
| DistilBERT | 4.517 | 1.297 | 47.0 | 21.0 | 0.22% |

- Golden preprocessing: 25 / 25 passed
- PII masking recall: 60 / 60 = 100%
- Sequence-length histogram: `artifacts/lab1_tokenizer_lengths.png`

## Lab 2 — Attention diagnostics

| Check | Result |
|---|---:|
| Numerical equivalence | True |
| Maximum difference | 2.38e-7 |
| Adjacent-token attention mass | 0.3136 |
| SEP sink mass | 0.2136 |
| Pad mass without mask | 0.1729 |
| Pad mass with mask | 0.0000 |
| Causal future-attention mass | 0.0000 |

## Lab 3 — Models
| Model | Metric | Validation | Frozen test | Train time |
|---|---|---:|---:|---:|
| TF-IDF + LinearSVC | macro-F1 | 1.0000 | 1.0000 | CPU |
| Topic classifier (XLM-R) | macro-F1 | 1.0000 | 1.0000 | Colab T4, 2 epochs |
| NER (XLM-R) | entity-F1 | 1.0000 | 1.0000 | 109.5 s, Colab T4 |
| QA | span/null smoke | 9/9 answerable | 3/3 null | CPU |

- Topic-classifier frozen-test improvement over baseline: +0.0000.
- Ceiling-effect note: The TF-IDF baseline already achieved 1.0000 macro-F1, so improving it by +0.08 was mathematically impossible on this dataset.
- NER alignment contract: 8/8 passed.
- QA post-processing contract: 2/2 passed.
- QA smoke result: 9/9 answerable and 3/3 unanswerable passed.

## Lab 4 — Arabic model bake-off
|| CAMeLBERT-mix (multilingual incumbent) | 1.0000 | 1.0000 | 1.0000 | 1.4054 |
| CAMeLBERT-DA (Arabic dialect-aware) | 1.0000 | 1.0000 | 1.0000 | 1.4054 |
| Optional third model | Not run | Not run | Not run | Not run |

- Arabic normalisation golden pairs: 30 / 30 passed.
- Clitic segmentation scheme: CAMeL Tools d3tok.
- NER LOCATION recall before segmentation: 1.0000.
- NER LOCATION recall after segmentation: 1.0000.
- LOCATION recall delta: +0.0000.
- Segmented NER train time: 152.38 s, Colab T4.
- Ceiling-effect note: The original NER model already achieved 1.0000 LOCATION recall, so a +4-point improvement was mathematically impossible on this dataset.
- CAMeLBERT-mix train time: 63.19 s, Colab T4.
- CAMeLBERT-DA train time: 56.23 s, Colab T4.
- Gulf macro-F1 delta (DA minus mix): +0.0000.
- Ceiling-effect note: CAMeLBERT-mix already achieved 1.0000 Gulf macro-F1, so a +4-point improvement was mathematically impossible on this held-out dataset.

## Lab 5 — Search

| Configuration | recall@10 | MRR@10 | p50 latency/query |
|---|---:|---:|---:|
| bi-encoder only | 1.0000 | 1.0000 | 32.90 ms |
| + cross-encoder rerank | 1.0000 | 1.0000 | 960.55 ms |
| cross-lingual slice | 1.0000 | 1.0000 | — |

- no-answer empty-correct: 20 / 20
- tuned no-answer threshold: -1.3641
- cross-lingual MRR gap: 0.0000
- reranking MRR lift: +0.0000; the bi-encoder already reached the topic-level ceiling, while reranking added substantial latency.
- Strict-ID audit: recall@10 = 0.0077 and MRR@10 = 0.0026 for the bi-encoder; the synthetic corpus contains relevant same-topic duplicates that are absent from `relevant_case_ids`.
- Unnormalised-vector check: topic recall@10 and MRR@10 remained 1.0000 in this run, so no ranking collapse was observed. L2 normalisation is still enforced and recorded in the manifest because raw inner-product scores are not reliably calibrated across models or datasets.

## Lab 6 — Evaluation
| Model | Aggregate macro-F1 [CI] | Gulf [CI] | Invariance pass | MFT pass |
|---|---|---|---:|---:|
| topic classifier | 0.8333 [0.8287, 0.8373] | N/A (no Gulf rows) | 100% (200/200) | 100% (400/400) |
| dialect-aware | | | | |

- paired comparison verdict: A paired language comparison is not valid because the Arabic and English examples are not paired. The observed Arabic-English macro-F1 gap is 0.4000, and the supplied validation predictions contain no Gulf-labelled rows.
- error taxonomy top categories:
- top-3 prioritised fixes:

## Lab 7 — Optimisation ladder
| Rung | p50 | p99 | quality metric / paired Δ | Artefact size |
|---|---:|---:|---|---:|
| fp32 torch @512 padded | | | | |
| fp32 torch @128 dynamic | | | | |
| ONNX fp32 @128 | | | | |
| ONNX INT8 @128 | | | | |

- HTTP p99, 16 concurrent:
- classifier quantisation decision:
- NER quantisation decision:
