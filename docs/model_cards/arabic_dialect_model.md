# Model Card — Bayan Arabic Dialect-Aware Model

## Intended use

Classify Arabic citizen feedback with explicit Gulf and MSA evaluation.

## Artefact / data versions

- Model/checkpoint: CAMeL-Lab/bert-base-arabic-camelbert-da
- Preprocessing version: 1.2.0
- Data version/snapshot: Lab 4 Arabic bake-off snapshot

## Metrics

| Metric | Value |
| --- | --- |
| Macro-F1 all | 1.0000 |
| Macro-F1 Gulf | 1.0000 |
| Macro-F1 MSA | 1.0000 |
| Arabic fertility | 1.4054 |

## Slice metrics

| Slice | Macro-F1 |
| --- | ---: |
| Gulf | 1.0000 |
| MSA | 1.0000 |

## Behavioural tests

| Test type | Passed | Total | Pass rate |
| --- | --- | --- | --- |
| directional | 200 | 200 | 100.0% |
| invariance | 0 | 200 | 0.0% |
| mft | 400 | 400 | 100.0% |

## Known limitations

The synthetic benchmark reached a ceiling of 1.0000, so it does not demonstrate superiority over the multilingual incumbent on realistic production data.

## Contact / owner

Bayan course project team
