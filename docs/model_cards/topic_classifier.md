# Model Card — Bayan Topic Classifier

## Intended use

Classify Arabic and English citizen feedback into Bayan service topics.

## Artefact / data versions

- Model/checkpoint: XLM-R topic classifier
- Preprocessing version: 1.2.0
- Data version/snapshot: validation_predictions.csv

## Metrics

| Metric | Value |
| --- | --- |
| Aggregate macro-F1 | 0.8333 [0.8287, 0.8373] |
| Arabic macro-F1 | 0.6000 |
| English macro-F1 | 1.0000 |

## Slice metrics

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

## Behavioural tests

| Test type | Passed | Total | Pass rate |
| --- | --- | --- | --- |
| directional | 200 | 200 | 100.0% |
| invariance | 0 | 200 | 0.0% |
| mft | 400 | 400 | 100.0% |

## Known limitations

The supplied validation set contains no Gulf-labelled rows. Arabic park requests are systematically confused with roads, especially for short inputs.

## Contact / owner

Bayan course project team
