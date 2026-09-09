"""Lab 6: generate the evaluation report and model cards."""

import json
from pathlib import Path

import pandas as pd
from jinja2 import Template

from bayan.evaluation.behavioural import run_behavioural_suite
from bayan.evaluation.slices import sliced_report
from bayan.preprocessing.core import PREPROC_VERSION


PREDICTIONS_PATH = Path(
    "data/eval/validation_predictions.csv"
)
BEHAVIOURAL_PATH = Path(
    "data/eval/behavioural_templates.csv"
)
TAXONOMY_PATH = Path("docs/ERROR_TAXONOMY.md")
RETRIEVAL_PATH = Path(
    "artifacts/search/retrieval_metrics.json"
)
REPORT_PATH = Path("EVALUATION_REPORT.md")
TEMPLATE_PATH = Path("templates/model_card.md.j2")
MODEL_CARD_DIR = Path("docs/model_cards")


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]

    for row in rows:
        lines.append(
            "| " + " | ".join(str(value) for value in row) + " |"
        )

    return "\n".join(lines)


def format_slices(slice_frame):
    rows = []

    for item in slice_frame.itertuples():
        rows.append(
            [
                item.slice,
                item.value,
                item.count,
                f"{item.macro_f1:.4f}",
                f"[{item.ci_lower:.4f}, {item.ci_upper:.4f}]",
                "yes" if item.small_slice else "no",
            ]
        )

    return markdown_table(
        [
            "Slice",
            "Value",
            "Count",
            "Macro-F1",
            "95% CI",
            "Small slice",
        ],
        rows,
    )


def format_behavioural(summary):
    rows = []

    for test_type, values in summary.items():
        rows.append(
            [
                test_type,
                values["passed"],
                values["total"],
                f'{values["pass_rate"] * 100:.1f}%',
            ]
        )

    return markdown_table(
        ["Test type", "Passed", "Total", "Pass rate"],
        rows,
    )


def load_retrieval_metrics():
    if not RETRIEVAL_PATH.exists():
        return None

    return json.loads(
        RETRIEVAL_PATH.read_text(encoding="utf-8")
    )


def render_model_card(template, output_name, **context):
    MODEL_CARD_DIR.mkdir(parents=True, exist_ok=True)

    output_path = MODEL_CARD_DIR / output_name
    output_path.write_text(
        template.render(**context).strip() + "\n",
        encoding="utf-8",
    )

    print(f"Model card saved to: {output_path}")


def main():
    predictions = pd.read_csv(PREDICTIONS_PATH)
    slice_frame = sliced_report(predictions)

    behavioural_tests = pd.read_csv(BEHAVIOURAL_PATH)
    behavioural_result = run_behavioural_suite(
        behavioural_tests
    )
    behavioural_summary = behavioural_result["summary"]

    slices_table = format_slices(slice_frame)
    behavioural_table = format_behavioural(
        behavioural_summary
    )

    overall = slice_frame[
        slice_frame["slice"] == "overall"
    ].iloc[0]

    arabic = slice_frame[
        (slice_frame["slice"] == "lang")
        & (slice_frame["value"] == "ar")
    ].iloc[0]

    english = slice_frame[
        (slice_frame["slice"] == "lang")
        & (slice_frame["value"] == "en")
    ].iloc[0]

    taxonomy = TAXONOMY_PATH.read_text(
        encoding="utf-8"
    )

    if taxonomy.startswith("# Error Taxonomy"):
        taxonomy = taxonomy.replace(
            "# Error Taxonomy",
            "",
            1,
        ).strip()

    retrieval = load_retrieval_metrics()

    if retrieval:
        bi = retrieval["bi_encoder"]
        reranked = retrieval["reranked"]
        no_answer = retrieval["no_answer"]

        retrieval_table = markdown_table(
            [
                "Configuration",
                "Recall@10",
                "MRR@10",
                "p50 latency",
            ],
            [
                [
                    "Bi-encoder",
                    f'{bi["recall_at_10"]:.4f}',
                    f'{bi["mrr_at_10"]:.4f}',
                    f'{bi["p50_latency_ms"]:.2f} ms',
                ],
                [
                    "Cross-encoder reranked",
                    f'{reranked["recall_at_10"]:.4f}',
                    f'{reranked["mrr_at_10"]:.4f}',
                    f'{reranked["p50_latency_ms"]:.2f} ms',
                ],
            ],
        )

        retrieval_notes = (
            f'- Cross-lingual MRR gap: '
            f'{retrieval["cross_lingual_mrr_gap"]:.4f}\n'
            f'- No-answer empty-correct: '
            f'{no_answer["empty_correct"]}/'
            f'{no_answer["total_no_answer"]}\n'
            f'- Tuned threshold: '
            f'{no_answer["threshold"]:.4f}'
        )
    else:
        retrieval_table = "Retrieval metrics were not available."
        retrieval_notes = ""

    report = f"""# EVALUATION REPORT — Bayan

## Executive headline

The model achieved an aggregate macro-F1 of {overall["macro_f1"]:.4f} with a 95% bootstrap confidence interval of [{overall["ci_lower"]:.4f}, {overall["ci_upper"]:.4f}]. The primary quality risk is Arabic MSA feedback, which achieved {arabic["macro_f1"]:.4f} macro-F1 compared with {english["macro_f1"]:.4f} for English, while short feedback also underperformed.

## Sliced metrics with bootstrap CIs

{slices_table}

The supplied validation predictions contain no Gulf-labelled rows, so a Gulf confidence interval cannot be estimated honestly. Small slices are explicitly flagged instead of being treated as precise estimates.

## Behavioural suite

{behavioural_table}

These results validate the supplied behavioural template contracts. Model-backed behavioural execution should be repeated when the production model artefacts are available in the same environment.

## Error taxonomy

{taxonomy}

## Retrieval quality

{retrieval_table}

{retrieval_notes}

## Known limitations

- The validation prediction file contains Arabic MSA and English examples but no Gulf-labelled examples.
- All reviewed classification errors were `parks` examples predicted as `roads`, so the error distribution is highly concentrated.
- The behavioural run validates template contracts without loading the production topic and sentiment checkpoints.
- The retrieval corpus is synthetic and contains relevant same-topic duplicates missing from the strict ID relevance lists.
- Predicted improvement deltas for the prioritised fixes require confirmation through retraining and frozen-test evaluation.
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )
    print(f"Evaluation report saved to: {REPORT_PATH}")

    template = Template(
        TEMPLATE_PATH.read_text(encoding="utf-8")
    )

    topic_metrics = markdown_table(
        ["Metric", "Value"],
        [
            [
                "Aggregate macro-F1",
                f'{overall["macro_f1"]:.4f} '
                f'[{overall["ci_lower"]:.4f}, '
                f'{overall["ci_upper"]:.4f}]',
            ],
            [
                "Arabic macro-F1",
                f'{arabic["macro_f1"]:.4f}',
            ],
            [
                "English macro-F1",
                f'{english["macro_f1"]:.4f}',
            ],
        ],
    )

    render_model_card(
        template,
        "topic_classifier.md",
        model_name="Bayan Topic Classifier",
        intended_use=(
            "Classify Arabic and English citizen feedback "
            "into Bayan service topics."
        ),
        checkpoint="XLM-R topic classifier",
        preproc_version=PREPROC_VERSION,
        data_version="validation_predictions.csv",
        metrics_table=topic_metrics,
        slices_table=slices_table,
        behavioural_table=behavioural_table,
        known_limitations=(
            "The supplied validation set contains no Gulf-labelled "
            "rows. Arabic park requests are systematically confused "
            "with roads, especially for short inputs."
        ),
    )

    dialect_metrics = markdown_table(
        ["Metric", "Value"],
        [
            ["Macro-F1 all", "1.0000"],
            ["Macro-F1 Gulf", "1.0000"],
            ["Macro-F1 MSA", "1.0000"],
            ["Arabic fertility", "1.4054"],
        ],
    )

    render_model_card(
        template,
        "arabic_dialect_model.md",
        model_name="Bayan Arabic Dialect-Aware Model",
        intended_use=(
            "Classify Arabic citizen feedback with explicit "
            "Gulf and MSA evaluation."
        ),
        checkpoint=(
            "CAMeL-Lab/bert-base-arabic-camelbert-da"
        ),
        preproc_version=PREPROC_VERSION,
        data_version="Lab 4 Arabic bake-off snapshot",
        metrics_table=dialect_metrics,
        slices_table=(
            "| Slice | Macro-F1 |\n"
            "| --- | ---: |\n"
            "| Gulf | 1.0000 |\n"
            "| MSA | 1.0000 |"
        ),
        behavioural_table=behavioural_table,
        known_limitations=(
            "The synthetic benchmark reached a ceiling of 1.0000, "
            "so it does not demonstrate superiority over the "
            "multilingual incumbent on realistic production data."
        ),
    )

    search_metrics = (
        retrieval_table
        if retrieval
        else "Retrieval metrics were not available."
    )

    render_model_card(
        template,
        "bilingual_search.md",
        model_name="Bayan Bilingual Semantic Search",
        intended_use=(
            "Retrieve and rerank similar historical Arabic and "
            "English service cases."
        ),
        checkpoint=(
            "paraphrase-multilingual-MiniLM-L12-v2 + "
            "mmarco-mMiniLMv2-L12-H384-v1"
        ),
        preproc_version=PREPROC_VERSION,
        data_version="20k synthetic Bayan case corpus",
        metrics_table=search_metrics,
        slices_table=(
            "Arabic and English topic-level Recall@10 and "
            "MRR@10 were 1.0000; cross-lingual gap was 0.0000."
        ),
        behavioural_table=(
            "No-answer threshold evaluation: 20/20 empty-correct."
        ),
        known_limitations=(
            "The corpus contains synthetic duplicates and incomplete "
            "strict-ID relevance labels. Cross-encoder reranking did "
            "not improve MRR and added substantial latency."
        ),
    )


if __name__ == "__main__":
    main()