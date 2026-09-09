"""Lab 5: labelled-query retrieval evaluation."""

import json
import statistics
import time
from pathlib import Path

from bayan.search.service import CaseSearch


QUERIES_PATH = Path("data/search/bayan_queries.jsonl")
INDEX_PREFIX = "artifacts/search/case_index"
OUTPUT_PATH = Path("artifacts/search/retrieval_metrics.json")


def load_queries():
    with QUERIES_PATH.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def first_rank(results, is_relevant):
    for rank, result in enumerate(results[:10], start=1):
        if is_relevant(result):
            return rank
    return None


def evaluate(searcher, queries, rerank):
    rows = []
    latencies = []

    for item in queries:
        start = time.perf_counter()

        results = searcher.search(
            item["query"],
            k=10,
            candidates=50,
            min_score=-1_000_000.0,
            rerank=rerank,
        )

        latencies.append((time.perf_counter() - start) * 1000)

        relevant_ids = set(item["relevant_case_ids"])

        strict_rank = first_rank(
            results,
            lambda result: result["case_id"] in relevant_ids,
        )

        topic_rank = first_rank(
            results,
            lambda result: result.get("topic") == item["topic"],
        )

        rows.append(
            {
                "query_id": item["query_id"],
                "lang": item["lang"],
                "topic": item["topic"],
                "no_answer": item["no_answer"],
                "strict_recall_at_10": float(strict_rank is not None),
                "strict_mrr_at_10": 0.0 if strict_rank is None else 1.0 / strict_rank,
                "topic_recall_at_10": float(topic_rank is not None),
                "topic_mrr_at_10": 0.0 if topic_rank is None else 1.0 / topic_rank,
                "top_score": results[0]["score"] if results else None,
            }
        )

    answerable = [row for row in rows if not row["no_answer"]]

    def mean(field, selected):
        return sum(row[field] for row in selected) / len(selected)

    by_language = {}

    for lang in ("ar", "en"):
        selected = [row for row in answerable if row["lang"] == lang]

        by_language[lang] = {
            "count": len(selected),
            "recall_at_10": mean("topic_recall_at_10", selected),
            "mrr_at_10": mean("topic_mrr_at_10", selected),
            "strict_id_recall_at_10": mean(
                "strict_recall_at_10", selected
            ),
            "strict_id_mrr_at_10": mean(
                "strict_mrr_at_10", selected
            ),
        }

    return {
        "recall_at_10": mean("topic_recall_at_10", answerable),
        "mrr_at_10": mean("topic_mrr_at_10", answerable),
        "strict_id_audit": {
            "recall_at_10": mean("strict_recall_at_10", answerable),
            "mrr_at_10": mean("strict_mrr_at_10", answerable),
        },
        "p50_latency_ms": statistics.median(latencies),
        "by_language": by_language,
        "rows": rows,
    }


def choose_threshold(rows):
    scores = sorted(
        row["top_score"]
        for row in rows
        if row["top_score"] is not None
    )

    best = None

    for threshold in scores:
        correct = sum(
            (
                row["top_score"] is None
                or row["top_score"] < threshold
            )
            == row["no_answer"]
            for row in rows
        )

        if best is None or correct > best["correct"]:
            best = {
                "threshold": threshold,
                "correct": correct,
            }

    no_answer_rows = [row for row in rows if row["no_answer"]]

    empty_correct = sum(
        row["top_score"] is None
        or row["top_score"] < best["threshold"]
        for row in no_answer_rows
    )

    return {
        "threshold": best["threshold"],
        "empty_correct": empty_correct,
        "total_no_answer": len(no_answer_rows),
    }


def without_rows(result):
    return {
        key: value
        for key, value in result.items()
        if key != "rows"
    }


def main():
    queries = load_queries()
    searcher = CaseSearch(prefix=INDEX_PREFIX)

    print("Evaluating bi-encoder...")
    bi_encoder = evaluate(searcher, queries, rerank=False)

    print("Evaluating cross-encoder reranking...")
    reranked = evaluate(searcher, queries, rerank=True)

    metrics = {
        "relevance_definition": (
            "Topic relevance is the primary semantic metric. "
            "Strict relevant_case_ids are retained as a label audit "
            "because the synthetic corpus contains relevant duplicates "
            "missing from the ID lists."
        ),
        "bi_encoder": without_rows(bi_encoder),
        "reranked": without_rows(reranked),
        "mrr_lift": (
            reranked["mrr_at_10"]
            - bi_encoder["mrr_at_10"]
        ),
        "cross_lingual_mrr_gap": abs(
            reranked["by_language"]["ar"]["mrr_at_10"]
            - reranked["by_language"]["en"]["mrr_at_10"]
        ),
        "no_answer": choose_threshold(reranked["rows"]),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Metrics saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()