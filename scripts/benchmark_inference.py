"""Lab 7: honest CPU p50/p99 inference benchmark."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)


DEFAULT_CLASSIFIER = os.environ.get(
    "BAYAN_CLASSIFIER_DIR",
    "artifacts/topic_classifier",
)
DEFAULT_NER = os.environ.get(
    "BAYAN_NER_DIR",
    "artifacts/ner",
)


def benchmark(
    model,
    tokenizer,
    texts,
    *,
    max_length,
    padding,
    threads=4,
    warmup=10,
    runs=200,
):
    """Measure end-to-end single-example CPU latency."""
    torch.set_num_threads(threads)
    model.to("cpu")
    model.eval()

    selected = list(texts[:runs])

    with torch.inference_mode():
        for text in selected[:warmup]:
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=padding,
            )
            model(**encoded)

        latencies_ms = []

        for text in selected:
            start = time.perf_counter()

            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=padding,
            )
            model(**encoded)

            latencies_ms.append(
                (time.perf_counter() - start) * 1000
            )

    return {
        "threads": threads,
        "runs": len(selected),
        "max_length": max_length,
        "padding": str(padding),
        "p50_ms": float(np.percentile(latencies_ms, 50)),
        "p99_ms": float(np.percentile(latencies_ms, 99)),
    }


def load_model(task, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if task == "classifier":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path
        )
    elif task == "ner":
        model = AutoModelForTokenClassification.from_pretrained(
            model_path
        )
    else:
        raise ValueError(f"Unsupported task: {task}")

    return model, tokenizer


def run_task(task, model_path, texts, threads, runs):
    print(f"Loading {task}: {model_path}")
    model, tokenizer = load_model(task, model_path)

    print(f"Benchmarking {task} fp32 @512 padded...")
    baseline = benchmark(
        model,
        tokenizer,
        texts,
        max_length=512,
        padding="max_length",
        threads=threads,
        runs=runs,
    )
    baseline["task"] = task
    baseline["configuration"] = "fp32_torch_512_padded"

    print(f"Benchmarking {task} fp32 @128 dynamic...")
    dynamic = benchmark(
        model,
        tokenizer,
        texts,
        max_length=128,
        padding=True,
        threads=threads,
        runs=runs,
    )
    dynamic["task"] = task
    dynamic["configuration"] = "fp32_torch_128_dynamic"

    dynamic["speedup_vs_baseline"] = (
        baseline["p99_ms"] / dynamic["p99_ms"]
    )

    return [baseline, dynamic]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=["classifier", "ner", "both"],
        default="classifier",
    )
    parser.add_argument(
        "--classifier-dir",
        default=DEFAULT_CLASSIFIER,
    )
    parser.add_argument(
        "--ner-dir",
        default=DEFAULT_NER,
    )
    parser.add_argument(
        "--bench-data",
        default="data/serving/bench_mix.npy",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument(
        "--output",
        default="artifacts/serving/benchmark_fp32.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    texts = np.load(
        args.bench_data,
        allow_pickle=False,
    )

    results = []

    if args.task in {"classifier", "both"}:
        results.extend(
            run_task(
                "classifier",
                args.classifier_dir,
                texts,
                args.threads,
                args.runs,
            )
        )

    if args.task in {"ner", "both"}:
        results.extend(
            run_task(
                "ner",
                args.ner_dir,
                texts,
                args.threads,
                args.runs,
            )
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(results, indent=2))
    print(f"Benchmark saved to: {output_path}")


if __name__ == "__main__":
    main()