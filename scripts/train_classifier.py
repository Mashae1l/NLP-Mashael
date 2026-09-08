"""Lab 3A: fine-tune the Bayan topic classifier."""

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from bayan.models.data import build_topic_dataset


MODEL_NAME = "xlm-roberta-base"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        default="artifacts/topic_classifier",
        help="Directory used to save the trained model.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    return parser.parse_args()


def compute_metrics(evaluation):
    logits, labels = evaluation
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
    }


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading leakage-free grouped dataset...")
    dataset = build_topic_dataset()

    label_names = sorted(
        set(dataset["train"]["topic"])
        | set(dataset["validation"]["topic"])
        | set(dataset["test"]["topic"])
    )

    label2id = {
        label: index
        for index, label in enumerate(label_names)
    }

    id2label = {
        index: label
        for label, index in label2id.items()
    }

    print("Labels:", label_names)
    print("Loading model:", MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_names),
        label2id=label2id,
        id2label=id2label,
    )

    def tokenize_batch(batch):
        encoded = tokenizer(
            batch["text"],
            truncation=True,
            max_length=128,
        )

        encoded["labels"] = [
            label2id[label]
            for label in batch["topic"]
        ]

        return encoded

    tokenized_dataset = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    training_options = {
        "output_dir": str(output_dir),
        "learning_rate": 2e-5,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": 0.01,
        "save_strategy": "epoch",
        "logging_steps": 50,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "save_total_limit": 1,
        "report_to": "none",
        "fp16": torch.cuda.is_available(),
        "seed": 42,
    }

    parameters = inspect.signature(
        TrainingArguments.__init__
    ).parameters

    if "eval_strategy" in parameters:
        training_options["eval_strategy"] = "epoch"
    else:
        training_options["evaluation_strategy"] = "epoch"

    training_args = TrainingArguments(**training_options)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("Evaluating validation split...")
    validation_metrics = trainer.evaluate(
        tokenized_dataset["validation"],
        metric_key_prefix="validation",
    )

    print("Evaluating frozen test split...")
    test_metrics = trainer.evaluate(
        tokenized_dataset["test"],
        metric_key_prefix="test",
    )

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    results = {
        "model": MODEL_NAME,
        "labels": label_names,
        "validation_accuracy": float(
            validation_metrics["validation_accuracy"]
        ),
        "validation_macro_f1": float(
            validation_metrics["validation_macro_f1"]
        ),
        "test_accuracy": float(
            test_metrics["test_accuracy"]
        ),
        "test_macro_f1": float(
            test_metrics["test_macro_f1"]
        ),
    }

    metrics_path = output_dir / "metrics.json"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nTraining completed.")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print("Model saved to:", output_dir)
    print("Metrics saved to:", metrics_path)


if __name__ == "__main__":
    main()