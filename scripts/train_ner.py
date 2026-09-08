"""Lab 3B: fine-tune the Bayan NER model."""

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from bayan.models.ner import align_labels


MODEL_NAME = "xlm-roberta-base"
DATA_PATH = Path("data/models/bayan_ner.conll")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        default="artifacts/ner",
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


def read_conll(path):
    sentences = []
    sentence_tokens = []
    sentence_labels = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                if sentence_tokens:
                    sentences.append(
                        {
                            "tokens": sentence_tokens,
                            "ner_tags": sentence_labels,
                        }
                    )

                    sentence_tokens = []
                    sentence_labels = []

                continue

            token, label = line.rsplit(maxsplit=1)
            sentence_tokens.append(token)
            sentence_labels.append(label)

    if sentence_tokens:
        sentences.append(
            {
                "tokens": sentence_tokens,
                "ner_tags": sentence_labels,
            }
        )

    return sentences


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Reading NER dataset...")
    examples = read_conll(DATA_PATH)

    dataset = Dataset.from_list(examples)

    first_split = dataset.train_test_split(
        test_size=0.20,
        seed=42,
    )

    second_split = first_split["test"].train_test_split(
        test_size=0.50,
        seed=42,
    )

    raw_dataset = DatasetDict(
        {
            "train": first_split["train"],
            "validation": second_split["train"],
            "test": second_split["test"],
        }
    )

    print("Train sentences:", len(raw_dataset["train"]))
    print("Validation sentences:", len(raw_dataset["validation"]))
    print("Test sentences:", len(raw_dataset["test"]))

    label_names = sorted(
        {
            label
            for example in examples
            for label in example["ner_tags"]
        }
    )

    if "O" in label_names:
        label_names.remove("O")
        label_names.insert(0, "O")

    label2id = {
        label: index
        for index, label in enumerate(label_names)
    }

    id2label = {
        index: label
        for label, index in label2id.items()
    }

    print("NER labels:", label_names)
    print("Loading model:", MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_names),
        label2id=label2id,
        id2label=id2label,
    )

    def tokenize_and_align(batch):
        tokenized = tokenizer(
            batch["tokens"],
            truncation=True,
            max_length=128,
            is_split_into_words=True,
        )

        aligned_batch = []

        for batch_index, labels in enumerate(batch["ner_tags"]):
            word_ids = tokenized.word_ids(
                batch_index=batch_index
            )

            numeric_labels = [
                label2id[label]
                for label in labels
            ]

            aligned_batch.append(
                align_labels(
                    word_ids,
                    numeric_labels,
                )
            )

        tokenized["labels"] = aligned_batch
        return tokenized

    tokenized_dataset = raw_dataset.map(
        tokenize_and_align,
        batched=True,
        remove_columns=raw_dataset["train"].column_names,
    )

    def compute_metrics(evaluation):
        logits, labels = evaluation
        predictions = np.argmax(logits, axis=-1)

        true_predictions = []
        true_labels = []

        for prediction_row, label_row in zip(
            predictions,
            labels,
        ):
            predicted_tags = []
            correct_tags = []

            for prediction, label in zip(
                prediction_row,
                label_row,
            ):
                if label == -100:
                    continue

                predicted_tags.append(
                    id2label[int(prediction)]
                )

                correct_tags.append(
                    id2label[int(label)]
                )

            true_predictions.append(predicted_tags)
            true_labels.append(correct_tags)

        return {
            "precision": precision_score(
                true_labels,
                true_predictions,
                zero_division=0,
            ),
            "recall": recall_score(
                true_labels,
                true_predictions,
                zero_division=0,
            ),
            "f1": f1_score(
                true_labels,
                true_predictions,
                zero_division=0,
            ),
            "accuracy": accuracy_score(
                true_labels,
                true_predictions,
            ),
        }

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
        "metric_for_best_model": "f1",
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

    training_args = TrainingArguments(
        **training_options
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(
            tokenizer=tokenizer
        ),
        compute_metrics=compute_metrics,
    )

    print("Starting NER training...")
    trainer.train()

    validation_metrics = trainer.evaluate(
        tokenized_dataset["validation"],
        metric_key_prefix="validation",
    )

    test_metrics = trainer.evaluate(
        tokenized_dataset["test"],
        metric_key_prefix="test",
    )

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    results = {
        "model": MODEL_NAME,
        "labels": label_names,
        "validation_entity_f1": float(
            validation_metrics["validation_f1"]
        ),
        "test_entity_f1": float(
            test_metrics["test_f1"]
        ),
        "test_precision": float(
            test_metrics["test_precision"]
        ),
        "test_recall": float(
            test_metrics["test_recall"]
        ),
        "test_accuracy": float(
            test_metrics["test_accuracy"]
        ),
    }

    metrics_path = output_dir / "metrics.json"

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nNER training completed.")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print("Model saved to:", output_dir)
    print("Metrics saved to:", metrics_path)


if __name__ == "__main__":
    main()