"""Lab 4: compare Arabic-centric checkpoints by all/Gulf/MSA slices."""

import argparse
import inspect
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


CHECKPOINTS = {
    "CAMeLBERT-mix": "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    "CAMeLBERT-DA": "CAMeL-Lab/bert-base-arabic-camelbert-da",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="artifacts/arabic_bakeoff",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=2.0,
    )
    return parser.parse_args()


def prepare_data():
    dataframe = pd.read_csv("data/raw/bayan_feedback.csv")

    arabic = dataframe[
        dataframe["lang"].eq("ar")
    ].dropna(
        subset=[
            "text",
            "topic",
            "dialect_region",
            "citizen_group_id",
        ]
    ).copy()

    label_names = sorted(arabic["topic"].unique())
    label_to_id = {
        label: index
        for index, label in enumerate(label_names)
    }

    arabic["labels"] = arabic["topic"].map(label_to_id)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=42,
    )

    train_indices, evaluation_indices = next(
        splitter.split(
            arabic,
            groups=arabic["citizen_group_id"],
        )
    )

    train_data = arabic.iloc[train_indices].copy()
    evaluation_data = arabic.iloc[evaluation_indices].copy()

    overlap = (
        set(train_data["citizen_group_id"])
        & set(evaluation_data["citizen_group_id"])
    )

    if overlap:
        raise RuntimeError(
            "Citizen-group leakage detected"
        )

    print("Arabic rows:", len(arabic))
    print("Train rows:", len(train_data))
    print("Evaluation rows:", len(evaluation_data))
    print("Citizen-group overlap:", len(overlap))
    print()
    print("Evaluation dialect distribution:")
    print(
        evaluation_data[
            "dialect_region"
        ].value_counts().to_string()
    )

    return (
        train_data.reset_index(drop=True),
        evaluation_data.reset_index(drop=True),
        label_names,
    )


def calculate_fertility(tokenizer, texts):
    total_words = 0
    total_tokens = 0

    for text in texts:
        total_words += max(1, len(text.split()))
        token_ids = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        total_tokens += len(token_ids)

    return total_tokens / total_words


def train_and_evaluate(
    display_name,
    checkpoint,
    train_data,
    evaluation_data,
    label_names,
    output_root,
    epochs,
):
    print()
    print("=" * 60)
    print("Training:", display_name)
    print("Checkpoint:", checkpoint)

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            checkpoint,
            num_labels=len(label_names),
            id2label={
                index: label
                for index, label
                in enumerate(label_names)
            },
            label2id={
                label: index
                for index, label
                in enumerate(label_names)
            },
            ignore_mismatched_sizes=True,
        )
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=128,
        )

    train_dataset = Dataset.from_pandas(
        train_data[["text", "labels"]],
        preserve_index=False,
    ).map(
        tokenize,
        batched=True,
    )

    evaluation_dataset = Dataset.from_pandas(
        evaluation_data[["text", "labels"]],
        preserve_index=False,
    ).map(
        tokenize,
        batched=True,
    )

    safe_name = display_name.lower().replace("-", "_")
    model_output = output_root / safe_name
    model_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_arguments = TrainingArguments(
        output_dir=str(
            model_output / "checkpoints"
        ),
        num_train_epochs=epochs,
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        save_strategy="no",
        logging_steps=50,
        report_to=[],
        fp16=torch.cuda.is_available(),
        seed=42,
    )

    trainer_arguments = {
        "model": model,
        "args": training_arguments,
        "train_dataset": train_dataset,
        "eval_dataset": evaluation_dataset,
        "data_collator": DataCollatorWithPadding(
            tokenizer=tokenizer
        ),
    }

    trainer_parameters = inspect.signature(
        Trainer.__init__
    ).parameters

    if "processing_class" in trainer_parameters:
        trainer_arguments[
            "processing_class"
        ] = tokenizer
    else:
        trainer_arguments["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_arguments)

    start_time = time.perf_counter()
    trainer.train()
    train_seconds = time.perf_counter() - start_time

    prediction_output = trainer.predict(
        evaluation_dataset
    )

    predictions = np.argmax(
        prediction_output.predictions,
        axis=-1,
    )

    gold_labels = evaluation_data[
        "labels"
    ].to_numpy()

    results = {
        "checkpoint": checkpoint,
        "macro_f1_all": float(
            f1_score(
                gold_labels,
                predictions,
                average="macro",
            )
        ),
        "train_seconds": round(
            train_seconds,
            2,
        ),
        "arabic_fertility": round(
            calculate_fertility(
                tokenizer,
                evaluation_data["text"].tolist(),
            ),
            4,
        ),
    }

    for region in ["Gulf", "MSA"]:
        region_mask = evaluation_data[
            "dialect_region"
        ].eq(region).to_numpy()

        results[
            f"macro_f1_{region.lower()}"
        ] = float(
            f1_score(
                gold_labels[region_mask],
                predictions[region_mask],
                average="macro",
            )
        )

    trainer.save_model(model_output)
    tokenizer.save_pretrained(model_output)

    print()
    print(
        json.dumps(
            results,
            indent=2,
        )
    )

    return results


def main():
    args = parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        train_data,
        evaluation_data,
        label_names,
    ) = prepare_data()

    all_results = {}

    for display_name, checkpoint in CHECKPOINTS.items():
        all_results[display_name] = (
            train_and_evaluate(
                display_name=display_name,
                checkpoint=checkpoint,
                train_data=train_data,
                evaluation_data=evaluation_data,
                label_names=label_names,
                output_root=output_root,
                epochs=args.epochs,
            )
        )

    results_path = output_root / "results.json"
    results_path.write_text(
        json.dumps(
            all_results,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Arabic model bake-off completed.")
    print("Results saved to:", results_path)


if __name__ == "__main__":
    main()