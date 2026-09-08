"""Lab 3A: dataset construction and split integrity."""

from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict

from bayan.preprocessing.core import preprocess


DATA_PATH = Path("data/raw/bayan_feedback.csv")


def build_topic_dataset(data_path=DATA_PATH):
    """Build leakage-free train, validation and test datasets."""

    data = pd.read_csv(data_path)

    required_columns = {
        "text",
        "topic",
        "split",
        "citizen_group_id",
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    data["text"] = (
        data["text"]
        .fillna("")
        .astype(str)
        .map(preprocess)
    )

    split_datasets = {}

    for split_name in ["train", "validation", "test"]:
        split_data = data[
            data["split"] == split_name
        ].reset_index(drop=True)

        split_datasets[split_name] = Dataset.from_pandas(
            split_data,
            preserve_index=False,
        )

    dataset = DatasetDict(split_datasets)

    train_groups = set(
        dataset["train"]["citizen_group_id"]
    )

    validation_groups = set(
        dataset["validation"]["citizen_group_id"]
    )

    test_groups = set(
        dataset["test"]["citizen_group_id"]
    )

    train_validation_overlap = (
        train_groups & validation_groups
    )

    train_test_overlap = (
        train_groups & test_groups
    )

    validation_test_overlap = (
        validation_groups & test_groups
    )

    if train_validation_overlap:
        raise ValueError(
            "Citizen leakage between train and validation"
        )

    if train_test_overlap:
        raise ValueError(
            "Citizen leakage between train and test"
        )

    if validation_test_overlap:
        raise ValueError(
            "Citizen leakage between validation and test"
        )

    print(f"Train rows: {len(dataset['train'])}")
    print(f"Validation rows: {len(dataset['validation'])}")
    print(f"Test rows: {len(dataset['test'])}")

    print(
        "Train/validation citizen overlap:",
        len(train_validation_overlap),
    )

    print(
        "Train/test citizen overlap:",
        len(train_test_overlap),
    )

    print(
        "Validation/test citizen overlap:",
        len(validation_test_overlap),
    )

    return dataset