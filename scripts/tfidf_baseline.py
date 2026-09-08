"""Lab 3A: TF-IDF and LinearSVC baseline."""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from bayan.preprocessing.core import preprocess


DATA_PATH = Path("data/raw/bayan_feedback.csv")


def evaluate(model, data, split_name):
    texts = data["text"].tolist()
    labels = data["topic"].tolist()

    predictions = model.predict(texts)

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    accuracy = accuracy_score(labels, predictions)

    print(f"\n{split_name} results")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    return macro_f1, predictions


def main():
    data = pd.read_csv(DATA_PATH)

    data["text"] = (
        data["text"]
        .fillna("")
        .astype(str)
        .map(preprocess)
    )

    train_data = data[data["split"] == "train"]
    validation_data = data[data["split"] == "validation"]
    test_data = data[data["split"] == "test"]

    print(f"Train rows: {len(train_data)}")
    print(f"Validation rows: {len(validation_data)}")
    print(f"Test rows: {len(test_data)}")

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=30000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LinearSVC(
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    print("\nTraining TF-IDF + LinearSVC...")
    model.fit(
        train_data["text"],
        train_data["topic"],
    )

    validation_f1, _ = evaluate(
        model,
        validation_data,
        "Validation",
    )

    test_f1, test_predictions = evaluate(
        model,
        test_data,
        "Frozen test",
    )

    print("\nFrozen-test classification report:")
    print(
        classification_report(
            test_data["topic"],
            test_predictions,
            zero_division=0,
        )
    )

    print("\nBaseline summary")
    print(f"Validation macro-F1: {validation_f1:.4f}")
    print(f"Frozen-test macro-F1: {test_f1:.4f}")


if __name__ == "__main__":
    main()
