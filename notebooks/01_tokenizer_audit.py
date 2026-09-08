"""Audit four tokenizer candidates on Bayan Arabic and English text."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from bayan.preprocessing.core import preprocess


CANDIDATES = {
    "bert-base-multilingual-cased": "mBERT",
    "xlm-roberta-base": "XLM-R",
    "CAMeL-Lab/bert-base-arabic-camelbert-mix": "CAMeLBERT",
    "distilbert-base-uncased": "DistilBERT",
}

DATA = Path("data/raw/bayan_feedback.csv")
HISTOGRAM = Path("artifacts/lab1_tokenizer_lengths.png")


def fertility(tokenizer, texts) -> float:
    """Calculate total subword pieces divided by whitespace words."""
    total_words = 0
    total_pieces = 0

    for text in texts:
        total_words += len(text.split())
        total_pieces += len(tokenizer.tokenize(text))

    return total_pieces / total_words if total_words else 0.0


def sequence_lengths(tokenizer, texts) -> list[int]:
    """Return token sequence lengths including special tokens."""
    return [
        len(
            tokenizer(
                text,
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]
        )
        for text in texts
    ]


def unknown_rate(tokenizer, texts) -> float:
    """Calculate the percentage of unknown tokens."""
    pieces = [
        piece
        for text in texts
        for piece in tokenizer.tokenize(text)
    ]

    if not pieces or tokenizer.unk_token is None:
        return 0.0

    unknown = sum(piece == tokenizer.unk_token for piece in pieces)
    return unknown / len(pieces)


def main():
    data = pd.read_csv(DATA)
    data["text"] = data["text"].fillna("").astype(str).map(preprocess)

    arabic_texts = data.loc[data["lang"] == "ar", "text"].tolist()
    english_texts = data.loc[data["lang"] == "en", "text"].tolist()

    results = []
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))

    for axis, (checkpoint, name) in zip(axes.flat, CANDIDATES.items()):
        print(f"Loading {name}...")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        ar_lengths = sequence_lengths(tokenizer, arabic_texts)
        en_lengths = sequence_lengths(tokenizer, english_texts)

        results.append(
            {
                "Tokenizer": name,
                "AR fertility": round(fertility(tokenizer, arabic_texts), 3),
                "EN fertility": round(fertility(tokenizer, english_texts), 3),
                "AR p95 len": round(float(np.percentile(ar_lengths, 95)), 1),
                "EN p95 len": round(float(np.percentile(en_lengths, 95)), 1),
                "AR UNK rate %": round(
                    unknown_rate(tokenizer, arabic_texts) * 100,
                    2,
                ),
            }
        )

        axis.hist(ar_lengths, bins=15, alpha=0.6, label="Arabic")
        axis.hist(en_lengths, bins=15, alpha=0.6, label="English")
        axis.set_title(name)
        axis.set_xlabel("Sequence length")
        axis.set_ylabel("Count")
        axis.legend()

    report = pd.DataFrame(results)

    print("\nTokenizer audit results:")
    print(report.to_string(index=False))

    HISTOGRAM.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(HISTOGRAM, dpi=150)

    print(f"\nHistogram saved to: {HISTOGRAM}")


if __name__ == "__main__":
    main()