"""Lab 1 starter: sentence segmentation."""

import re
import spacy

from bayan.preprocessing.core import preprocess


_NUMBERED_ITEM = re.compile(r"(?<!\w)(?=\d+[.)]\s+)")


def build_pipeline():
    """Build a bilingual spaCy sentence segmentation pipeline."""
    nlp = spacy.blank("xx")
    nlp.add_pipe(
        "sentencizer",
        config={"punct_chars": [".", "!", "?", "؟"]}
    )
    return nlp


def split_sentences(raw: str, nlp) -> list[str]:
    """Preprocess text and return non-empty sentences."""
    cleaned = preprocess(raw)

    # التعامل مع القوائم المرقمة مثل: 1. ... 2. ...
    parts = [
        part.strip()
        for part in _NUMBERED_ITEM.split(cleaned)
        if part.strip()
    ]

    sentences = []

    for part in parts:
        marker_match = re.match(r"^(\d+[.)])\s+(.*)$", part)

        if marker_match:
            marker = marker_match.group(1)
            content = marker_match.group(2)
        else:
            marker = ""
            content = part

        doc = nlp(content)
        current = [
            sentence.text.strip()
            for sentence in doc.sents
            if sentence.text.strip()
        ]

        if marker and current:
            current[0] = f"{marker} {current[0]}"

        sentences.extend(current)

    return sentences
