"""Arabic normalisation profiles used by Bayan."""

import re
import unicodedata
from functools import lru_cache
from dataclasses import dataclass


_ARABIC_DIACRITICS = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]"
)

_WHITESPACE = re.compile(r"\s+")

_BAYAN_TRANSLATION = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ؤ": "و",
        "ئ": "ي",
        "ى": "ي",
        "ة": "ه",
    }
)


@dataclass(frozen=True)
class ArabicProfile:
    name: str
    dediacritize: bool = False


def normalize_arabic(text: str, profile: ArabicProfile) -> str:
    """Return Arabic text normalised according to the selected profile."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("ـ", "")

    if profile.dediacritize:
        normalized = _ARABIC_DIACRITICS.sub("", normalized)

    if profile.name == "bayan_ar_v1":
        normalized = normalized.translate(_BAYAN_TRANSLATION)
    elif profile.name == "display_ar_v1":
        pass
    else:
        raise ValueError(f"Unknown Arabic profile: {profile.name}")

    normalized = _WHITESPACE.sub(" ", normalized).strip()
    return normalized


@lru_cache(maxsize=1)
def _get_morphological_tokenizer():
    """Load the CAMeL morphological tokenizer once."""

    from camel_tools.disambig.mle import MLEDisambiguator
    from camel_tools.tokenizers.morphological import MorphologicalTokenizer

    disambiguator = MLEDisambiguator.pretrained()
    return MorphologicalTokenizer(
        disambiguator,
        scheme="d3tok",
        split=True,
    )


def segment(text: str) -> list[str]:
    """Split Arabic clitics using CAMeL Tools d3tok."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    from camel_tools.tokenizers.word import simple_word_tokenize

    words = simple_word_tokenize(text)
    tokenizer = _get_morphological_tokenizer()
    raw_tokens = tokenizer.tokenize(words)

    segmented_tokens = []

    for raw_token in raw_tokens:
        pieces = re.split(r"(?:\+_|_\+)", raw_token)

        for piece in pieces:
            cleaned_piece = piece.replace("+", "").replace("_", "").strip()

            if cleaned_piece:
                segmented_tokens.append(cleaned_piece)

    return segmented_tokens