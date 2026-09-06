"""Lab 1 starter: versioned bilingual preprocessing for Bayan."""

import re
import unicodedata

PREPROC_VERSION = "1.2.0"


def normalize(text: str) -> str:
    """Return deterministic Bayan normalisation while preserving task signal."""
    
    # توحيد أشكال Unicode
    text = unicodedata.normalize("NFKC", text)

    # حذف التطويل العربي ـ
    text = text.replace("ـ", "")

    # تقليل تكرار الحروف أو العلامات إلى مرتين
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # حذف المسافات الزائدة والأسطر الفارغة
    return " ".join(text.split())


def mask_pii(text: str) -> str:
    """Mask supported phone numbers and Saudi national-ID-shaped values."""
    
    # أرقام الجوال السعودية
    phone_pattern = r"(?<!\d)(?:\+?9665\d{8}|05\d{8})(?!\d)"

    # الهوية الوطنية أو هوية المقيم
    national_id_pattern = r"(?<!\d)[12]\d{9}(?!\d)"

    text = re.sub(phone_pattern, "<PHONE>", text)
    text = re.sub(national_id_pattern, "<NATIONAL_ID>", text)

    return text


def preprocess(text: str) -> str:
    """Apply the shared train/eval/serve preprocessing contract."""
    
    # نخفي البيانات الشخصية أولًا ثم ننظف النص
    return normalize(mask_pii(text))
