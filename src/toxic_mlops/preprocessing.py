"""Stateless, versioned preprocessing shared by training and inference."""

import html
import re
import unicodedata

from sklearn.base import BaseEstimator, TransformerMixin


def clean_text(text: str) -> str:
    # Iterate entity decoding to a fixed point: cleaning must be idempotent.
    for _ in range(10):
        decoded = html.unescape(text)
        if decoded == text:
            break
        text = decoded
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"</?[A-Za-z][^>]*>", " ", text)
    text = re.sub("[\ufeff\u200b]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def duplicate_key(text: str) -> str:
    return unicodedata.normalize("NFKC", clean_text(text)).lower().replace("ё", "е")


def preprocess(text: str, version: str = "v1") -> str:
    text = clean_text(text)
    if version == "v1":
        return text
    if version != "v2":
        raise ValueError(f"Unknown preprocessing version: {version}")
    text = text.lower().replace("ё", "е")
    text = re.sub(r"https?://\S+|www\.\S+", " url ", text)
    text = re.sub(r"(?<!\w)@\w+", " user ", text)
    return re.sub(r"\s+", " ", text).strip()


class TextPreprocessor(TransformerMixin, BaseEstimator):
    def __init__(self, version: str = "v1"):
        self.version = version

    def fit(self, texts, y=None):
        return self

    def transform(self, texts):
        return [preprocess(text, self.version) for text in texts]
