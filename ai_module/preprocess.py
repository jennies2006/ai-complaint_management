"""
preprocess.py
Owner: Member 2 (AI module)

FR-04: NLP text preprocessing pipeline for the
       AI-Based Complaint Management System (e-commerce).

The function clean_text() is the SHARED implementation (Doc 1 §10
shared-code rule). Both train.py and predict.py MUST import it from
here — never redefine it, or the model sees different text in
production than it learned from.

Pipeline (FR-04):
    lowercase -> strip URLs -> strip punctuation
    -> tokenize -> remove stopwords (except negations)
    -> lemmatize -> join

Example (e-commerce complaint):
    Input:  "The heating element is NOT working!!!"
    Output: "heating element not working"
"""

import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)

_LEMMATIZER = WordNetLemmatizer()
_STOPWORDS  = set(stopwords.words("english"))

# Keep negations — they carry meaning in complaint text.
_KEEP_WORDS = {"not", "no", "nor", "never", "none", "cannot"}

_URL_RE   = re.compile(r"http\S+|www\.\S+|\S+@\S+")
_PUNCT_RE = re.compile(r"[^a-z0-9\s]")


def clean_text(raw_text: str) -> str:
    """
    FR-04 pipeline. Returns cleaned text ready for TF-IDF.

    Applied identically in training and prediction — do not modify
    one place without the other.
    """
    if not isinstance(raw_text, str):
        return ""

    text = raw_text.lower()
    text = _URL_RE.sub(" ", text)
    text = _PUNCT_RE.sub(" ", text)

    tokens = text.split()
    tokens = [t for t in tokens
              if t not in _STOPWORDS or t in _KEEP_WORDS]
    tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if len(t) > 1]

    return " ".join(tokens)


if __name__ == "__main__":
    # Sanity test using REAL e-commerce complaint samples
    # (taken from our augmented dataset).
    samples = [
        "The heating element is NOT working!!!",
        "Third time my order is late! I want a refund NOW",
        "Heating element not working, device won't turn on",
        "Buttons are unresponsive, unit does not power on",
        "Product dimensions do not match the listing",
        "Buttons are unresponsive, unit does not power on",
        "Package contained a completely different product",
        "Refund not credited despite return being approved",
        "",
        "   ",
    ]
    for s in samples:
        print(f"IN : {s!r}")
        print(f"OUT: {clean_text(s)!r}")
        print()
