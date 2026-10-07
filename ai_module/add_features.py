"""
add_features.py
Owner: Member 2 (AI module)
Step 1: add missing features to the Kaggle e-commerce support dataset.
Follows Doc 1 (AI Module Interface Spec) and Member 2.docx exactly.

The functions compute_sentiment_score() and compute_urgency_signals()
are the SHARED implementations (Doc 1 §10 shared-code rule).
predict.py MUST import these from here, not redefine them.
"""

import argparse
import re
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ─────────────────────────────────────────────────────────────
# SHARED FEATURE FUNCTIONS (Doc 1 §10 — shared-code rule)
# ─────────────────────────────────────────────────────────────
VADER = SentimentIntensityAnalyzer()

URGENCY_VOCAB = [
    "refund demand", "repeat issue", "all caps",
    "exclamation", "urgency word",
]

_REFUND_RE   = re.compile(
    r"\b(refund|money\s*back|reimburse(?:ment)?|compensat(?:e|ion))\b",
    re.IGNORECASE,
)
_REPEAT_RE   = re.compile(
    r"\b(again|repeatedly|repeat(?:ed)?|second time|third time|"
    r"multiple times|still)\b",
    re.IGNORECASE,
)
_ALLCAPS_RE  = re.compile(r"\b[A-Z]{4,}\b")
_EXCLAM_RE   = re.compile(r"!")
_URGENCY_RE  = re.compile(
    r"\b(urgent|urgently|asap|immediately|right away)\b",
    re.IGNORECASE,
)


def compute_sentiment_score(raw_text: str) -> float:
    """VADER compound on RAW text. Range [-1, +1], 3 decimals."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        return 0.0
    return round(float(VADER.polarity_scores(raw_text)["compound"]), 3)


def compute_urgency_signals(raw_text: str) -> list:
    """Urgency signals on RAW text (capital letters and '!' matter)."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        return []
    signals = []
    if _REFUND_RE.search(raw_text):  signals.append("refund demand")
    if _REPEAT_RE.search(raw_text):  signals.append("repeat issue")
    if _ALLCAPS_RE.search(raw_text): signals.append("all caps")
    if _EXCLAM_RE.search(raw_text):  signals.append("exclamation")
    if _URGENCY_RE.search(raw_text): signals.append("urgency word")
    return signals


# ─────────────────────────────────────────────────────────────
# LABEL MAPPINGS (Doc 1 §5 and §7)
# ─────────────────────────────────────────────────────────────
CATEGORY_MAP = {
    "Late Delivery":         "Delivery",
    "Product Defect":        "Product Quality",
    "Damaged in Transit":    "Product Quality",
    "Not as Described":      "Product Quality",
    "Wrong Item Delivered":  "Returns & Replacements",
    "Size/Fit Mismatch":     "Returns & Replacements",
    "Billing Issue":         "Payments & Refunds",
    "Warranty Claim":        "Other",
    "Other":                 "Other",
}

CATEGORY_TO_DEPARTMENT = {
    "Delivery":               "Logistics & Delivery",
    "Payments & Refunds":     "Payments & Refunds",
    "Product Quality":        "Quality Assurance",
    "Returns & Replacements": "Returns & Replacements",
    "Seller Support":         "Seller Support",
    "Account & Technical":    "Technical Support",
    "Other":                  "Admin",
}

VALID_PRIORITIES = ["Low", "Medium", "High", "Critical"]
AI_CATEGORIES    = [
    "Delivery", "Product Quality",
    "Returns & Replacements", "Payments & Refunds", "Other",
]


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def add_features(input_csv: str, output_csv: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    print(f"Loaded: {len(df)} rows")

    # 1. text <- issue_description
    df = df.rename(columns={"issue_description": "text"})
    df["text"] = df["text"].fillna("").astype(str)

    # 2. category <- issue_category (mapped to 5 AI categories)
    df["category"] = df["issue_category"].apply(
        lambda c: CATEGORY_MAP.get(
            c.strip() if isinstance(c, str) else "", "Other"
        )
    )

    # 3. priority — no remapping (Doc 1 §10). Safety net only.
    bad = ~df["priority"].isin(VALID_PRIORITIES)
    if bad.any():
        print(f"[WARN] {int(bad.sum())} invalid priority rows -> 'Medium'")
        df.loc[bad, "priority"] = "Medium"

    # 4. sentiment_score (VADER on raw text)
    df["sentiment_score"] = df["text"].apply(compute_sentiment_score)

    # 5. urgency_signals (list of strings from fixed vocab)
    df["urgency_signals"] = df["text"].apply(compute_urgency_signals)

    # 6. department (lookup from category — AI never predicts it)
    df["department"] = df["category"].apply(
        lambda c: CATEGORY_TO_DEPARTMENT.get(c, "Admin")
    )

    # Final column order (Doc 1 §10)
    final_cols = [
        "ticket_id", "text", "category", "priority",
        "sentiment_score", "urgency_signals", "department",
    ]
    out = df[final_cols].copy()

    # ───────────────────────────────────────────────────────
    # DATA QUALITY REPORT (Member 2.docx requirement)
    # ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)

    n_missing_text = int((out["text"].str.strip() == "").sum())
    n_dupes        = int(out["text"].duplicated().sum())
    print(f"[1] Empty-text rows     : {n_missing_text}")
    print(f"[2] Duplicate-text rows : {n_dupes}")

    cat_counts = out["category"].value_counts()
    print("\n[3] Category distribution:")
    for cat, n in cat_counts.items():
        print(f"    {cat:<25} {n:>6}  ({n/len(out)*100:5.1f}%)")

    pri_counts = out["priority"].value_counts()
    print("\n[4] Priority distribution:")
    for pri, n in pri_counts.items():
        print(f"    {pri:<25} {n:>6}  ({n/len(out)*100:5.1f}%)")

    print("\n[5] Warnings:")
    warned = False
    for cat, n in cat_counts.items():
        if n < 50:
            print(f"    ⚠️  category '{cat}' has only {n} rows")
            warned = True
    for pri, n in pri_counts.items():
        if n < 50:
            print(f"    ⚠️  priority '{pri}' has only {n} rows")
            warned = True
    if not warned:
        print("    (none)")
    print("=" * 60 + "\n")

    # ───────────────────────────────────────────────────────
    # DROP EMPTY-TEXT ROWS (unusable for training)
    # ───────────────────────────────────────────────────────
    n_before = len(out)
    out = out[out["text"].str.strip() != ""].copy()
    n_dropped = n_before - len(out)
    if n_dropped > 0:
        print(f"[INFO] Dropped {n_dropped} empty-text rows.")

    # ───────────────────────────────────────────────────────
    # SAVE CATEGORY MAPPING DEBUG FILE
    # ───────────────────────────────────────────────────────
    (df[["issue_category", "category"]]
        .drop_duplicates()
        .to_csv("data/_category_mapping_debug.csv", index=False))
    print("[INFO] Saved data/_category_mapping_debug.csv")

    # ───────────────────────────────────────────────────────
    # ASSERTIONS (Doc 1 safety net)
    # ───────────────────────────────────────────────────────
    assert set(out["priority"].unique()) <= set(VALID_PRIORITIES)
    assert set(out["category"].unique()) <= set(AI_CATEGORIES)
    for lst in out["urgency_signals"]:
        assert isinstance(lst, list) and set(lst) <= set(URGENCY_VOCAB)

    # ───────────────────────────────────────────────────────
    # SAVE
    # ───────────────────────────────────────────────────────
    out.to_csv(output_csv, index=False)
    print(f"[INFO] Wrote {len(out)} rows -> {output_csv}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input",  default="data/support_tickets_dataset.csv")
    p.add_argument("--output", default="data/final_dataset.csv")
    args = p.parse_args()
    add_features(args.input, args.output)
