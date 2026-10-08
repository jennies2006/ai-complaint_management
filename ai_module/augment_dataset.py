"""
augment_dataset.py  (v10 — tightened Low label rule)
Owner: Member 2 (AI module)

v10 change from v9:
    Low no longer absorbs mild complaints. Only genuinely
    informational queries are Low now.

    Specifically:
      - removed bare "invoice" match (was catching "invoice amount
        does not match" — a billing dispute, now Medium)
      - removed "size/fit | too large to fit" from Low; moved to
        Medium (a wrong-size product is a real complaint, not a
        question)
      - removed over-broad "asking about" (was catching complaints
        phrased as questions)
      - added explicit "extended warranty" for the common
        informational case
      - added "register the item" and "invoice with my company"
        for the specific informational GST case

    Result: Low labels are now traceable to a clean rule —
    informational query, no emotional content.

Priority is DETERMINISTIC — same text -> same priority.
Priority labels use the SAME signals the priority model consumes
at inference time (Doc 1 §10): sentiment_score + 5 urgency flags.
"""

import argparse
import random
import re

import pandas as pd

from ai_module.add_features import (compute_sentiment_score,
                                    compute_urgency_signals)


# ─────────────────────────────────────────────────────────────
# EXTRA SEED TEMPLATES (27 new, hand-written)
# Grouped by category so we can verify coverage.
# ─────────────────────────────────────────────────────────────
EXTRA_SEEDS = {
    "Delivery": [
        "Courier marked my package as delivered but it never arrived",
        "Shipping notification came five days ago and nothing since",
        "Delivery was scheduled for today but got pushed to next week",
        "Tracking page has not updated in over two weeks",
        "Package was left outside my building and went missing",
        "Delivery agent called once and never came back",
    ],
    "Product Quality": [
        "The appliance started smoking on the second use",
        "Paint is peeling off the exterior after only a month",
        "The motor died after just ten uses",
        "Item smells strongly of burnt plastic when turned on",
        "The lid does not close properly and leaks every time",
        "There is a rattling sound inside the unit when shaken",
    ],
    "Returns & Replacements": [
        "Return pickup was scheduled three times and no one came",
        "I returned the item three weeks ago and got no acknowledgment",
        "The replacement unit they sent was also defective",
        "Refund is still pending although I returned the item last month",
        "The pickup agent refused to take the item and left",
    ],
    "Payments & Refunds": [
        "The discount shown at checkout was not applied to my card",
        "I was promised a partial refund but received nothing",
        "Payment gateway charged me in USD instead of INR",
        "Money was debited twice for one order",
        "I paid for express shipping but was charged standard fee",
    ],
    "Other": [
        "I need the invoice with my company's GST number",
        "Can I change the delivery address on an active order",
        "How do I register the product for extended warranty",
        "Please share the warranty card for this product",
        "Is this model compatible with the accessory from last year",
    ],
}


# ─────────────────────────────────────────────────────────────
# Tone tiers (weighted sampling)
# ─────────────────────────────────────────────────────────────
TONE_NEUTRAL = ["", " Please help.", " Please respond soon.",
                " Can someone look into this?"]
TONE_MILD    = [" I'm very disappointed.", " I'm a bit frustrated.",
                " This has been inconvenient.",
                " Hoping this gets resolved quickly."]
TONE_STRONG  = [" I'm extremely frustrated!", " This is unacceptable!",
                " I want a refund NOW!",
                " This is the third time this has happened.",
                " URGENT: please handle this immediately."]
TONE_TIERS   = [TONE_NEUTRAL, TONE_MILD, TONE_STRONG]
TONE_WEIGHTS = [0.55, 0.25, 0.20]

CONTEXT_PREFIXES = [
    "", "I ordered this last week. ", "I've been a customer for years. ",
    "I need this for work tomorrow. ", "Just received the package today. ",
    "Second order in a row with issues. ", "This was a gift for my mother. ",
    "I spent a lot on this. ",
]

PHRASE_VARIATIONS = [
    ("not working", "broken"), ("not working", "not functioning"),
    ("not working", "stopped working"),
    ("device won't turn on", "device refuses to start"),
    ("device won't turn on", "unit won't power up"),
    ("product", "item"), ("product", "unit"),
    ("arrived", "was delivered"), ("arrived", "showed up"),
    ("ordered", "purchased"), ("ordered", "bought"),
    ("Requesting", "I am requesting"),
    ("Asking about", "I am asking about"),
    ("does not", "doesn't"),
]

PUNCTUATION_MODES = ["as_is", "as_is", "lowercase", "exclaim", "title"]


# ─────────────────────────────────────────────────────────────
# Seed floor priorities  (v10 — tightened Low)
# ─────────────────────────────────────────────────────────────
SEED_FLOORS = [
    # ── Low: informational queries only ──────────────────────
    (re.compile(r"how do i|register the product|register the item|"
                r"share the warranty card|requesting user manual|"
                r"change the delivery address|compatible with|"
                r"general query about|accessory availability|"
                r"invoice with my company|"
                r"extended warranty",
                re.IGNORECASE), "Low"),

    # ── Medium: warranty / billing / not-as-described ────────
    (re.compile(r"warranty (claim|repair)|filing a warranty|"
                r"manufacturing defect|malfunctioned", re.IGNORECASE), "Medium"),
    (re.compile(r"invoice amount|charged twice|billing|"
                r"refund not credited|discount shown|"
                r"partial refund|payment gateway|debited twice|"
                r"express shipping but|"
                r"size/fit|too large to fit", re.IGNORECASE), "Medium"),
    (re.compile(r"order delayed|delivery partner missed|"
                r"still not delivered|no movement|"
                r"scheduled for today|tracking page has not updated",
                re.IGNORECASE), "Medium"),
    (re.compile(r"not as described|does not match|features do not match|"
                r"capacity/wattage|material quality|"
                r"dimensions do not match|lid does not close|"
                r"rattling sound", re.IGNORECASE), "Medium"),

    # ── Medium: product defect (can rise with emotion) ───────
    (re.compile(r"product defect|stopped working|"
                r"won't turn on|not working|buttons are unresponsive|"
                r"motor makes a loud noise|heating element|"
                r"started smoking|paint is peeling|"
                r"motor died|burnt plastic", re.IGNORECASE), "Medium"),

    # ── High: wrong item / replacement problems ──────────────
    (re.compile(r"wrong item|different model|wrong color|"
                r"different product|replacement unit was also", re.IGNORECASE), "High"),
    (re.compile(r"return pickup was scheduled|pickup agent refused|"
                r"returned the item three weeks ago", re.IGNORECASE), "High"),

    # ── High: damaged in transit / shattered ─────────────────
    (re.compile(r"damaged in transit|outer packaging torn|"
                r"box arrived crushed|visible cracks|"
                r"package was left outside|went missing", re.IGNORECASE), "High"),
    (re.compile(r"shattered|completely different", re.IGNORECASE), "High"),
]

PRIORITY_ORDER = ["Low", "Medium", "High", "Critical"]


def get_seed_floor(seed_text: str) -> str:
    for pattern, floor in SEED_FLOORS:
        if pattern.search(seed_text):
            return floor
    return "Medium"


def bump_priority(base: str, delta: int) -> str:
    try:
        idx = PRIORITY_ORDER.index(base)
    except ValueError:
        return "Medium"
    new_idx = max(0, min(len(PRIORITY_ORDER) - 1, idx + delta))
    return PRIORITY_ORDER[new_idx]


def priority_from_signals(seed_text: str, augmented_text: str) -> str:
    """
    Formula B:
        floor = get_seed_floor(seed_text)
        n     = len(compute_urgency_signals(augmented_text))   # 0..5
        s     = compute_sentiment_score(augmented_text)        # -1..1
        bonus = 1 if s < -0.5 else 0
        priority = bump_priority(floor, min(n + bonus, 3))
    """
    floor = get_seed_floor(seed_text)

    try:
        urgency = compute_urgency_signals(augmented_text)
        n = len(urgency) if isinstance(urgency, list) else 0
    except Exception:
        n = 0

    try:
        s = float(compute_sentiment_score(augmented_text))
    except Exception:
        s = 0.0

    bonus = 1 if s < -0.5 else 0
    return bump_priority(floor, min(n + bonus, 3))


# ─────────────────────────────────────────────────────────────
# Augmentation helpers
# ─────────────────────────────────────────────────────────────
def sample_tone(rng):
    tier = rng.choices(TONE_TIERS, weights=TONE_WEIGHTS, k=1)[0]
    return rng.choice(tier)


def phrase_variants(text, max_variants=4):
    variants = {text}
    for old, new in PHRASE_VARIATIONS:
        if old.lower() in text.lower():
            v = re.sub(re.escape(old), new, text, count=1, flags=re.IGNORECASE)
            variants.add(v)
            if len(variants) >= max_variants:
                break
    return list(variants)


def punctuation_variant(text, mode):
    return {"as_is": text, "lowercase": text.lower(),
            "exclaim": text.rstrip(".!?") + "!!!",
            "title": text.title()}.get(mode, text)


def augment_seed(seed_text, seed_idx, extra_passes=0):
    rng = random.Random(seed_idx * 1000)
    texts = []

    for _ in range(15):
        texts.append(f"{rng.choice(CONTEXT_PREFIXES)}"
                     f"{seed_text}{sample_tone(rng)}".strip())

    for phrase_var in phrase_variants(seed_text, max_variants=3):
        for _ in range(3):
            texts.append(f"{phrase_var}{sample_tone(rng)}".strip())

    for mode in PUNCTUATION_MODES:
        texts.append(punctuation_variant(seed_text, mode))

    for _ in range(extra_passes * 5):
        texts.append(f"{rng.choice(CONTEXT_PREFIXES)}"
                     f"{rng.choice(phrase_variants(seed_text, 3))}"
                     f"{sample_tone(rng)}".strip())

    return [{"text": t,
             "priority": priority_from_signals(seed_text, t)} for t in texts]


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def augment_dataset(input_csv, output_csv, random_seed=42):
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows from {input_csv}")

    kaggle_seeds = (df.groupby("text")
                      .agg(category=("category", "first"))
                      .reset_index())
    print(f"Kaggle seed templates: {len(kaggle_seeds)}")

    extra_rows = []
    for cat, seeds_list in EXTRA_SEEDS.items():
        for s in seeds_list:
            extra_rows.append({"text": s, "category": cat})
    extra_df = pd.DataFrame(extra_rows)
    print(f"Extra hand-written seeds: {len(extra_df)}")

    seeds = pd.concat([kaggle_seeds, extra_df], ignore_index=True)
    seeds = seeds.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"Total seed templates: {len(seeds)}")

    cat_counts = seeds["category"].value_counts()
    max_cat_count = cat_counts.max()

    augmented_rows = []
    for i, row in seeds.iterrows():
        base_cat_count = cat_counts[row["category"]]
        extra = max(0, min(4, (max_cat_count - base_cat_count)))
        variants = augment_seed(row["text"], i, extra)
        for v in variants:
            v["category"] = row["category"]
            v["seed_text"] = row["text"]
            augmented_rows.append(v)

    aug_df = pd.DataFrame(augmented_rows)
    print(f"After augmentation: {len(aug_df)} rows")

    aug_df = aug_df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    aug_df = aug_df[aug_df["text"].str.len() > 5].reset_index(drop=True)
    print(f"After dedup: {len(aug_df)} rows")

    aug_df = aug_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    print("\n" + "=" * 60)
    print("AUGMENTED DATASET REPORT (v10)")
    print("=" * 60)
    print(f"Total unique rows: {len(aug_df)}")
    print(f"Total seeds: {seeds['text'].nunique()}")

    print("\nSeeds per category:")
    for cat, n in seeds["category"].value_counts().items():
        print(f"  {cat:<25} {n:>3}")

    print("\nCategory distribution (rows):")
    for cat, n in aug_df["category"].value_counts().items():
        print(f"  {cat:<25} {n:>5}  ({n/len(aug_df)*100:5.1f}%)")

    print("\nPriority distribution (rows):")
    for pri, n in aug_df["priority"].value_counts().items():
        print(f"  {pri:<25} {n:>5}  ({n/len(aug_df)*100:5.1f}%)")

    print("\nCategory x Priority matrix:")
    pivot = aug_df.groupby(["category", "priority"]).size().unstack(fill_value=0)
    print(pivot)

    print("=" * 60)
    aug_df.to_csv(output_csv, index=False)
    print(f"\n[INFO] Wrote {len(aug_df)} rows -> {output_csv}")
    return aug_df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input",  default="data/final_dataset.csv")
    p.add_argument("--output", default="data/augmented_dataset.csv")
    p.add_argument("--seed",   type=int, default=42)
    args = p.parse_args()
    augment_dataset(args.input, args.output, args.seed)
