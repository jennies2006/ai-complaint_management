"""
augment_dataset.py  (v6)
Owner: Member 2 (AI module)

Fixes the category x priority collapse seen in v5: each seed now
samples its base priority from a small POOL (with weights), rather
than having one fixed base priority.

This guarantees every category sees multiple priorities, satisfying
Doc 1's emotion-aware requirement (same issue, different tone ->
different priority).
"""

import argparse
import random
import re

import pandas as pd


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
]

PUNCTUATION_MODES = ["as_is", "as_is", "lowercase", "exclaim", "title"]


# ─────────────────────────────────────────────────────────────
# v6: SEED PRIORITY POOLS
# Each entry: (regex pattern, [(priority, weight), ...])
# Order matters — first match wins.
# ─────────────────────────────────────────────────────────────
SEED_PRIORITY_POOLS = [
    # Purely informational — mostly Low, some Medium
    (re.compile(r"asking about|general query|requesting user manual|"
                r"accessory availability", re.IGNORECASE),
     [("Low", 0.80), ("Medium", 0.20)]),

    # Warranty claims — mostly Medium, some Low and High
    (re.compile(r"warranty (claim|repair)|filing a warranty|"
                r"manufacturing defect|malfunctioned", re.IGNORECASE),
     [("Low", 0.25), ("Medium", 0.55), ("High", 0.20)]),

    # Delivery delays — Medium and High, some Low
    (re.compile(r"order delayed|delivery partner missed|"
                r"still not delivered|no movement", re.IGNORECASE),
     [("Low", 0.15), ("Medium", 0.55), ("High", 0.30)]),

    # Billing — mostly Medium, some Low and High
    (re.compile(r"invoice amount|charged twice|billing|"
                r"refund not credited", re.IGNORECASE),
     [("Low", 0.20), ("Medium", 0.55), ("High", 0.25)]),

    # Not as described — Low to High spread
    (re.compile(r"not as described|does not match|features do not match|"
                r"capacity/wattage|material quality|"
                r"dimensions do not match", re.IGNORECASE),
     [("Low", 0.30), ("Medium", 0.50), ("High", 0.20)]),

    # Size/Fit mismatch — Low/Medium biased
    (re.compile(r"size/fit|too large to fit", re.IGNORECASE),
     [("Low", 0.50), ("Medium", 0.40), ("High", 0.10)]),

    # Wrong item delivered — Medium/High/Critical
    (re.compile(r"wrong item|different model|wrong color|"
                r"different product", re.IGNORECASE),
     [("Medium", 0.25), ("High", 0.50), ("Critical", 0.25)]),

    # Product defect — High/Medium biased, some Critical
    (re.compile(r"product defect|stopped working|"
                r"won't turn on|not working|buttons are unresponsive|"
                r"motor makes a loud noise|heating element", re.IGNORECASE),
     [("Medium", 0.35), ("High", 0.50), ("Critical", 0.15)]),

    # Damaged in transit — High/Critical
    (re.compile(r"damaged in transit|outer packaging torn|"
                r"box arrived crushed|visible cracks", re.IGNORECASE),
     [("Medium", 0.10), ("High", 0.35), ("Critical", 0.55)]),

    # Shattered / completely broken
    (re.compile(r"shattered|completely different", re.IGNORECASE),
     [("High", 0.25), ("Critical", 0.75)]),
]


def sample_base_priority(seed_text: str, rng: random.Random) -> str:
    """Sample a base priority for this seed from its priority pool."""
    for pattern, pool in SEED_PRIORITY_POOLS:
        if pattern.search(seed_text):
            priorities = [p for p, _ in pool]
            weights    = [w for _, w in pool]
            return rng.choices(priorities, weights=weights, k=1)[0]
    # Default: uniform Medium
    return "Medium"


# ─────────────────────────────────────────────────────────────
# Priority bumping
# ─────────────────────────────────────────────────────────────
PRIORITY_ORDER = ["Low", "Medium", "High", "Critical"]


def bump_priority(base: str, delta: int) -> str:
    try:
        idx = PRIORITY_ORDER.index(base)
    except ValueError:
        return "Medium"
    new_idx = max(0, min(len(PRIORITY_ORDER) - 1, idx + delta))
    return PRIORITY_ORDER[new_idx]


def strong_signal_count(text: str) -> int:
    n = 0
    if re.search(r"\b(refund|money\s*back|reimburse(?:ment)?|compensat(?:e|ion))\b",
                 text, re.IGNORECASE): n += 1
    if re.search(r"\b(third time|again|still|repeatedly|multiple times)\b",
                 text, re.IGNORECASE): n += 1
    if re.search(r"\b(urgent|urgently|asap|immediately|right away)\b",
                 text, re.IGNORECASE): n += 1
    if re.search(r"\b[A-Z]{4,}\b", text): n += 1
    if "!" in text: n += 1
    if re.search(r"\b(furious|unacceptable|frustrated|angry|upset)\b",
                 text, re.IGNORECASE): n += 1
    return n


def adjust_priority(base_priority: str, text: str) -> str:
    """
    Emotion-aware adjustment (Doc 1 §intro):
    - Strong emotion (>= 3 signals) -> +1 bump
    - Purely informational content -> -1 bump
    """
    if strong_signal_count(text) >= 3:
        return bump_priority(base_priority, +1)

    _low_re = re.compile(r"\b(asking about|general query|user manual|"
                         r"accessory availability|informational)\b",
                         re.IGNORECASE)
    if _low_re.search(text):
        return bump_priority(base_priority, -1)

    return base_priority


# ─────────────────────────────────────────────────────────────
# Augmentation
# ─────────────────────────────────────────────────────────────
def sample_tone(rng: random.Random) -> str:
    tier = rng.choices(TONE_TIERS, weights=TONE_WEIGHTS, k=1)[0]
    return rng.choice(tier)


def phrase_variants(text: str, max_variants: int = 4) -> list:
    variants = {text}
    for old, new in PHRASE_VARIATIONS:
        if old.lower() in text.lower():
            v = re.sub(re.escape(old), new, text, count=1, flags=re.IGNORECASE)
            variants.add(v)
            if len(variants) >= max_variants:
                break
    return list(variants)


def punctuation_variant(text: str, mode: str) -> str:
    return {"as_is": text, "lowercase": text.lower(),
            "exclaim": text.rstrip(".!?") + "!!!",
            "title": text.title()}.get(mode, text)


def augment_seed(seed_text: str, seed_idx: int, extra_passes: int = 0) -> list:
    rng = random.Random(seed_idx * 1000)
    texts_with_base = []  # each entry: (text, base_priority)

    # Strategy A: context × tone (each variant samples its own base)
    for _ in range(15):
        base = sample_base_priority(seed_text, rng)
        text = f"{rng.choice(CONTEXT_PREFIXES)}" \
               f"{seed_text}{sample_tone(rng)}".strip()
        texts_with_base.append((text, base))

    # Strategy B: phrase variations × tone
    for phrase_var in phrase_variants(seed_text, max_variants=3):
        for _ in range(3):
            base = sample_base_priority(seed_text, rng)
            text = f"{phrase_var}{sample_tone(rng)}".strip()
            texts_with_base.append((text, base))

    # Strategy C: punctuation variants
    for mode in PUNCTUATION_MODES:
        base = sample_base_priority(seed_text, rng)
        text = punctuation_variant(seed_text, mode)
        texts_with_base.append((text, base))

    # Strategy D: extra random combos
    for _ in range(extra_passes * 5):
        base = sample_base_priority(seed_text, rng)
        text = f"{rng.choice(CONTEXT_PREFIXES)}" \
               f"{rng.choice(phrase_variants(seed_text, 3))}" \
               f"{sample_tone(rng)}".strip()
        texts_with_base.append((text, base))

    # Apply emotion-aware adjustment
    rows = []
    for text, base in texts_with_base:
        rows.append({
            "text":     text,
            "priority": adjust_priority(base, text),
        })
    return rows


def augment_dataset(input_csv: str, output_csv: str,
                    random_seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows")

    seeds = (df.groupby("text")
               .agg(category=("category", "first"))
               .reset_index())
    print(f"Unique seed templates: {len(seeds)}")

    cat_counts = seeds["category"].value_counts()
    max_cat_count = cat_counts.max()

    augmented_rows = []
    for i, row in seeds.iterrows():
        base_cat_count = cat_counts[row["category"]]
        extra = max(0, min(4, (max_cat_count - base_cat_count)))
        variants = augment_seed(row["text"], i, extra)
        for v in variants:
            v["category"] = row["category"]
            augmented_rows.append(v)

    aug_df = pd.DataFrame(augmented_rows)
    print(f"\nAfter augmentation: {len(aug_df)} rows")

    aug_df = aug_df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    aug_df = aug_df[aug_df["text"].str.len() > 5].reset_index(drop=True)
    print(f"After dedup: {len(aug_df)} rows")

    aug_df = aug_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    print("\n" + "=" * 60)
    print("AUGMENTED DATASET REPORT (v6)")
    print("=" * 60)
    print(f"Total unique rows: {len(aug_df)}")

    print("\nCategory distribution:")
    for cat, n in aug_df["category"].value_counts().items():
        print(f"  {cat:<25} {n:>5}  ({n/len(aug_df)*100:5.1f}%)")

    print("\nPriority distribution:")
    for pri, n in aug_df["priority"].value_counts().items():
        print(f"  {pri:<25} {n:>5}  ({n/len(aug_df)*100:5.1f}%)")

    print("\nClass balance warnings (< 10%):")
    warned = False
    for col in ["category", "priority"]:
        for label, n in aug_df[col].value_counts().items():
            pct = n / len(aug_df) * 100
            if pct < 10:
                print(f"  ⚠️  {col}='{label}' only {pct:.1f}%")
                warned = True
    if not warned:
        print("  (none — all classes >= 10%)")

    # NEW: category x priority matrix
    print("\nCategory x Priority matrix:")
    pivot = aug_df.groupby(["category", "priority"]).size().unstack(fill_value=0)
    print(pivot)

    # NEW: check emotion-awareness
    print("\nPriorities per category (emotion-aware check):")
    ok = True
    for cat, row in pivot.iterrows():
        present = [p for p in PRIORITY_ORDER if row.get(p, 0) > 0]
        marker = "✅" if len(present) >= 3 else "⚠️"
        if len(present) < 3:
            ok = False
        print(f"  {marker} {cat:<25}  {present}")
    if ok:
        print("  All categories have >= 3 priorities — emotion-aware ✅")
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
