"""
train.py  (v3 — sentence embeddings)
Owner: Member 2 (AI module)

v3 change from v2:
    Replaced TF-IDF with sentence embeddings (all-MiniLM-L6-v2).
    Reason: TF-IDF could not handle synonym variation in our
    templated dataset. F1 was stuck at 0.51 category / 0.44 priority.
    Sentence embeddings understand that "delayed" ≈ "postponed"
    and "broken" ≈ "damaged", which are exactly the distinctions
    our categories depend on.

    Model architecture (LogisticRegression) unchanged.
    Train/test group split unchanged.
    Shared-code rule unchanged.

Follows Doc 1 §10e (shared-code rule) and Member 2.docx FR-06/07/09.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import GroupShuffleSplit

from ai_module.add_features import (URGENCY_VOCAB,
                                    compute_sentiment_score,
                                    compute_urgency_signals)
from ai_module.preprocess import clean_text


DATA_PATH   = Path("data/augmented_dataset.csv")
MODELS_DIR  = Path("ai_module/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_SEED = 42

ENCODER_NAME = "all-MiniLM-L6-v2"


def build_urgency_flags(urgency_lists: pd.Series) -> pd.DataFrame:
    cols = {}
    for sig in URGENCY_VOCAB:
        col = "urg_" + sig.replace(" ", "_")
        cols[col] = urgency_lists.apply(
            lambda lst, s=sig: 1 if isinstance(lst, list) and s in lst else 0
        )
    return pd.DataFrame(cols, index=urgency_lists.index)


def evaluate(y_true, y_pred, labels, model_name):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n{'=' * 60}")
    print(f"{model_name} — EVALUATION")
    print(f"{'=' * 60}")
    print(f"Accuracy (micro): {acc:.4f}")
    print(f"Precision (macro): {prec:.4f}")
    print(f"Recall (macro):    {rec:.4f}")
    print(f"F1 (macro):        {f1:.4f}")
    print(f"\nClassification report:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    print(f"Confusion matrix (rows=true, cols=pred):")
    print(f"Labels order: {labels}")
    print(confusion_matrix(y_true, y_pred, labels=labels))

    return {
        "accuracy":  round(float(acc),  4),
        "precision": round(float(prec), 4),
        "recall":    round(float(rec),  4),
        "f1_macro":  round(float(f1),   4),
    }


def main():
    print(f"Loading {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows")

    if "seed_text" not in df.columns:
        raise RuntimeError("Re-run augment_dataset.py first.")

    # ── FEATURES ─────────────────────────────────────────────
    print("\nCleaning text...")
    df["text_clean"] = df["text"].apply(clean_text)

    print("Computing sentiment + urgency on RAW text...")
    df["sentiment_score"] = df["text"].apply(compute_sentiment_score)
    df["urgency_signals"] = df["text"].apply(compute_urgency_signals)
    urg_flags = build_urgency_flags(df["urgency_signals"])
    df = pd.concat([df, urg_flags], axis=1)

    df = df[df["text_clean"].str.strip() != ""].reset_index(drop=True)
    print(f"Rows after cleaning: {len(df)}")

    # ── GROUP SPLIT ──────────────────────────────────────────
    print("\nSplitting by SEED...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20,
                            random_state=RANDOM_SEED)
    train_idx, test_idx = next(gss.split(df, groups=df["seed_text"]))
    train_df = df.loc[train_idx].copy()
    test_df  = df.loc[test_idx].copy()
    print(f"  Train: {len(train_df)} rows ({train_df['seed_text'].nunique()} seeds)")
    print(f"  Test : {len(test_df)} rows ({test_df['seed_text'].nunique()} seeds)")
    overlap = set(train_df["seed_text"]).intersection(set(test_df["seed_text"]))
    print(f"  Seed overlap (must be 0): {len(overlap)}")

    # ── LOAD ENCODER ─────────────────────────────────────────
    print(f"\nLoading encoder '{ENCODER_NAME}'...")
    encoder = SentenceTransformer(ENCODER_NAME)

    print("Encoding train and test texts...")
    X_train_emb = encoder.encode(train_df["text_clean"].tolist(),
                                 show_progress_bar=True,
                                 convert_to_numpy=True)
    X_test_emb  = encoder.encode(test_df["text_clean"].tolist(),
                                 show_progress_bar=True,
                                 convert_to_numpy=True)
    print(f"Embedding shape: {X_train_emb.shape}")

    urg_cols = list(urg_flags.columns)

    # ── CATEGORY MODEL ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("CATEGORY MODEL — training")
    print("=" * 60)

    cat_model = LogisticRegression(max_iter=2000,
                                   class_weight="balanced",
                                   random_state=RANDOM_SEED)
    cat_model.fit(X_train_emb, train_df["category"])

    y_pred_cat = cat_model.predict(X_test_emb)
    cat_metrics = evaluate(test_df["category"], y_pred_cat,
                           labels=sorted(df["category"].unique()),
                           model_name="CATEGORY MODEL")

    # ── PRIORITY MODEL ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("PRIORITY MODEL — training")
    print("=" * 60)

    # Priority model: embeddings + sentiment + 5 urgency flags
    # Priority model: signals only (per Doc 1 §10).
    # Embeddings dominated the 6 signal features and collapsed Low recall to 0.01.
    # Diagnostic: signals-only model reaches Low recall 0.73, macro F1 0.55.
    X_train_pri = train_df[["sentiment_score"] + urg_cols].values
    X_test_pri  = test_df[["sentiment_score"] + urg_cols].values

    pri_model = LogisticRegression(max_iter=2000,
                                   class_weight="balanced",
                                   random_state=RANDOM_SEED)
    pri_model.fit(X_train_pri, train_df["priority"])

    y_pred_pri = pri_model.predict(X_test_pri)
    pri_metrics = evaluate(test_df["priority"], y_pred_pri,
                           labels=["Low", "Medium", "High", "Critical"],
                           model_name="PRIORITY MODEL")

    # ── SAVE ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SAVING ARTIFACTS")
    print("=" * 60)

    joblib.dump(cat_model, MODELS_DIR / "category_model.joblib")
    joblib.dump(pri_model, MODELS_DIR / "priority_model.joblib")
    joblib.dump(urg_cols,  MODELS_DIR / "urgency_flags_order.joblib")
    with open(MODELS_DIR / "encoder_name.txt", "w") as f:
        f.write(ENCODER_NAME)
    with open(MODELS_DIR / "vectorizer_type.txt", "w") as f:
        f.write("sentence-transformers")

    # Remove old TF-IDF artifacts (no longer used)
    for old in ["tfidf_vectorizer.joblib", "priority_tfidf.joblib"]:
        p = MODELS_DIR / old
        if p.exists():
            p.unlink()
            print(f"  removed old {p.name}")

    for p in sorted(MODELS_DIR.glob("*")):
        if p.is_file():
            print(f"  ✅ {p}  ({p.stat().st_size / 1024:.1f} KB)")

    summary = {
        "category_model": cat_metrics,
        "priority_model": pri_metrics,
        "category_model_features": "sentence-transformers:" + ENCODER_NAME,
        "priority_model_features": "sentiment_score + 5 urgency flags",
        "trained_on_rows": len(train_df),
        "tested_on_rows":  len(test_df),
        "seed_overlap":    len(overlap),
        "urgency_flags_order": urg_cols,
    }
    with open(MODELS_DIR / "training_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Category F1 (macro): {cat_metrics['f1_macro']}")
    print(f"Priority F1 (macro): {pri_metrics['f1_macro']}")


if __name__ == "__main__":
    main()
