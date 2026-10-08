# ai-complaint_management

AI module for the e-commerce complaint management system.

Member 2 deliverable — Python package exposing a single function:

    from ai_module.predict import predict_complaint

    result = predict_complaint("Third time my order is late! I want a refund NOW")

Returns a dict with keys:
`category`, `category_confidence`, `priority`, `priority_confidence`,
`sentiment_score`, `urgency_signals`, `model_version`, `status`.

Never raises — on error returns `status="error"` with `category=None`
and `priority=None`.

## Install

    pip install -r requirements.txt

## NLTK data (required)

Run once after install:

    python -m nltk.downloader stopwords wordnet

## Models

Trained models live in `ai_module/models/` and are loaded once at
module import.

- **Category model:** sentence-transformers `all-MiniLM-L6-v2`
  (384-dim) → LogisticRegression. Trained on 1,728 augmented rows,
  grouped train/test split by seed (no leakage). Category F1 = 0.826.
- **Priority model:** `sentiment_score` + 5 binary urgency flags
  (Doc 1 §6 vocabulary) → LogisticRegression. Priority F1 = 0.632.

Priority is emotion-aware: sentiment and urgency signals drive the
final priority, and tone pairs (same issue, different tone) escalate
accordingly.

## Retrain

    python -m ai_module.augment_dataset    # regenerates data/augmented_dataset.csv
    python -m ai_module.train              # retrains and saves to ai_module/models/

## Layout

    ai_module/
        add_features.py     # shared sentiment + urgency (Doc 1 §10e)
        preprocess.py       # FR-04 cleaning pipeline
        augment_dataset.py  # seed templates + augmentation
        train.py            # trains category + priority models
        predict.py          # public predict_complaint() interface
        models/             # joblib artifacts
    data/
        augmented_dataset.csv
    requirements.txt
    README.md
