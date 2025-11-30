"""Helpers to load WELFake data and construct a TF-IDF + RandomForest pipeline.
Provides a factory `your_model()` returning a baseline RF pipeline for reuse."""

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "WELFake_Dataset.csv")
RANDOM_STATE = 42


def load_dataset(csv_path: str):
    """Load dataset CSV, normalize labels, concatenate title+body, and return text list and label list."""
    df = pd.read_csv(csv_path)
    expected = {"title", "text", "label"}
    missing = expected - set(df.columns)
    if len(missing) > 0:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
    if df["label"].dtype == object:
        df["label"] = df["label"].str.lower().map({"fake": 1, "real": 0})
    df["label"] = df["label"].astype(int)
    texts = (
        df["title"].fillna("").astype(str).str.strip() + " \n" + df["text"].fillna("").astype(str).str.strip()
    ).tolist()
    labels = df["label"].tolist()
    return texts, labels


def build_pipeline(n_estimators=200, max_depth=None, max_features="sqrt", ngram_range=(1, 2)):
    """Create TF-IDF + RandomForest pipeline with adjustable hyperparameters."""
    tfidf = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=ngram_range)
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
    )
    pipe = Pipeline([("tfidf", tfidf), ("clf", rf)])
    return pipe


def your_model():
    """Return a baseline random forest pipeline; tune with tune.py for best params."""
    return build_pipeline()


if __name__ == "__main__":
    X, y = load_dataset(DATA_PATH)
    model = your_model()
    print("Created random forest pipeline instance. Use tune.py for tuning.")