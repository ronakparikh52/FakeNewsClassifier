"""Helpers to load WELFake data and construct a TF-IDF + Logistic Regression pipeline.
Provides a factory `your_model()` returning a tuned pipeline instance for reuse."""

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "WELFake_Dataset.csv")
RANDOM_STATE = 42


def load_dataset(csv_path: str):
    """Load dataset CSV, normalize labels, concatenate title+body, and return text list and label list.
    Raises ValueError if required columns are missing."""
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


def build_pipeline(C=1.0, solver="liblinear", penalty="l2", ngram_range=(1, 2)):
    """Create TF-IDF + LR pipeline with adjustable regularization and n-gram settings.
    Returns a scikit-learn Pipeline ready for fitting/predicting."""
    tfidf = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=ngram_range)
    lr = LogisticRegression(
        C=C,
        solver=solver,
        penalty=penalty if penalty != "none" else None,
        max_iter=500,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    pipe = Pipeline([("tfidf", tfidf), ("clf", lr)])
    return pipe


def your_model():
    """Return a preselected logistic regression pipeline matching tuned hyperparameters.
    Adjust values here if subsequent experiments find better settings."""
    return build_pipeline(C=1.0, solver="liblinear", penalty="l2", ngram_range=(1, 2))


if __name__ == "__main__":
    X, y = load_dataset(DATA_PATH)
    model = your_model()
    print("Created logistic regression pipeline instance. Use tune.py for tuning.")