"""BERT model and dataset utilities for fake news classification.
Provides dataset class, tokenizer setup, and model wrapper for fine-tuning."""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "WELFake_Dataset.csv")
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 256  # Max tokens; balance between context and memory
RANDOM_STATE = 42


def load_dataset(csv_path: str):
    """Load dataset CSV, normalize labels, concatenate title+body, return texts and labels."""
    df = pd.read_csv(csv_path)
    expected = {"title", "text", "label"}
    missing = expected - set(df.columns)
    if len(missing) > 0:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
    if df["label"].dtype == object:
        df["label"] = df["label"].str.lower().map({"fake": 1, "real": 0})
    df["label"] = df["label"].astype(int)
    texts = (
        df["title"].fillna("").astype(str).str.strip() + " [SEP] " + df["text"].fillna("").astype(str).str.strip()
    ).tolist()
    labels = df["label"].tolist()
    return texts, labels


class FakeNewsDataset(Dataset):
    """PyTorch Dataset for tokenized fake news articles."""

    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def get_tokenizer(model_name=MODEL_NAME):
    """Return the BERT tokenizer for the specified model."""
    return BertTokenizer.from_pretrained(model_name)


def get_model(model_name=MODEL_NAME, num_labels=2):
    """Return a BERT model for sequence classification with the specified number of labels."""
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model


def get_device():
    """Return the best available device: CUDA > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


if __name__ == "__main__":
    texts, labels = load_dataset(DATA_PATH)
    print(f"Loaded {len(texts)} samples.")
    tokenizer = get_tokenizer()
    model = get_model()
    device = get_device()
    print(f"Device: {device}")
    print("BERT model and tokenizer ready. Use tune.py for fine-tuning.")
