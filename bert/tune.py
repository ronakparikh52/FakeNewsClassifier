"""Fine-tune BERT for fake news classification.
Trains with AdamW optimizer and linear LR schedule; evaluates with PR-AUC and F2 score."""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    average_precision_score,
    fbeta_score,
    confusion_matrix,
    roc_curve,
)
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model import (
    load_dataset,
    FakeNewsDataset,
    get_tokenizer,
    get_model,
    get_device,
    DATA_PATH,
    RANDOM_STATE,
    MAX_LENGTH,
)

# ============ Hyperparameters ============
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
MAX_SAMPLES = None  # Set to int (e.g., 5000) for faster testing; None for full dataset


def compute_metrics(y_true, y_pred, y_prob):
    """Compute classification metrics including PR-AUC and F2 score."""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    pr_auc = average_precision_score(y_true, y_prob)
    f2 = fbeta_score(y_true, y_pred, beta=2)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "f2": f2,
        "confusion_matrix": cm,
    }


def save_confusion_matrix(cm, out_path):
    """Save confusion matrix as a JPG image."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(5, 4))
    plt.title("Confusion Matrix (BERT)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    plt.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(out_path, format="jpg", dpi=200)
    plt.close()
    return out_path


def save_roc_curve(y_true, y_prob, out_path):
    """Save ROC curve as a JPG image."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (BERT)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, format="jpg", dpi=200)
    plt.close()
    return out_path


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, total_epochs, batch_size):
    """Train the model for one epoch and return average loss."""
    model.train()
    total_loss = 0
    total_batches = len(dataloader)
    total_samples = total_batches * batch_size
    last_log_time = time.time()
    log_interval = 120  # Log every 2 minutes (120 seconds)
    
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        
        # Log progress every 2 minutes
        current_time = time.time()
        if current_time - last_log_time >= log_interval:
            samples_processed = (batch_idx + 1) * batch_size
            progress_pct = (batch_idx + 1) / total_batches * 100
            avg_loss = total_loss / (batch_idx + 1)
            elapsed = current_time - last_log_time
            print(f"  [Epoch {epoch}/{total_epochs}] Batch {batch_idx + 1}/{total_batches} "
                  f"({progress_pct:.1f}%) | Samples: {samples_processed}/{total_samples} | "
                  f"Avg Loss: {avg_loss:.4f}")
            last_log_time = current_time
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model and return predictions, probabilities, and labels."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def main():
    """Fine-tune BERT on WELFake dataset and save evaluation artifacts."""
    print("Loading data...")
    texts, labels = load_dataset(DATA_PATH)

    # Optionally limit samples for faster testing
    if MAX_SAMPLES is not None and MAX_SAMPLES < len(texts):
        indices = list(range(len(texts)))
        np.random.seed(RANDOM_STATE)
        np.random.shuffle(indices)
        indices = indices[:MAX_SAMPLES]
        texts = [texts[i] for i in indices]
        labels = [labels[i] for i in indices]
        print(f"Using {MAX_SAMPLES} samples for faster testing.")

    tokenizer = get_tokenizer()
    dataset = FakeNewsDataset(texts, labels, tokenizer, max_length=MAX_LENGTH)

    # Train/val/test split: 70/15/15
    n = len(dataset)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val
    generator = torch.Generator().manual_seed(RANDOM_STATE)
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    device = get_device()
    print(f"Device: {device}")
    model = get_model()
    model.to(device)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # Training loop
    best_val_f2 = 0
    best_model_state = None
    epochs_without_improvement = 0
    early_stopping_patience = 1  # Stop if no improvement for 1 epoch
    history = {"epoch": [], "train_loss": [], "val_pr_auc": [], "val_f2": []}

    print("Starting training...")
    print(f"Total samples: {n_train} train, {n_val} val, {n_test} test")
    print(f"Batches per epoch: {len(train_loader)} | Logging progress every 2 minutes")
    print(f"Early stopping: patience = {early_stopping_patience} epoch(s)")
    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        print(f"\n=== Epoch {epoch}/{EPOCHS} ===")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch, EPOCHS, BATCH_SIZE)
        epoch_time = time.time() - epoch_start
        y_val_true, y_val_pred, y_val_prob = evaluate(model, val_loader, device)
        val_metrics = compute_metrics(y_val_true, y_val_pred, y_val_prob)
        print(f"Epoch {epoch}/{EPOCHS} Complete ({epoch_time/60:.1f} min) | Loss: {train_loss:.4f} | Val PR-AUC: {val_metrics['pr_auc']:.4f} | Val F2: {val_metrics['f2']:.4f}")
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_pr_auc"].append(val_metrics["pr_auc"])
        history["val_f2"].append(val_metrics["f2"])
        if val_metrics["f2"] > best_val_f2:
            best_val_f2 = val_metrics["f2"]
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
            print(f"  ↑ New best F2: {best_val_f2:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s)")
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs (no improvement for {early_stopping_patience} epoch(s))")
                break

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f}s")

    # Load best model and evaluate on test set
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    y_test_true, y_test_pred, y_test_prob = evaluate(model, test_loader, device)
    test_metrics = compute_metrics(y_test_true, y_test_pred, y_test_prob)

    # Save artifacts
    cm_path = save_confusion_matrix(test_metrics["confusion_matrix"], os.path.join(RESULTS_DIR, "confusion_matrix.jpg"))
    roc_path = save_roc_curve(y_test_true, y_test_prob, os.path.join(RESULTS_DIR, "roc_curve.jpg"))

    # ROC data CSV
    fpr, tpr, _ = roc_curve(y_test_true, y_test_prob)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    roc_csv_path = os.path.join(RESULTS_DIR, "roc_curve_data.csv")
    roc_df.to_csv(roc_csv_path, index=False)

    # Metrics summary
    metrics_path = os.path.join(RESULTS_DIR, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("=== BERT Fine-Tuned Model ===\n\n")
        f.write(f"Training Time: {training_time:.2f}s\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n")
        f.write(f"Max Length: {MAX_LENGTH}\n\n")
        f.write("=== Test Set Metrics ===\n")
        f.write(f"accuracy: {test_metrics['accuracy']:.4f}\n")
        f.write(f"precision: {test_metrics['precision']:.4f}\n")
        f.write(f"recall: {test_metrics['recall']:.4f}\n")
        f.write(f"f1: {test_metrics['f1']:.4f}\n")
        f.write(f"f2: {test_metrics['f2']:.4f}\n")
        f.write(f"pr_auc: {test_metrics['pr_auc']:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        for row in test_metrics["confusion_matrix"]:
            f.write("\t" + " ".join(str(x) for x in row) + "\n")
        f.write("\n=== Training History ===\n")
        for i, ep in enumerate(history["epoch"]):
            f.write(f"Epoch {ep}: loss={history['train_loss'][i]:.4f}, val_pr_auc={history['val_pr_auc'][i]:.4f}, val_f2={history['val_f2'][i]:.4f}\n")

    # Best params JSON
    best_params_path = os.path.join(RESULTS_DIR, "best_params.json")
    with open(best_params_path, "w") as f:
        json.dump({
            "model": "bert-base-uncased",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "max_length": MAX_LENGTH,
            "warmup_ratio": WARMUP_RATIO,
            "best_val_f2": best_val_f2,
        }, f, indent=2)

    # Save model checkpoint
    checkpoint_path = os.path.join(RESULTS_DIR, "bert_finetuned.pt")
    torch.save(best_model_state, checkpoint_path)

    print(f"Metrics: {metrics_path}")
    print(f"Best params: {best_params_path}")
    print(f"Confusion matrix: {cm_path}")
    print(f"ROC curve: {roc_path}")
    print(f"ROC data: {roc_csv_path}")
    print(f"Model checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
