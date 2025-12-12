"""Generate interpretability outputs for fine-tuned BERT model.
Uses Integrated Gradients to compute token importance."""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Add project root to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from bert.model import load_dataset, get_tokenizer, get_model, get_device, DATA_PATH, MAX_LENGTH

RESULTS_DIR = os.path.join(CURRENT_DIR, "results")
MODEL_PATH = os.path.join(RESULTS_DIR, "bert_finetuned.pt")

# Reduced for faster execution (~5-8 min)
NUM_EXAMPLES = 50
STEPS = 20  # Reduced from 50 for speed


def compute_integrated_gradients(model, tokenizer, text, label, device, steps=STEPS):
    """Compute Integrated Gradients for a single text."""
    
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length"
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Get embeddings layer
    embeddings = model.bert.embeddings.word_embeddings
    
    # Baseline: all PAD tokens
    baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
    
    # Get embeddings without tracking gradients
    with torch.no_grad():
        baseline_embeds = embeddings(baseline_ids)
        input_embeds = embeddings(input_ids)
    
    # Compute gradients at each interpolation step
    all_grads = []
    for step in range(steps + 1):
        alpha = step / steps
        
        # Interpolate and enable gradients
        scaled = baseline_embeds + alpha * (input_embeds - baseline_embeds)
        scaled = scaled.detach().clone()
        scaled.requires_grad_(True)
        
        # Forward pass
        outputs = model(
            inputs_embeds=scaled,
            attention_mask=attention_mask
        )
        
        # Backward pass on target class
        target_score = outputs.logits[0, label]
        target_score.backward()
        
        # Store gradient
        all_grads.append(scaled.grad.detach().clone())
        
        # Clear gradients
        model.zero_grad()
    
    # Average gradients across steps
    avg_grads = torch.stack(all_grads).mean(dim=0)
    
    # Integrated gradients = (input - baseline) * avg_gradients
    integrated_grads = (input_embeds - baseline_embeds).detach() * avg_grads
    
    # Sum over embedding dimension for per-token importance
    token_attributions = integrated_grads.sum(dim=-1).squeeze().cpu().numpy()
    
    # Get tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().cpu().numpy())
    mask = attention_mask.squeeze().cpu().numpy()
    
    # Return token-attribution pairs (only real tokens, not padding)
    results = []
    for tok, attr, m in zip(tokens, token_attributions, mask):
        if m == 1:
            results.append((tok, float(attr)))
    
    return results


def aggregate_importance(all_attributions, top_k=20):
    """Aggregate attributions across examples to get global importance."""
    fake_scores = defaultdict(list)
    real_scores = defaultdict(list)
    
    for token, attr in all_attributions:
        # Skip special tokens, subwords, and short tokens
        if token in ["[CLS]", "[SEP]", "[PAD]", "[UNK]"]:
            continue
        if token.startswith("##"):
            continue
        if len(token) < 2:
            continue
        
        if attr > 0:
            fake_scores[token].append(attr)
        else:
            real_scores[token].append(abs(attr))
    
    # Average scores per token (require at least 2 occurrences)
    fake_avg = {tok: np.mean(scores) for tok, scores in fake_scores.items() if len(scores) >= 2}
    real_avg = {tok: np.mean(scores) for tok, scores in real_scores.items() if len(scores) >= 2}
    
    # Sort and get top-k
    top_fake = sorted(fake_avg.items(), key=lambda x: x[1], reverse=True)[:top_k]
    top_real = sorted(real_avg.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    return top_fake, top_real


def save_plot(top_fake, top_real, out_path):
    """Save horizontal bar plot of top words."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    
    if top_fake:
        words_f = [w for w, _ in top_fake][::-1]
        scores_f = [s for _, s in top_fake][::-1]
        ax1.barh(range(len(words_f)), scores_f, color="#ef4444")
        ax1.set_yticks(range(len(words_f)))
        ax1.set_yticklabels(words_f)
    ax1.set_xlabel("Mean Attribution")
    ax1.set_title("Top Words → FAKE")
    
    if top_real:
        words_r = [w for w, _ in top_real][::-1]
        scores_r = [s for _, s in top_real][::-1]
        ax2.barh(range(len(words_r)), scores_r, color="#22c55e")
        ax2.set_yticks(range(len(words_r)))
        ax2.set_yticklabels(words_r)
    ax2.set_xlabel("Mean Attribution")
    ax2.set_title("Top Words → REAL")
    
    plt.suptitle("BERT Token Importance (Integrated Gradients)", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_path}")


def save_txt(top_fake, top_real, out_path):
    """Save top words to text file."""
    with open(out_path, "w") as f:
        f.write("=== BERT Top Words (Integrated Gradients) ===\n\n")
        f.write("Top Words Pushing Toward FAKE:\n")
        f.write("-" * 40 + "\n")
        for i, (word, score) in enumerate(top_fake, 1):
            f.write(f"{i:2}. {word:<20} {score:.6f}\n")
        
        f.write("\n\nTop Words Pushing Toward REAL:\n")
        f.write("-" * 40 + "\n")
        for i, (word, score) in enumerate(top_real, 1):
            f.write(f"{i:2}. {word:<20} {score:.6f}\n")
    
    print(f"Saved text: {out_path}")


def main():
    print("=" * 50)
    print("BERT Interpretability Analysis")
    print("=" * 50)
    
    device = get_device()
    print(f"Device: {device}")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Please run bert/tune.py first to train the model.")
        return
    
    # Load tokenizer and model
    print("Loading model...")
    tokenizer = get_tokenizer()
    model = get_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from: {MODEL_PATH}")
    
    # Load data
    print("Loading data...")
    texts, labels = load_dataset(DATA_PATH)
    
    # Sample examples (balanced)
    np.random.seed(42)
    fake_idx = [i for i, l in enumerate(labels) if l == 1]
    real_idx = [i for i, l in enumerate(labels) if l == 0]
    
    n_each = NUM_EXAMPLES // 2
    sample_idx = (
        list(np.random.choice(fake_idx, n_each, replace=False)) +
        list(np.random.choice(real_idx, n_each, replace=False))
    )
    
    print(f"\nAnalyzing {NUM_EXAMPLES} examples ({STEPS} gradient steps each)")
    
    # Collect attributions
    all_attributions = []
    success_count = 0
    
    for i, idx in enumerate(sample_idx):

        text = texts[idx]
        label = labels[idx]
        
        try:
            with torch.enable_grad():
                attribs = compute_integrated_gradients(model, tokenizer, text, label, device)
            all_attributions.extend(attribs)
            success_count += 1
        except Exception as e:
            # Silent skip - don't print every error
            continue
    
    print(f"Processed {success_count}/{NUM_EXAMPLES} examples successfully")
    print(f"Collected {len(all_attributions)} token attributions")
    
    print("Aggregating results...")
    top_fake, top_real = aggregate_importance(all_attributions, top_k=20)    
    # Save outputs
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_plot(top_fake, top_real, os.path.join(RESULTS_DIR, "shap_top_words.jpg"))
    save_txt(top_fake, top_real, os.path.join(RESULTS_DIR, "shap_top_words.txt"))
    
    print("Done!")


if __name__ == "__main__":
    main()
