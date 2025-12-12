"""Generate directional SHAP plot for Logistic Regression using best hyperparameters.
Creates Top Words → FAKE and Top Words → REAL plots."""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.model_selection import train_test_split
from model import load_dataset, build_pipeline, DATA_PATH, RANDOM_STATE

# Load best params from results
BEST_PARAMS_PATH = os.path.join(RESULTS_DIR, "best_params.json")


def load_best_params():
    """Load best hyperparameters from JSON file."""
    with open(BEST_PARAMS_PATH, "r") as f:
        data = json.load(f)
    params = data["best_params"]
    # Convert ngram_range list to tuple
    if "tfidf__ngram_range" in params and isinstance(params["tfidf__ngram_range"], list):
        params["tfidf__ngram_range"] = tuple(params["tfidf__ngram_range"])
    return params


def get_directional_importance(shap_values, feature_names, top_k=20):
    """Separate features by direction: positive (FAKE) vs negative (REAL)."""
    # Mean SHAP value per feature (with sign)
    mean_shap = shap_values.mean(axis=0)
    
    # Separate positive (push toward FAKE=1) and negative (push toward REAL=0)
    fake_indices = np.where(mean_shap > 0)[0]
    real_indices = np.where(mean_shap < 0)[0]
    
    # Get top-k for each direction
    fake_scores = [(feature_names[i], mean_shap[i]) for i in fake_indices]
    fake_scores = sorted(fake_scores, key=lambda x: x[1], reverse=True)[:top_k]
    
    real_scores = [(feature_names[i], abs(mean_shap[i])) for i in real_indices]
    real_scores = sorted(real_scores, key=lambda x: x[1], reverse=True)[:top_k]
    
    return fake_scores, real_scores


def save_directional_plot(fake_scores, real_scores, out_path):
    """Save side-by-side horizontal bar plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    
    # Fake words (left panel)
    if fake_scores:
        words_f = [w for w, _ in fake_scores][::-1]
        scores_f = [s for _, s in fake_scores][::-1]
        ax1.barh(range(len(words_f)), scores_f, color="#ef4444")
        ax1.set_yticks(range(len(words_f)))
        ax1.set_yticklabels(words_f)
    ax1.set_xlabel("Mean SHAP Value")
    ax1.set_title("Top Words → FAKE")
    
    # Real words (right panel)
    if real_scores:
        words_r = [w for w, _ in real_scores][::-1]
        scores_r = [s for _, s in real_scores][::-1]
        ax2.barh(range(len(words_r)), scores_r, color="#22c55e")
        ax2.set_yticks(range(len(words_r)))
        ax2.set_yticklabels(words_r)
    ax2.set_xlabel("Mean |SHAP| Value")
    ax2.set_title("Top Words → REAL")
    
    plt.suptitle("Logistic Regression Feature Importance (SHAP)", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_path}")


def save_directional_txt(fake_scores, real_scores, out_path):
    """Save top words to text file."""
    with open(out_path, "w") as f:
        f.write("=== Logistic Regression Top Words (SHAP) ===\n\n")
        f.write("Top Words Pushing Toward FAKE:\n")
        f.write("-" * 40 + "\n")
        for i, (word, score) in enumerate(fake_scores, 1):
            f.write(f"{i:2}. {word:<25} {score:.6f}\n")
        
        f.write("\n\nTop Words Pushing Toward REAL:\n")
        f.write("-" * 40 + "\n")
        for i, (word, score) in enumerate(real_scores, 1):
            f.write(f"{i:2}. {word:<25} {score:.6f}\n")
    
    print(f"Saved text: {out_path}")


def main():
    print("=" * 50)
    print("Logistic Regression SHAP Analysis")
    print("=" * 50)
    
    # Load best params
    print("Loading best hyperparameters...")
    best_params = load_best_params()
    print(f"Best params: {best_params}")
    
    # Load data
    print("Loading data...")
    X, y = load_dataset(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Build and train pipeline with best params
    print("Training model with best params...")
    pipe = build_pipeline()
    pipe.set_params(**best_params)
    pipe.fit(X_train, y_train)
    
    # Extract components
    tfidf = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    feature_names = tfidf.get_feature_names_out()
    
    # Transform test data
    X_test_tfidf = tfidf.transform(X_test)
    
    # Use SHAP LinearExplainer
    print("Computing SHAP values (this may take a few minutes)...")
    import shap
    
    # Sample background data for efficiency
    n_background = min(100, X_test_tfidf.shape[0])
    background_idx = np.random.choice(X_test_tfidf.shape[0], n_background, replace=False)
    background = X_test_tfidf[background_idx]
    
    explainer = shap.LinearExplainer(clf, background, feature_perturbation="interventional")
    
    # Compute SHAP values on a sample
    n_explain = min(500, X_test_tfidf.shape[0])
    explain_idx = np.random.choice(X_test_tfidf.shape[0], n_explain, replace=False)
    X_explain = X_test_tfidf[explain_idx]
    
    shap_values = explainer.shap_values(X_explain)
    
    # Handle array shape
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Class 1 (FAKE)
    if hasattr(shap_values, "values"):
        shap_values = shap_values.values
    if hasattr(shap_values, "toarray"):
        shap_values = shap_values.toarray()
    
    print(f"SHAP values shape: {shap_values.shape}")
    
    # Get directional importance
    fake_scores, real_scores = get_directional_importance(shap_values, feature_names, top_k=20)
    
    print(f"Found {len(fake_scores)} fake-indicating words, {len(real_scores)} real-indicating words")
    
    # Save outputs
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_directional_plot(fake_scores, real_scores, os.path.join(RESULTS_DIR, "shap_top_words.jpg"))
    save_directional_txt(fake_scores, real_scores, os.path.join(RESULTS_DIR, "shap_top_words.txt"))
    
    print("\n" + "=" * 50)
    print("Done!")
    print("=" * 50)


if __name__ == "__main__":
    main()
