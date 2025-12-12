"""Quick SHAP global plot for RandomForest with specified parameters.
Runs on a small sample for speed and saves outputs in random_forest/results.
"""

import os
import sys
import random

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model import load_dataset, build_pipeline, DATA_PATH
import shap
import numpy as np


def main():
    # Fixed best params provided
    best_params = {
        "tfidf__ngram_range": (1, 1),
        "clf__n_estimators": 200,
        "clf__min_samples_split": 2,
        "clf__max_features": "sqrt",
        "clf__max_depth": None,
    }

    # Load and subsample for speed
    X, y = load_dataset(DATA_PATH)
    n = len(X)
    idx = list(range(n))
    random.seed(42)
    random.shuffle(idx)
    # Use up to 1000 samples to fit quickly
    sample_size = min(1000, n)
    sel = idx[:sample_size]
    X_small = [X[i] for i in sel]
    y_small = [y[i] for i in sel]

    # Build and fit pipeline
    pipe = build_pipeline()
    pipe.set_params(**best_params)
    pipe.fit(X_small, y_small)

    # SHAP global with a tiny background for speed (TreeExplainer directly here)
    tfidf = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    # Background: smaller subset for memory safety
    bg_size = min(200, len(X_small))
    X_bg_text = X_small[:bg_size]
    X_bg = tfidf.transform(X_bg_text)
    # Densify for TreeExplainer
    X_bg_dense = X_bg.toarray() if hasattr(X_bg, "toarray") else X_bg
    masker = shap.maskers.Independent(X_bg_dense)
    explainer = shap.TreeExplainer(clf, data=masker, feature_perturbation="interventional", model_output="raw")
    shap_values = explainer.shap_values(X_bg_dense, check_additivity=False)
    # Select positive class or reduce extra dimensions
    if isinstance(shap_values, list) and len(shap_values) >= 2:
        sv = shap_values[1]
    else:
        sv = shap_values
    # Reduce any trailing dims beyond (samples, features)
    while hasattr(sv, 'ndim') and sv.ndim > 2:
        sv = sv.mean(axis=-1)
    
    sv = np.array(sv)
    feat_names = tfidf.get_feature_names_out()
    top_k = 20
    
    # Separate positive (→ FAKE) and negative (→ REAL) contributions
    fake_mean = np.maximum(sv, 0).mean(axis=0)
    real_mean = np.abs(np.minimum(sv, 0)).mean(axis=0)
    
    # Get top-k for each direction
    top_fake_idx = np.argsort(fake_mean)[-top_k:][::-1]
    top_real_idx = np.argsort(real_mean)[-top_k:][::-1]
    
    top_fake_scores = fake_mean[top_fake_idx]
    top_fake_feats = feat_names[top_fake_idx]
    top_real_scores = real_mean[top_real_idx]
    top_real_feats = feat_names[top_real_idx]
    
    # Save directional bar plot (two panels)
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    
    # FAKE panel
    ax1.barh(range(len(top_fake_feats))[::-1], top_fake_scores[::-1], color="#ef4444")
    ax1.set_yticks(range(len(top_fake_feats))[::-1])
    ax1.set_yticklabels(top_fake_feats[::-1])
    ax1.set_xlabel("Mean SHAP (importance)")
    ax1.set_title("Top Words → FAKE")
    
    # REAL panel
    ax2.barh(range(len(top_real_feats))[::-1], top_real_scores[::-1], color="#22c55e")
    ax2.set_yticks(range(len(top_real_feats))[::-1])
    ax2.set_yticklabels(top_real_feats[::-1])
    ax2.set_xlabel("Mean SHAP (importance)")
    ax2.set_title("Top Words → REAL")
    
    plt.suptitle("Top Features by SHAP (RandomForest TF-IDF)", fontsize=14)
    plt.tight_layout()
    shap_plot_path = os.path.join(RESULTS_DIR, "shap_top_words.jpg")
    plt.savefig(shap_plot_path, format="jpg", dpi=200)
    plt.close()
    
    # Save TXT with both directions
    shap_txt_path = os.path.join(RESULTS_DIR, "shap_top_words.txt")
    with open(shap_txt_path, "w") as f:
        f.write("=== Random Forest Top Words (SHAP) ===\n\n")
        f.write("Top Words Pushing Toward FAKE:\n")
        f.write("-" * 40 + "\n")
        for i, (feat, score) in enumerate(zip(top_fake_feats, top_fake_scores), 1):
            f.write(f"{i:2}. {feat:<20} {score:.6f}\n")
        f.write("\n\nTop Words Pushing Toward REAL:\n")
        f.write("-" * 40 + "\n")
        for i, (feat, score) in enumerate(zip(top_real_feats, top_real_scores), 1):
            f.write(f"{i:2}. {feat:<20} {score:.6f}\n")
    print("Saved:", shap_plot_path, shap_txt_path)


if __name__ == "__main__":
    main()