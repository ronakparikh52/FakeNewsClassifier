import os
import time
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    average_precision_score,
    f1_score,
    roc_curve,
    RocCurveDisplay,
)
from sklearn.pipeline import Pipeline

# Optional heavy imports are kept local inside functions where used to avoid import-time overhead


def compute_classification_metrics(y_true, y_pred, y_prob):
    """Compute core classification metrics (accuracy, precision, recall, F1, ROC-AUC, AUPRC) plus confusion matrix.
    Returns a dictionary keyed by metric names for downstream reporting and persistence."""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    roc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc,
        "auprc": auprc,
        "confusion_matrix": cm,
    }


def compute_search_metrics(y_true, y_pred, start_time):
    """Compute minimal search metrics: F1 and elapsed time. Keeps searches fast by avoiding proba.
    Use for comparing hyperparameter searches when optimizing F1."""
    f1 = f1_score(y_true, y_pred)
    elapsed = time.time() - start_time
    return {"F1": f1, "Time": elapsed}


def get_roc_data(y_true, y_prob):
    """Generate ROC curve data (arrays of false positive rate and true positive rate).
    Returned dict is suitable for serialization or plotting routines."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return {"fpr": fpr, "tpr": tpr}


def run_grid_search(pipeline, param_grid, X_train, y_train, X_test, y_test, scoring="f1", cv=5):
    """Run GridSearchCV optimizing F1 and evaluate best on test set. Returns perf dict and best params.
    Skips probability/ROC during search to speed up."""
    start = time.time()
    grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring=scoring)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    y_pred = best.predict(X_test)
    result = compute_search_metrics(y_test, y_pred, start)
    return result, grid.best_params_


def run_random_search(pipeline, param_grid, X_train, y_train, X_test, y_test, scoring="f1", cv=5, fraction=0.33, random_state=None):
    """Run RandomizedSearchCV optimizing F1; evaluate best on test set. Returns perf dict and best params.
    Samples a fraction of the grid to reduce runtime."""
    start = time.time()
    total = 1
    for v in param_grid.values():
        total *= len(v)
    n_iter = max(1, int(total * fraction))
    rand = RandomizedSearchCV(pipeline, param_grid, n_iter=n_iter, cv=cv, scoring=scoring, random_state=random_state)
    rand.fit(X_train, y_train)
    best = rand.best_estimator_
    y_pred = best.predict(X_test)
    result = compute_search_metrics(y_test, y_pred, start)
    return result, rand.best_params_


def save_confusion_matrix(cm, out_path):
    """Render and persist a confusion matrix image to out_path as a JPG for visual inspection.
    Returns the file path after successful save."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(5, 4))
    plt.title("Confusion Matrix")
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
    """Plot ROC curve from true labels and predicted probabilities and save as JPG.
    Returns path to the saved image for later reference."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(out_path, format="jpg", dpi=200)
    plt.close()
    return out_path


def write_metrics_file(metrics, cv_summary=None, best_params=None, out_path="metrics.txt"):
    """Write aggregated metrics, optional CV summary, and best hyperparameters to a text file.
    Produces a human-readable artifact consolidating evaluation results."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("=== Holdout Test Metrics ===\n")
        for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "auprc"]:
            f.write(f"{k}: {metrics[k]:.4f}\n")
        f.write("Confusion Matrix:\n")
        cm = metrics["confusion_matrix"]
        for row in cm:
            row_text_parts = []
            for x in row:
                row_text_parts.append(str(x))
            row_text = " ".join(row_text_parts)
            f.write("\t" + row_text + "\n")
        if cv_summary is not None:
            f.write("\n=== Cross-Validation (K-Fold) ===\n")
            for m in cv_summary:
                vals = cv_summary[m]
                f.write(f"{m}: mean={vals['mean']:.4f}, std={vals['std']:.4f}\n")
        if best_params is not None:
            f.write("\n=== Best Hyperparameters ===\n")
            for p in best_params:
                f.write(f"{p}: {best_params[p]}\n")
    return out_path


def explain_shap_linear_text(pipeline: Pipeline, X_train_text, out_dir, max_background=2000, top_k=20):
    """Compute SHAP global importance for a linear text model and save top-k terms as bar plot and TXT.
    Shows directional importance: words pushing toward FAKE vs REAL.
    Uses a sample of training documents as background for faster computation."""
    import shap
    os.makedirs(out_dir, exist_ok=True)
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    # Background sample for SHAP (limit size for speed)
    if len(X_train_text) > max_background:
        bg_idx = random.sample(range(len(X_train_text)), max_background)
        X_bg_text = [X_train_text[i] for i in bg_idx]
    else:
        X_bg_text = list(X_train_text)
    X_bg = tfidf.transform(X_bg_text)
    explainer = shap.LinearExplainer(clf, X_bg, feature_perturbation="interventional")
    # Explain same background (good enough for global importance)
    shap_values = explainer.shap_values(X_bg)
    # For binary classifier, shap_values is (n_samples, n_features)
    sv = shap_values
    # If sparse-like, densify
    if hasattr(sv, "toarray"):
        sv = sv.toarray()
    sv = np.array(sv)
    # If numpy matrix, convert
    if hasattr(sv, "A"):
        sv = np.asarray(sv)
    
    # Map indices to feature names
    feat_names = tfidf.get_feature_names_out()
    
    # Separate positive (→ FAKE) and negative (→ REAL) contributions
    # Positive SHAP values push toward class 1 (FAKE)
    fake_mean = np.maximum(sv, 0).mean(axis=0)
    real_mean = np.abs(np.minimum(sv, 0)).mean(axis=0)
    
    # Flatten if needed
    if hasattr(fake_mean, "A1"):
        fake_mean = fake_mean.A1
    if hasattr(real_mean, "A1"):
        real_mean = real_mean.A1
    fake_mean = np.asarray(fake_mean).flatten()
    real_mean = np.asarray(real_mean).flatten()
    
    # Get top-k for each direction
    top_fake_idx = np.argsort(fake_mean)[-top_k:][::-1]
    top_real_idx = np.argsort(real_mean)[-top_k:][::-1]
    
    top_fake_scores = fake_mean[top_fake_idx]
    top_fake_feats = feat_names[top_fake_idx]
    top_real_scores = real_mean[top_real_idx]
    top_real_feats = feat_names[top_real_idx]
    
    # Save directional bar plot (two panels)
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
    
    plt.suptitle("Top Features by SHAP (LogReg TF-IDF)", fontsize=14)
    plt.tight_layout()
    shap_plot_path = os.path.join(out_dir, "shap_top_words.jpg")
    plt.savefig(shap_plot_path, format="jpg", dpi=200)
    plt.close()
    
    # Save TXT with both directions
    shap_txt_path = os.path.join(out_dir, "shap_top_words.txt")
    with open(shap_txt_path, "w") as f:
        f.write("=== Logistic Regression Top Words (SHAP) ===\n\n")
        f.write("Top Words Pushing Toward FAKE:\n")
        f.write("-" * 40 + "\n")
        for i, (feat, score) in enumerate(zip(top_fake_feats, top_fake_scores), 1):
            f.write(f"{i:2}. {feat:<20} {score:.6f}\n")
        f.write("\n\nTop Words Pushing Toward REAL:\n")
        f.write("-" * 40 + "\n")
        for i, (feat, score) in enumerate(zip(top_real_feats, top_real_scores), 1):
            f.write(f"{i:2}. {feat:<20} {score:.6f}\n")
    return shap_plot_path, shap_txt_path


# LIME removed per project requirements; using SHAP for interpretability.


__all__ = [
    "compute_classification_metrics",
    "compute_search_metrics",
    "get_roc_data",
    "run_grid_search",
    "run_random_search",
    "save_confusion_matrix",
    "save_roc_curve",
    "write_metrics_file",
    "explain_shap_linear_text",
]

# Global top words via coefficients removed; prefer SHAP global importance.
