import os
import sys
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

"""Tune TF-IDF + Logistic Regression using grid and random search helpers.
Outputs metrics (AUC, AUPRC, F1, Time), ROC data, best hyperparameters, and plots."""

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model import load_dataset, build_pipeline, DATA_PATH, RANDOM_STATE
from metrics import (
    run_grid_search,
    run_random_search,
    save_confusion_matrix,
    save_roc_curve,
    explain_shap_linear_text,
    compute_classification_metrics,
)


def preprocess_data(X_train, X_test):
    """Return inputs unchanged; TF-IDF vectorization is performed inside the pipeline.
    Extend here if additional feature preprocessing is required later."""
    return X_train, X_test


def get_parameter_grid():
    return {
        "tfidf__ngram_range": [(1,1), (1,2)],
        "clf__C": [0.25, 0.5, 1.0, 2.0],
        "clf__solver": ["liblinear"],
        "clf__penalty": ["l2"],
        "clf__max_iter": [3000]
    }


def main():
    """Run searches, pick best by AUC, save artifacts and interpretability outputs.
    No CLI args; paths default to the logistic_regression folder."""
    roc_output = os.path.join(RESULTS_DIR, "roc_curve_data.csv")
    best_params_output = os.path.join(RESULTS_DIR, "best_params.json")
    metrics_output = os.path.join(RESULTS_DIR, "metrics.txt")

    X, y = load_dataset(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_test = preprocess_data(X_train, X_test)

    pipe = build_pipeline()
    pgrid = get_parameter_grid()

    print("Running GridSearchCV (optimize F1)...")
    grid_perf, grid_params = run_grid_search(pipe, pgrid, X_train, y_train, X_test, y_test)
    print("Running RandomizedSearchCV (optimize F1)...")
    rand_perf, rand_params = run_random_search(pipe, pgrid, X_train, y_train, X_test, y_test, random_state=RANDOM_STATE)

    # Choose best by F1
    best_overall = ("Grid", grid_perf, grid_params) if grid_perf["F1"] >= rand_perf["F1"] else ("Random", rand_perf, rand_params)
    best_label, best_perf, best_params = best_overall

    # Confusion matrix from best model retrained on train set
    best_pipe = build_pipeline()
    best_pipe.set_params(**best_params)
    best_pipe.fit(X_train, y_train)
    y_pred_best = best_pipe.predict(X_test)
    y_prob_best = best_pipe.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred_best)
    cm_path = save_confusion_matrix(cm, os.path.join(RESULTS_DIR, "confusion_matrix.jpg"))
    roc_path = save_roc_curve(y_test, y_prob_best, os.path.join(RESULTS_DIR, "roc_curve.jpg"))

    # Write ROC data CSV from final best model
    from metrics import get_roc_data
    roc_data = get_roc_data(y_test, y_prob_best)
    roc_df = pd.DataFrame({"fpr": roc_data["fpr"], "tpr": roc_data["tpr"], "model": [best_label]*len(roc_data["fpr"])})
    roc_df.to_csv(roc_output, index=False)

    # Write best params JSON
    with open(best_params_output, 'w') as f:
        json.dump({"best_search": best_label, "best_params": best_params}, f, indent=2)

    # Compute full binary classification metrics for the best model
    cls_metrics = compute_classification_metrics(y_test, y_pred_best, y_prob_best)

    # Write metrics summary text
    with open(metrics_output, 'w') as f:
        f.write("=== Best Search Type ===\n" + best_label + "\n\n")
        f.write("=== Performance (F1/Time) ===\n")
        for name, perf in [("Grid", grid_perf), ("Random", rand_perf)]:
            f.write(f"{name}: F1={perf['F1']:.4f}, Time={perf['Time']:.2f}s\n")
        f.write("\n=== Best Parameters ===\n")
        for k, v in best_params.items():
            f.write(f"{k}: {v}\n")
        f.write("\n=== Best Model Metrics (Binary Classification) ===\n")
        f.write(f"accuracy: {cls_metrics['accuracy']:.4f}\n")
        f.write(f"precision: {cls_metrics['precision']:.4f}\n")
        f.write(f"recall: {cls_metrics['recall']:.4f}\n")
        f.write(f"f1: {cls_metrics['f1']:.4f}\n")
        f.write(f"roc_auc: {cls_metrics['roc_auc']:.4f}\n")
        f.write(f"auprc: {cls_metrics['auprc']:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        for row in cm:
            f.write("\t" + " ".join(str(x) for x in row) + "\n")

    # SHAP global top words and LIME local explanations
    try:
        shap_plot, shap_txt = explain_shap_linear_text(best_pipe, X_train, RESULTS_DIR, max_background=300, top_k=20)
        print("SHAP outputs:", shap_plot, shap_txt)
    except Exception as e:
        print("SHAP explanation failed:", e)
    # LIME and coefficient-based global summaries removed; SHAP global only.

    print("Best params JSON:", best_params_output)
    print("ROC data CSV:", roc_output)
    print("Metrics summary:", metrics_output)
    print("Confusion matrix image:", cm_path)
    print("ROC curve image:", roc_path)


if __name__ == '__main__':
    main()
