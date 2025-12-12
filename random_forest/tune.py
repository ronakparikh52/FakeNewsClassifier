import os
import sys
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

"""Tune TF-IDF + RandomForest using grid and random search helpers.
Outputs F1/Time for searches, full metrics for best model, ROC data, and SHAP global files."""

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
    compute_classification_metrics,
    get_roc_data,
    explain_shap_tree_text,
)


def preprocess_data(X_train, X_test):
    """Return inputs unchanged; TF-IDF vectorization is inside the pipeline."""
    return X_train, X_test


def get_parameter_grid():
    return {
        "tfidf__ngram_range": [(1,1)],
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 10],
        "clf__max_features": ["sqrt"],
        "clf__min_samples_split": [2, 5],
    }


def main():
    """Run grid and random search optimizing F1, pick best, save artifacts in results folder."""
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

    # Retrain best on train and evaluate on test
    best_pipe = build_pipeline()
    best_pipe.set_params(**best_params)
    best_pipe.fit(X_train, y_train)
    y_pred_best = best_pipe.predict(X_test)
    y_prob_best = best_pipe.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred_best)
    cm_path = save_confusion_matrix(cm, os.path.join(RESULTS_DIR, "confusion_matrix.jpg"))
    roc_path = save_roc_curve(y_test, y_prob_best, os.path.join(RESULTS_DIR, "roc_curve.jpg"))

    # ROC data CSV from final best
    roc_data = get_roc_data(y_test, y_prob_best)
    roc_df = pd.DataFrame({"fpr": roc_data["fpr"], "tpr": roc_data["tpr"], "model": [best_label]*len(roc_data["fpr"])})
    roc_df.to_csv(roc_output, index=False)

    # Best params JSON
    with open(best_params_output, 'w') as f:
        json.dump({"best_search": best_label, "best_params": best_params}, f, indent=2)

    # Full binary classification metrics for best model
    cls_metrics = compute_classification_metrics(y_test, y_pred_best, y_prob_best)
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

    # SHAP global top words for RandomForest
    try:
        shap_plot, shap_txt = explain_shap_tree_text(best_pipe, X_train, RESULTS_DIR)
        print("SHAP RF outputs:", shap_plot, shap_txt)
    except Exception as e:
        print("SHAP RF explanation failed:", e)

    print("Best params JSON:", best_params_output)
    print("ROC data CSV:", roc_output)
    print("Metrics summary:", metrics_output)
    print("Confusion matrix image:", cm_path)
    print("ROC curve image:", roc_path)


if __name__ == '__main__':
    main()