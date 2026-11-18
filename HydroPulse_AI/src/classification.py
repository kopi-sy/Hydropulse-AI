"""
HydroPulse AI - Phase 3: Classification Models Module
Author: HydroPulse ML Engineering Team
"""

import os
import sys
import json
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import Dict, Any, Tuple
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from src.preprocessing import (
    separate_features_and_targets,
    FEATURE_COLUMNS,
    CONDITION_TO_INT,
    INT_TO_CONDITION,
)
from src.model_evaluation import (
    evaluate_classification_predictions,
    plot_combined_confusion_matrices,
)


def get_baseline_classifiers(random_state: int = 42) -> Dict[str, Any]:
    """Define baseline classification algorithms."""
    return {
        "Decision_Tree": DecisionTreeClassifier(random_state=random_state),
        "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
        "Gradient_Boosting": GradientBoostingClassifier(n_estimators=100, random_state=random_state),
    }


def evaluate_classifier_group_kfold(
    model_name: str,
    model_instance: Any,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 5,
) -> Dict[str, Any]:
    """
    Perform GroupKFold cross-validation on a classifier.
    Ensures zero filter cycle overlap between train and test sets.
    Computes fold-level metrics and out-of-fold overall metrics.
    """
    gkf = GroupKFold(n_splits=n_splits)

    fold_metrics = []
    oof_predictions = np.zeros(len(y), dtype=int)
    all_classes_present_per_fold = True

    print(f"  Evaluating {model_name} across {n_splits} GroupKFold splits...")

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        test_cycles = groups.iloc[test_idx].unique().tolist()

        # Check representation of classes in test fold
        test_classes = set(y_test.unique())
        if len(test_classes) < 3:
            all_classes_present_per_fold = False

        # Fit model
        model = type(model_instance)(**model_instance.get_params())
        model.fit(X_train, y_train)

        # Predict test fold
        y_pred = model.predict(X_test)
        oof_predictions[test_idx] = y_pred

        # Fold evaluation
        fold_eval = evaluate_classification_predictions(y_test.values, y_pred)
        fold_metrics.append({
            "fold": fold_idx + 1,
            "test_cycles": test_cycles,
            "accuracy": fold_eval["accuracy"],
            "macro_f1": fold_eval["f1_macro"],
            "weighted_f1": fold_eval["f1_weighted"],
            "warning_f1": fold_eval["per_class"]["Warning"]["f1_score"],
            "clogged_f1": fold_eval["per_class"]["Clogged"]["f1_score"],
        })

    # Summary of metrics across folds (Mean & Standard Deviation)
    acc_scores = [f["accuracy"] for f in fold_metrics]
    macro_f1_scores = [f["macro_f1"] for f in fold_metrics]
    weighted_f1_scores = [f["weighted_f1"] for f in fold_metrics]
    warning_f1_scores = [f["warning_f1"] for f in fold_metrics]
    clogged_f1_scores = [f["clogged_f1"] for f in fold_metrics]

    # Global out-of-fold evaluation
    oof_eval = evaluate_classification_predictions(y.values, oof_predictions)

    results = {
        "model_name": model_name,
        "n_splits": n_splits,
        "all_classes_present_in_every_test_split": all_classes_present_per_fold,
        "cv_summary": {
            "accuracy_mean": float(np.mean(acc_scores)),
            "accuracy_std": float(np.std(acc_scores)),
            "macro_f1_mean": float(np.mean(macro_f1_scores)),
            "macro_f1_std": float(np.std(macro_f1_scores)),
            "weighted_f1_mean": float(np.mean(weighted_f1_scores)),
            "weighted_f1_std": float(np.std(weighted_f1_scores)),
            "warning_f1_mean": float(np.mean(warning_f1_scores)),
            "warning_f1_std": float(np.std(warning_f1_scores)),
            "clogged_f1_mean": float(np.mean(clogged_f1_scores)),
            "clogged_f1_std": float(np.std(clogged_f1_scores)),
        },
        "fold_details": fold_metrics,
        "out_of_fold_metrics": oof_eval,
    }

    return results, oof_predictions


def get_default_dataset_path() -> str:
    """Resolve dataset path prioritizing the user's downloads folder directly."""
    direct_download_path = r"c:\Users\shyam\Downloads\Atomberg Filter\HydroPulse_synthetic_dataset.csv"
    if os.path.exists(direct_download_path):
        return direct_download_path
    local_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "HydroPulse_synthetic_dataset.csv")
    if os.path.exists(local_path):
        return os.path.abspath(local_path)
    return "HydroPulse_synthetic_dataset.csv"


def run_classification_pipeline(
    csv_path: str = None,
    reports_dir: str = None,
    visualizations_dir: str = None,
) -> Dict[str, Any]:
    """Execute classification pipeline across Decision Tree, Random Forest, and Gradient Boosting."""
    if csv_path is None:
        csv_path = get_default_dataset_path()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if reports_dir is None:
        reports_dir = os.path.join(base_dir, "reports")
    if visualizations_dir is None:
        visualizations_dir = os.path.join(base_dir, "visualizations")

    print("==================================================")
    print("   HydroPulse AI - Phase 3: Classification Models")
    print(f"   Dataset Source: {csv_path}")
    print("==================================================")

    df = pd.read_csv(csv_path)
    X, y_class, _, groups = separate_features_and_targets(df)

    models = get_baseline_classifiers()
    all_results = {}
    oof_preds_dict = {}
    cms_dict = {}

    for name, model in models.items():
        res, oof_pred = evaluate_classifier_group_kfold(name, model, X, y_class, groups, n_splits=5)
        all_results[name] = res
        oof_preds_dict[name] = oof_pred
        cms_dict[name] = res["out_of_fold_metrics"]["confusion_matrix"]

        print(f"  -> {name:<20}: Mean Acc = {res['cv_summary']['accuracy_mean']:.4f} "
              f"(+/- {res['cv_summary']['accuracy_std']:.4f}), "
              f"Macro F1 = {res['cv_summary']['macro_f1_mean']:.4f} "
              f"(+/- {res['cv_summary']['macro_f1_std']:.4f}), "
              f"Clogged F1 = {res['cv_summary']['clogged_f1_mean']:.4f}")

    # Save combined confusion matrix plot
    os.makedirs(visualizations_dir, exist_ok=True)
    cm_plot_path = os.path.join(visualizations_dir, "09_confusion_matrices.png")
    plot_combined_confusion_matrices(cms_dict, cm_plot_path)

    # Save JSON report
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, "classification_evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)
    print(f"Classification report saved successfully to: {report_path}")
    print("==================================================")

    return all_results


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csv_file = os.path.join(base_dir, "dataset", "HydroPulse_synthetic_dataset.csv")
    rep_dir = os.path.join(base_dir, "reports")
    vis_dir = os.path.join(base_dir, "visualizations")
    run_classification_pipeline(csv_file, rep_dir, vis_dir)
