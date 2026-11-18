"""
HydroPulse AI - Phase 6: Hyperparameter Optimization and Model Selection
Author: HydroPulse ML Engineering Team
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import Dict, Any, Tuple
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from src.preprocessing import (
    separate_features_and_targets,
    build_preprocessing_pipeline,
    FEATURE_COLUMNS,
    CONDITION_TO_INT,
    INT_TO_CONDITION,
)
from src.classification import evaluate_classifier_group_kfold
from src.regression import evaluate_regressor_group_kfold


def get_default_dataset_path() -> str:
    """Resolve dataset path prioritizing the user's downloads folder directly."""
    direct_download_path = r"c:\Users\shyam\Downloads\Atomberg Filter\HydroPulse_synthetic_dataset.csv"
    if os.path.exists(direct_download_path):
        return direct_download_path
    local_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "HydroPulse_synthetic_dataset.csv")
    if os.path.exists(local_path):
        return os.path.abspath(local_path)
    return "HydroPulse_synthetic_dataset.csv"


def tune_classification_model(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series
) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """
    Tune Random Forest and Gradient Boosting classifiers using GroupKFold.
    Optimizes for Macro F1 to balance all three classes (especially Warning and Clogged).
    """
    print("  [Tuning Classification] Searching hyperparameter space with 5-fold GroupKFold...")
    gkf = GroupKFold(n_splits=5)
    cv_splits = list(gkf.split(X, y, groups=groups))

    # Grid for Random Forest
    rf_grid = {
        "n_estimators": [100, 150],
        "max_depth": [6, 10, None],
        "min_samples_split": [2, 5],
        "class_weight": ["balanced", None],
    }

    rf_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        rf_grid,
        cv=cv_splits,
        scoring="f1_macro",
        n_jobs=-1,
    )
    rf_search.fit(X, y)
    best_rf_params = rf_search.best_params_
    best_rf_score = rf_search.best_score_
    print(f"    -> Best RF Params: {best_rf_params} (CV Macro F1: {best_rf_score:.4f})")

    # Evaluate best RF with full group CV
    best_rf = RandomForestClassifier(**best_rf_params, random_state=42, n_jobs=-1)
    rf_eval, _ = evaluate_classifier_group_kfold("Optimized_Random_Forest", best_rf, X, y, groups, n_splits=5)

    # Fit final model on complete dataset
    best_rf.fit(X, y)

    return best_rf, best_rf_params, rf_eval


def tune_regression_model(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series
) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """
    Tune Random Forest and Gradient Boosting regressors using GroupKFold.
    Optimizes for Negative Mean Absolute Error (MAE).
    """
    print("  [Tuning Regression] Searching hyperparameter space with 5-fold GroupKFold...")
    gkf = GroupKFold(n_splits=5)
    cv_splits = list(gkf.split(X, y, groups=groups))

    # Grid for Gradient Boosting
    gb_grid = {
        "n_estimators": [100, 150],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5],
        "min_samples_split": [2, 5],
    }

    gb_search = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        gb_grid,
        cv=cv_splits,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    gb_search.fit(X, y)
    best_gb_params = gb_search.best_params_
    best_gb_score = -gb_search.best_score_
    print(f"    -> Best GB Regressor Params: {best_gb_params} (CV MAE: {best_gb_score:.3f} days)")

    # Evaluate best GB with full group CV
    best_gb = GradientBoostingRegressor(**best_gb_params, random_state=42)
    gb_eval, _ = evaluate_regressor_group_kfold("Optimized_Gradient_Boosting", best_gb, X, y, groups, n_splits=5)

    # Fit final model on complete dataset
    best_gb.fit(X, y)

    return best_gb, best_gb_params, gb_eval


def save_models_and_artifacts(
    best_clf: Any,
    best_clf_params: Dict[str, Any],
    clf_eval: Dict[str, Any],
    best_reg: Any,
    best_reg_params: Dict[str, Any],
    reg_eval: Dict[str, Any],
    models_dir: str,
    reports_dir: str,
    X: pd.DataFrame = None,
) -> None:
    """Save all trained models, preprocessing pipeline, metadata, and comparison reports."""
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Save Preprocessing Pipeline
    prep_pipeline = build_preprocessing_pipeline(scale_numeric=False)
    prep_pipeline.fit(X)
    prep_pipeline_path = os.path.join(models_dir, "preprocessing_pipeline.pkl")
    joblib.dump(prep_pipeline, prep_pipeline_path)
    print(f"  [Model Saved] Preprocessing pipeline -> {prep_pipeline_path}")

    # 2. Save Classification Model
    clf_path = os.path.join(models_dir, "filter_condition_classifier.pkl")
    joblib.dump(best_clf, clf_path)
    print(f"  [Model Saved] Classifier -> {clf_path}")

    # 3. Save Regression Model
    reg_path = os.path.join(models_dir, "rul_regressor.pkl")
    joblib.dump(best_reg, reg_path)
    print(f"  [Model Saved] Regressor -> {reg_path}")

    # 4. Save Model Metadata
    metadata = {
        "project_name": "HydroPulse AI - Intelligent Water Filter Monitoring",
        "version": "1.0.0",
        "feature_order": FEATURE_COLUMNS,
        "classification": {
            "model_type": type(best_clf).__name__,
            "target": "Filter_Condition",
            "classes": INT_TO_CONDITION,
            "best_hyperparameters": best_clf_params,
            "validation_metrics": clf_eval["cv_summary"],
            "out_of_fold_metrics": {
                "accuracy": clf_eval["out_of_fold_metrics"]["accuracy"],
                "macro_f1": clf_eval["out_of_fold_metrics"]["f1_macro"],
                "weighted_f1": clf_eval["out_of_fold_metrics"]["f1_weighted"],
                "per_class": clf_eval["out_of_fold_metrics"]["per_class"],
            },
        },
        "regression": {
            "model_type": type(best_reg).__name__,
            "target": "RUL_days",
            "best_hyperparameters": best_reg_params,
            "validation_metrics": reg_eval["cv_summary"],
            "out_of_fold_metrics": {
                "mae": reg_eval["out_of_fold_metrics"]["mae"],
                "rmse": reg_eval["out_of_fold_metrics"]["rmse"],
                "r2_score": reg_eval["out_of_fold_metrics"]["r2_score"],
                "median_absolute_error": reg_eval["out_of_fold_metrics"]["median_absolute_error"],
                "rul_range_error_analysis": reg_eval["out_of_fold_metrics"]["rul_range_error_analysis"],
            },
        },
    }

    metadata_path = os.path.join(models_dir, "model_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"  [Metadata Saved] -> {metadata_path}")

    # 5. Write Model Comparison Markdown Summary
    summary_md_path = os.path.join(reports_dir, "model_comparison_summary.md")
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("# HydroPulse AI - Model Optimization & Comparison Report\n\n")
        f.write("## 1. Selected Best Classification Model\n")
        f.write(f"- **Algorithm**: `{type(best_clf).__name__}`\n")
        f.write(f"- **Best Parameters**: `{best_clf_params}`\n")
        f.write(f"- **GroupKFold Mean Accuracy**: `{clf_eval['cv_summary']['accuracy_mean']:.4f} ± {clf_eval['cv_summary']['accuracy_std']:.4f}`\n")
        f.write(f"- **GroupKFold Macro F1**: `{clf_eval['cv_summary']['macro_f1_mean']:.4f} ± {clf_eval['cv_summary']['macro_f1_std']:.4f}`\n")
        f.write(f"- **Warning Class F1**: `{clf_eval['cv_summary']['warning_f1_mean']:.4f}`\n")
        f.write(f"- **Clogged Class F1**: `{clf_eval['cv_summary']['clogged_f1_mean']:.4f}`\n\n")

        f.write("## 2. Selected Best RUL Regression Model\n")
        f.write(f"- **Algorithm**: `{type(best_reg).__name__}`\n")
        f.write(f"- **Best Parameters**: `{best_reg_params}`\n")
        f.write(f"- **GroupKFold Mean MAE**: `{reg_eval['cv_summary']['mae_mean']:.3f} ± {reg_eval['cv_summary']['mae_std']:.3f} days`\n")
        f.write(f"- **GroupKFold Mean RMSE**: `{reg_eval['cv_summary']['rmse_mean']:.3f} ± {reg_eval['cv_summary']['rmse_std']:.3f} days`\n")
        f.write(f"- **GroupKFold Mean R²**: `{reg_eval['cv_summary']['r2_mean']:.4f} ± {reg_eval['cv_summary']['r2_std']:.4f}`\n")
        f.write(f"- **Median Absolute Error**: `{reg_eval['cv_summary']['medae_mean']:.3f} days`\n")
    print(f"  [Report Saved] -> {summary_md_path}")


def run_model_optimization(
    csv_path: str = None,
    models_dir: str = None,
    reports_dir: str = None,
) -> None:
    """Execute end-to-end model tuning, selection, and artifact serialization."""
    if csv_path is None:
        csv_path = get_default_dataset_path()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if models_dir is None:
        models_dir = os.path.join(base_dir, "models")
    if reports_dir is None:
        reports_dir = os.path.join(base_dir, "reports")

    print("==================================================")
    print("   HydroPulse AI - Phase 6: Model Optimization")
    print(f"   Dataset Source: {csv_path}")
    print("==================================================")

    df = pd.read_csv(csv_path)
    X, y_class, y_reg, groups = separate_features_and_targets(df)

    # 1. Classification Optimization
    best_clf, best_clf_params, clf_eval = tune_classification_model(X, y_class, groups)

    # 2. Regression Optimization
    best_reg, best_reg_params, reg_eval = tune_regression_model(X, y_reg, groups)

    # 3. Model Saving & Artifact Generation
    save_models_and_artifacts(
        best_clf,
        best_clf_params,
        clf_eval,
        best_reg,
        best_reg_params,
        reg_eval,
        models_dir,
        reports_dir,
        X=X,
    )
    print("==================================================")


if __name__ == "__main__":
    run_model_optimization()
