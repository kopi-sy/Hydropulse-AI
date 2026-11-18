"""
HydroPulse AI - Phase 4: Remaining Useful Life (RUL) Regression Models Module
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
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from src.preprocessing import (
    separate_features_and_targets,
    FEATURE_COLUMNS,
)
from src.model_evaluation import (
    evaluate_regression_predictions,
    plot_rul_error_distributions,
)


def get_default_dataset_path() -> str:
    """Resolve dataset path prioritizing the user's downloads folder directly."""
    direct_download_path = r"c:\Users\shyam\Downloads\Atomberg Filter\HydroPulse_synthetic_dataset.csv"
    if os.path.exists(direct_download_path):
        return direct_download_path
    local_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "HydroPulse_synthetic_dataset.csv")
    if os.path.exists(local_path):
        return os.path.abspath(local_path)
    return "HydroPulse_synthetic_dataset.csv"


def get_baseline_regressors(random_state: int = 42) -> Dict[str, Any]:
    """Define baseline regression algorithms."""
    return {
        "Decision_Tree": DecisionTreeRegressor(random_state=random_state),
        "Random_Forest": RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1),
        "Gradient_Boosting": GradientBoostingRegressor(n_estimators=100, random_state=random_state),
    }


def evaluate_regressor_group_kfold(
    model_name: str,
    model_instance: Any,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 5,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Perform GroupKFold cross-validation on an RUL regressor.
    Ensures zero filter cycle overlap between train and test sets.
    Computes fold-level metrics and out-of-fold overall metrics.
    """
    gkf = GroupKFold(n_splits=n_splits)

    fold_metrics = []
    oof_predictions = np.zeros(len(y), dtype=float)

    print(f"  Evaluating {model_name} across {n_splits} GroupKFold splits...")

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        test_cycles = groups.iloc[test_idx].unique().tolist()

        # Fit model
        model = type(model_instance)(**model_instance.get_params())
        model.fit(X_train, y_train)

        # Predict test fold
        y_pred = model.predict(X_test)
        oof_predictions[test_idx] = y_pred

        # Fold evaluation
        fold_eval = evaluate_regression_predictions(y_test.values, y_pred)
        fold_metrics.append({
            "fold": fold_idx + 1,
            "test_cycles": test_cycles,
            "mae": fold_eval["mae"],
            "rmse": fold_eval["rmse"],
            "r2_score": fold_eval["r2_score"],
            "median_absolute_error": fold_eval["median_absolute_error"],
            "mean_bias": fold_eval["mean_bias"],
            "negative_predictions": fold_eval["negative_predictions_count"],
        })

    # Summary of metrics across folds (Mean & Standard Deviation)
    mae_scores = [f["mae"] for f in fold_metrics]
    rmse_scores = [f["rmse"] for f in fold_metrics]
    r2_scores = [f["r2_score"] for f in fold_metrics]
    medae_scores = [f["median_absolute_error"] for f in fold_metrics]
    bias_scores = [f["mean_bias"] for f in fold_metrics]

    # Global out-of-fold evaluation
    oof_eval = evaluate_regression_predictions(y.values, oof_predictions)

    results = {
        "model_name": model_name,
        "n_splits": n_splits,
        "cv_summary": {
            "mae_mean": float(np.mean(mae_scores)),
            "mae_std": float(np.std(mae_scores)),
            "rmse_mean": float(np.mean(rmse_scores)),
            "rmse_std": float(np.std(rmse_scores)),
            "r2_mean": float(np.mean(r2_scores)),
            "r2_std": float(np.std(r2_scores)),
            "medae_mean": float(np.mean(medae_scores)),
            "medae_std": float(np.std(medae_scores)),
            "mean_bias": float(np.mean(bias_scores)),
        },
        "fold_details": fold_metrics,
        "out_of_fold_metrics": oof_eval,
    }

    return results, oof_predictions


def run_regression_pipeline(
    csv_path: str = None,
    reports_dir: str = None,
    visualizations_dir: str = None,
) -> Dict[str, Any]:
    """Execute regression pipeline across Decision Tree, Random Forest, and Gradient Boosting."""
    if csv_path is None:
        csv_path = get_default_dataset_path()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if reports_dir is None:
        reports_dir = os.path.join(base_dir, "reports")
    if visualizations_dir is None:
        visualizations_dir = os.path.join(base_dir, "visualizations")

    print("==================================================")
    print("   HydroPulse AI - Phase 4: RUL Regression Models")
    print(f"   Dataset Source: {csv_path}")
    print("==================================================")

    df = pd.read_csv(csv_path)
    X, _, y_reg, groups = separate_features_and_targets(df)

    models = get_baseline_regressors()
    all_results = {}
    model_predictions = {}

    for name, model in models.items():
        res, oof_pred = evaluate_regressor_group_kfold(name, model, X, y_reg, groups, n_splits=5)
        all_results[name] = res
        model_predictions[name] = {
            "y_true": y_reg.values,
            "y_pred": oof_pred,
        }

        print(f"  -> {name:<20}: MAE = {res['cv_summary']['mae_mean']:.3f} (+/- {res['cv_summary']['mae_std']:.3f}) d, "
              f"RMSE = {res['cv_summary']['rmse_mean']:.3f} (+/- {res['cv_summary']['rmse_std']:.3f}) d, "
              f"R² = {res['cv_summary']['r2_mean']:.4f}, "
              f"MedAE = {res['cv_summary']['medae_mean']:.3f} d")

    # Save RUL error analysis plots
    os.makedirs(visualizations_dir, exist_ok=True)
    error_plot_path = os.path.join(visualizations_dir, "10_rul_error_analysis.png")
    plot_rul_error_distributions(model_predictions, error_plot_path)

    # Save JSON report
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, "regression_evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)
    print(f"Regression report saved successfully to: {report_path}")
    print("==================================================")

    return all_results


if __name__ == "__main__":
    run_regression_pipeline()
