"""
HydroPulse AI - Model Evaluation Utilities
Author: HydroPulse ML Engineering Team
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
)

CLASS_NAMES = ["Normal", "Warning", "Clogged"]


def evaluate_classification_predictions(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, Any]:
    """Calculate comprehensive classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Per-class metrics
    prec_per_class = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
    rec_per_class = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
    report_dict = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )

    per_class_dict = {}
    for idx, name in enumerate(CLASS_NAMES):
        per_class_dict[name] = {
            "precision": float(prec_per_class[idx]),
            "recall": float(rec_per_class[idx]),
            "f1_score": float(f1_per_class[idx]),
        }

    return {
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "precision_weighted": float(prec_weighted),
        "recall_macro": float(rec_macro),
        "recall_weighted": float(rec_weighted),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "per_class": per_class_dict,
        "confusion_matrix": cm,
        "classification_report": report_dict,
    }


def evaluate_regression_predictions(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, Any]:
    """Calculate comprehensive regression metrics and error segment analysis."""
    mae = mean_absolute_error(y_true, y_pred)
    # RMSE calculation compatible across scikit-learn versions
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)

    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    # Error analysis across RUL ranges
    # 1. Critical Phase: RUL <= 30 days
    # 2. Warning Phase: 30 < RUL <= 90 days
    # 3. Healthy Phase: RUL > 90 days
    mask_critical = y_true <= 30.0
    mask_warning = (y_true > 30.0) & (y_true <= 90.0)
    mask_healthy = y_true > 90.0

    range_analysis = {}
    for name, mask in [
        ("Critical_0_to_30_days", mask_critical),
        ("Warning_30_to_90_days", mask_warning),
        ("Healthy_above_90_days", mask_healthy),
    ]:
        if np.sum(mask) > 0:
            range_analysis[name] = {
                "count": int(np.sum(mask)),
                "mae": float(np.mean(abs_errors[mask])),
                "rmse": float(np.sqrt(np.mean(errors[mask] ** 2))),
                "mean_bias": float(np.mean(errors[mask])),  # positive = overestimation, negative = underestimation
            }

    negative_predictions = int(np.sum(y_pred < 0))
    min_pred = float(np.min(y_pred))
    max_pred = float(np.max(y_pred))

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2_score": float(r2),
        "median_absolute_error": float(medae),
        "mean_bias": float(np.mean(errors)),
        "negative_predictions_count": negative_predictions,
        "min_prediction": min_pred,
        "max_prediction": max_pred,
        "rul_range_error_analysis": range_analysis,
    }


def plot_combined_confusion_matrices(
    model_cms: Dict[str, List[List[int]]], output_path: str
):
    """Plot confusion matrices side-by-side for all evaluated classification models."""
    n_models = len(model_cms)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for idx, (model_name, cm_list) in enumerate(model_cms.items()):
        cm = np.array(cm_list)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[idx],
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            cbar=False,
        )
        axes[idx].set_title(f"Confusion Matrix: {model_name}", fontweight="bold")
        axes[idx].set_xlabel("Predicted Label")
        axes[idx].set_ylabel("True Label")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  [Saved Confusion Matrices] {output_path}")


def plot_rul_error_distributions(
    model_predictions: Dict[str, Dict[str, np.ndarray]], output_path: str
):
    """Plot actual vs predicted RUL scatter and error distribution for regression models."""
    n_models = len(model_predictions)
    fig, axes = plt.subplots(2, n_models, figsize=(6 * n_models, 10))
    if n_models == 1:
        axes = np.expand_dims(axes, axis=1)

    for idx, (model_name, preds) in enumerate(model_predictions.items()):
        y_true = preds["y_true"]
        y_pred = preds["y_pred"]
        errors = y_pred - y_true

        # Row 0: Actual vs Predicted
        axes[0, idx].scatter(y_true, y_pred, alpha=0.35, color="#1f77b4", s=15)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[0, idx].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Ideal (y = x)")
        axes[0, idx].set_title(f"{model_name}: Predicted vs Actual", fontweight="bold")
        axes[0, idx].set_xlabel("Actual RUL (days)")
        axes[0, idx].set_ylabel("Predicted RUL (days)")
        axes[0, idx].legend()

        # Row 1: Error distribution
        sns.histplot(errors, kde=True, ax=axes[1, idx], color="#e377c2", bins=30, edgecolor="black")
        axes[1, idx].axvline(0, color="black", linestyle="--", linewidth=1.5)
        axes[1, idx].set_title(f"{model_name}: Residual Error (Pred - True)", fontweight="bold")
        axes[1, idx].set_xlabel("Residual Error (days)")
        axes[1, idx].set_ylabel("Count")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  [Saved RUL Error Analysis] {output_path}")
