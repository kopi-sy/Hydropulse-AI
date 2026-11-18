"""
HydroPulse AI - Phase 2: Data Preprocessing Module
Author: HydroPulse ML Engineering Team
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "TDS_ppm",
    "Turbidity_NTU",
    "Flow_Rate_L_min",
    "Inlet_Pressure_bar",
    "Outlet_Pressure_bar",
    "Differential_Pressure_bar",
    "Filter_Usage_days",
]

CLASSIFICATION_TARGET = "Filter_Condition"
REGRESSION_TARGET = "RUL_days"
GROUP_COLUMN = "Filter_Cycle_ID"

# Canonical label mapping
CONDITION_TO_INT = {"Normal": 0, "Warning": 1, "Clogged": 2}
INT_TO_CONDITION = {0: "Normal", 1: "Warning", 2: "Clogged"}

# Sensor physical boundaries for integrity checks
SENSOR_BOUNDS = {
    "TDS_ppm": (0.0, 2000.0),
    "Turbidity_NTU": (0.0, 100.0),
    "Flow_Rate_L_min": (0.0, 20.0),
    "Inlet_Pressure_bar": (0.0, 10.0),
    "Outlet_Pressure_bar": (0.0, 10.0),
    "Differential_Pressure_bar": (0.0, 10.0),
    "Filter_Usage_days": (0.0, 1000.0),
}


class SensorFeaturePreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compliant transformer that validates feature order,
    checks numerical boundaries, and optionally scales inputs.
    Ensures identical feature ordering across training, validation, and inference.
    """

    def __init__(self, feature_columns: list = None, scale_numeric: bool = False):
        self.feature_columns = feature_columns or FEATURE_COLUMNS
        self.scale_numeric = scale_numeric
        self.scaler = StandardScaler() if scale_numeric else None

    def fit(self, X, y=None):
        X_df = self._ensure_dataframe(X)
        if self.scale_numeric:
            self.scaler.fit(X_df[self.feature_columns])
        self.fitted_ = True
        return self

    def transform(self, X):
        X_df = self._ensure_dataframe(X)
        X_subset = X_df[self.feature_columns].copy()

        if self.scale_numeric:
            scaled_array = self.scaler.transform(X_subset)
            return pd.DataFrame(scaled_array, columns=self.feature_columns, index=X_df.index)
        return X_subset

    def _ensure_dataframe(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        elif isinstance(X, dict):
            return pd.DataFrame([X])
        elif isinstance(X, (list, np.ndarray)):
            return pd.DataFrame(X, columns=self.feature_columns)
        else:
            raise TypeError(f"Unsupported data type for X: {type(X)}")


def validate_raw_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate dataset integrity: missing values, duplicates, and physical limits.
    Does not drop rows or mutate records.
    """
    missing_count = int(df.isna().sum().sum())
    duplicate_count = int(df.duplicated().sum())

    # Check dtypes
    non_numeric_features = [
        col for col in FEATURE_COLUMNS if not np.issubdtype(df[col].dtype, np.number)
    ]

    # Outlier detection using IQR (for reporting purposes only, no silent row removal)
    outlier_summary = {}
    for col in FEATURE_COLUMNS:
        q25 = df[col].quantile(0.25)
        q75 = df[col].quantile(0.75)
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        num_outliers = int(((df[col] < lower_bound) | (df[col] > upper_bound)).sum())
        outlier_summary[col] = {
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "outlier_count": num_outliers,
            "outlier_percentage": float((num_outliers / len(df)) * 100),
        }

    return {
        "is_valid": (missing_count == 0 and duplicate_count == 0 and len(non_numeric_features) == 0),
        "total_missing_values": missing_count,
        "duplicate_records": duplicate_count,
        "non_numeric_features": non_numeric_features,
        "outlier_analysis": outlier_summary,
    }


def separate_features_and_targets(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Separate dataset into:
    - X (Feature DataFrame, 7 sensor features)
    - y_class (Classification target encoded as integers 0, 1, 2)
    - y_reg (Regression target RUL in days)
    - groups (Filter_Cycle_ID for GroupKFold)
    """
    X = df[FEATURE_COLUMNS].copy()
    y_class = df[CLASSIFICATION_TARGET].map(CONDITION_TO_INT)
    y_reg = df[REGRESSION_TARGET].copy()
    groups = df[GROUP_COLUMN].copy()

    return X, y_class, y_reg, groups


def validate_sensor_reading(reading: Dict[str, Any]) -> Tuple[bool, str, Dict[str, float]]:
    """
    Validate a single input payload before inference.
    Checks:
    1. Presence of required sensor keys.
    2. Missing / NaN / None values.
    3. Pressure consistency between Inlet, Outlet, and Differential pressure.
    4. Sensor physical ranges.
    Returns: (is_valid, error_message, sanitized_dict)
    """
    sanitized = {}

    # Check presence and finite values
    for col in FEATURE_COLUMNS:
        if col not in reading or reading[col] is None:
            return False, f"Missing required sensor feature: '{col}'", {}
        try:
            val = float(reading[col])
            if np.isnan(val) or np.isinf(val):
                return False, f"Invalid non-finite value for '{col}': {reading[col]}", {}
            sanitized[col] = val
        except (ValueError, TypeError):
            return False, f"Value for '{col}' must be a valid numeric float, received: {reading[col]}", {}

    # Physical range check
    for col, (min_v, max_v) in SENSOR_BOUNDS.items():
        val = sanitized[col]
        if val < min_v or val > max_v:
            return False, f"Sensor '{col}' value {val} is outside physical bounds [{min_v}, {max_v}]", {}

    # Pressure consistency check
    p_in = sanitized["Inlet_Pressure_bar"]
    p_out = sanitized["Outlet_Pressure_bar"]
    p_diff = sanitized["Differential_Pressure_bar"]

    if p_out > p_in:
        return False, (
            f"Physical violation: Outlet pressure ({p_out:.3f} bar) cannot exceed inlet pressure ({p_in:.3f} bar)."
        ), {}

    computed_diff = p_in - p_out
    discrepancy = abs(p_diff - computed_diff)
    # Allow small tolerance of 0.05 bar for sensor precision / noise
    if discrepancy > 0.05:
        return False, (
            f"Inconsistent pressure readings: Differential_Pressure_bar ({p_diff:.3f} bar) deviates by {discrepancy:.3f} bar "
            f"from Inlet ({p_in:.3f}) - Outlet ({p_out:.3f}) = {computed_diff:.3f} bar (tolerance: 0.05 bar)."
        ), {}

    return True, "", sanitized


def build_preprocessing_pipeline(scale_numeric: bool = False) -> Pipeline:
    """Construct an sklearn Pipeline for clean preprocessing."""
    pipeline = Pipeline(
        steps=[
            ("sensor_preprocessor", SensorFeaturePreprocessor(FEATURE_COLUMNS, scale_numeric=scale_numeric))
        ]
    )
    return pipeline
