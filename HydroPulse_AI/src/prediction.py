"""
HydroPulse AI - Phase 7: Real-Time Prediction Engine
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
from typing import Dict, Any, Union

from src.preprocessing import (
    validate_sensor_reading,
    FEATURE_COLUMNS,
    INT_TO_CONDITION,
)


class HydroPulsePredictor:
    """
    Production inference engine for HydroPulse AI.
    Loads trained models, executes pre-inference sanity checks,
    and generates dual predictions: Filter Condition and RUL (days).
    """

    def __init__(self, models_dir: str = None):
        if models_dir is None:
            models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
        self.models_dir = models_dir

        self.classifier_path = os.path.join(models_dir, "filter_condition_classifier.pkl")
        self.regressor_path = os.path.join(models_dir, "rul_regressor.pkl")
        self.pipeline_path = os.path.join(models_dir, "preprocessing_pipeline.pkl")
        self.metadata_path = os.path.join(models_dir, "model_metadata.json")

        self.classifier = None
        self.regressor = None
        self.preprocessor = None
        self.metadata = {}
        self._load_artifacts()

    def _load_artifacts(self):
        """Load serialized models, preprocessors, and metadata."""
        if not os.path.exists(self.classifier_path):
            raise FileNotFoundError(f"Classifier model not found at: {self.classifier_path}")
        if not os.path.exists(self.regressor_path):
            raise FileNotFoundError(f"Regressor model not found at: {self.regressor_path}")

        self.classifier = joblib.load(self.classifier_path)
        self.regressor = joblib.load(self.regressor_path)

        if os.path.exists(self.pipeline_path):
            self.preprocessor = joblib.load(self.pipeline_path)

        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

    def predict_single(self, reading: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and predict condition and RUL for a single sensor snapshot.
        """
        # Validate input
        is_valid, err_msg, sanitized = validate_sensor_reading(reading)
        if not is_valid:
            return {
                "success": False,
                "error": err_msg,
                "prediction": None,
            }

        # Build feature DataFrame enforcing exact ordering
        input_df = pd.DataFrame([sanitized])[FEATURE_COLUMNS]

        # Apply preprocessor if available
        if self.preprocessor is not None:
            processed_features = self.preprocessor.transform(input_df)
        else:
            processed_features = input_df

        # Predict Filter Condition
        class_idx = int(self.classifier.predict(processed_features)[0])
        condition = INT_TO_CONDITION.get(class_idx, "Unknown")

        # Class probabilities
        probabilities = {}
        if hasattr(self.classifier, "predict_proba"):
            probs = self.classifier.predict_proba(processed_features)[0]
            for idx, prob in enumerate(probs):
                name = INT_TO_CONDITION.get(idx, f"Class_{idx}")
                probabilities[name] = round(float(prob), 4)

        # Predict Remaining Useful Life
        predicted_rul = float(self.regressor.predict(processed_features)[0])
        # Post-processing physical clamp: RUL cannot realistically be negative
        clamped_rul = max(0.0, round(predicted_rul, 2))

        # Maintenance recommendation logic
        if condition == "Clogged" or clamped_rul <= 15.0:
            urgency = "CRITICAL: Immediate filter replacement required."
            recommended_action = "Halt automated dispense or switch to bypass; replace sediment/RO cartridge."
        elif condition == "Warning" or clamped_rul <= 45.0:
            urgency = "WARNING: Filter performance degrading."
            recommended_action = "Order replacement cartridge; schedule maintenance within 7-14 days."
        else:
            urgency = "NORMAL: Filter operating within healthy parameters."
            recommended_action = "Routine continuous sensor telemetry monitoring."

        return {
            "success": True,
            "error": None,
            "prediction": {
                "filter_condition": condition,
                "condition_code": class_idx,
                "class_probabilities": probabilities,
                "estimated_rul_days": clamped_rul,
                "maintenance_urgency": urgency,
                "recommended_action": recommended_action,
                "validated_inputs": sanitized,
            },
        }

    def predict_batch(self, readings: list) -> list:
        """Process batch of sensor readings."""
        return [self.predict_single(r) for r in readings]


def demo_predictions():
    """Run interactive demonstration on sample sensor telemetry."""
    print("==================================================")
    print("   HydroPulse AI - Phase 7: Prediction Engine Demo")
    print("==================================================")
    predictor = HydroPulsePredictor()

    sample_normal = {
        "TDS_ppm": 375.0,
        "Turbidity_NTU": 0.82,
        "Flow_Rate_L_min": 2.38,
        "Inlet_Pressure_bar": 2.05,
        "Outlet_Pressure_bar": 1.95,
        "Differential_Pressure_bar": 0.10,
        "Filter_Usage_days": 5.0,
    }

    sample_warning = {
        "TDS_ppm": 310.0,
        "Turbidity_NTU": 3.20,
        "Flow_Rate_L_min": 1.25,
        "Inlet_Pressure_bar": 2.08,
        "Outlet_Pressure_bar": 1.35,
        "Differential_Pressure_bar": 0.73,
        "Filter_Usage_days": 130.0,
    }

    sample_clogged = {
        "TDS_ppm": 220.0,
        "Turbidity_NTU": 6.35,
        "Flow_Rate_L_min": 0.65,
        "Inlet_Pressure_bar": 2.00,
        "Outlet_Pressure_bar": 0.65,
        "Differential_Pressure_bar": 1.35,
        "Filter_Usage_days": 210.0,
    }

    samples = [
        ("Normal Test Case", sample_normal),
        ("Warning Test Case", sample_warning),
        ("Clogged Test Case", sample_clogged),
    ]

    for label, payload in samples:
        print(f"\n--- Testing: {label} ---")
        result = predictor.predict_single(payload)
        if result["success"]:
            p = result["prediction"]
            print(f"  Condition: {p['filter_condition']} (Probabilities: {p['class_probabilities']})")
            print(f"  Estimated RUL: {p['estimated_rul_days']} days")
            print(f"  Urgency: {p['maintenance_urgency']}")
        else:
            print(f"  Error: {result['error']}")
    print("==================================================")


if __name__ == "__main__":
    demo_predictions()
