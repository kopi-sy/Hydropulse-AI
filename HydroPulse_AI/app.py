"""
HydroPulse AI - Phase 8: Production Flask REST API
Author: HydroPulse ML Engineering Team
"""

import os
import sys
from flask import Flask, request, jsonify

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.prediction import HydroPulsePredictor
from src.preprocessing import FEATURE_COLUMNS, SENSOR_BOUNDS

app = Flask(__name__)

# Initialize predictor singleton on startup
predictor = None
try:
    predictor = HydroPulsePredictor()
    print("[Flask Startup] HydroPulse ML models and pipelines successfully loaded.")
except Exception as e:
    print(f"[Flask Startup Warning] Models could not be loaded immediately: {e}")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for IoT gateways and ESP32 connectivity."""
    return jsonify({
        "status": "healthy",
        "service": "HydroPulse AI Inference Server",
        "models_loaded": predictor is not None,
        "supported_features": FEATURE_COLUMNS,
    }), 200


@app.route("/metadata", methods=["GET"])
def metadata():
    """Expose feature specification and model evaluation metadata."""
    if predictor and predictor.metadata:
        return jsonify(predictor.metadata), 200
    return jsonify({
        "feature_order": FEATURE_COLUMNS,
        "sensor_bounds": SENSOR_BOUNDS,
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Primary inference endpoint for ESP32 and Edge Water Filter Telemetry.
    Expects JSON payload with 7 sensor features:
      - TDS_ppm
      - Turbidity_NTU
      - Flow_Rate_L_min
      - Inlet_Pressure_bar
      - Outlet_Pressure_bar
      - Differential_Pressure_bar
      - Filter_Usage_days
    """
    if predictor is None:
        return jsonify({
            "success": False,
            "error": "Prediction engine is not initialized. Ensure models are trained.",
        }), 503

    if not request.is_json:
        return jsonify({
            "success": False,
            "error": "Request Content-Type must be 'application/json'.",
        }), 400

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({
            "success": False,
            "error": "Malformed JSON body.",
        }), 400

    # Auto-calculate Differential Pressure if omitted but Inlet & Outlet are provided
    if "Differential_Pressure_bar" not in payload and "Inlet_Pressure_bar" in payload and "Outlet_Pressure_bar" in payload:
        try:
            p_in = float(payload["Inlet_Pressure_bar"])
            p_out = float(payload["Outlet_Pressure_bar"])
            payload["Differential_Pressure_bar"] = round(p_in - p_out, 3)
        except (ValueError, TypeError):
            pass

    # Execute prediction
    result = predictor.predict_single(payload)

    if not result["success"]:
        return jsonify(result), 422

    return jsonify(result), 200


if __name__ == "__main__":
    # Deployment configuration: debug disabled, binds to 0.0.0.0 for LAN/ESP32 access
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
