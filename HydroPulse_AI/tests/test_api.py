"""
Integration Tests for HydroPulse AI Flask API
"""

import json
import pytest
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "healthy"
    assert "supported_features" in data


def test_metadata_endpoint(client):
    response = client.get("/metadata")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "feature_order" in data or "feature_order" in str(data)


def test_predict_endpoint_missing_features(client):
    payload = {
        "TDS_ppm": 350.0,
        "Turbidity_NTU": 1.2,
        # missing other 5 features
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = json.loads(response.data)
    assert data["success"] is False
    assert "Missing required sensor feature" in data["error"]


def test_predict_endpoint_inconsistent_pressure(client):
    payload = {
        "TDS_ppm": 350.0,
        "Turbidity_NTU": 1.2,
        "Flow_Rate_L_min": 2.0,
        "Inlet_Pressure_bar": 2.0,
        "Outlet_Pressure_bar": 1.5,
        "Differential_Pressure_bar": 1.1,  # Inconsistent: 2.0 - 1.5 = 0.5 != 1.1
        "Filter_Usage_days": 20.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = json.loads(response.data)
    assert data["success"] is False
    assert "Inconsistent pressure" in data["error"]
