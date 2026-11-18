"""
Unit Tests for HydroPulse AI Preprocessing and Validation
"""

import pytest
import numpy as np
import pandas as pd

from src.preprocessing import (
    validate_raw_dataset,
    validate_sensor_reading,
    separate_features_and_targets,
    FEATURE_COLUMNS,
    SENSOR_BOUNDS,
)


def test_validate_sensor_reading_valid():
    sample = {
        "TDS_ppm": 350.0,
        "Turbidity_NTU": 1.5,
        "Flow_Rate_L_min": 2.1,
        "Inlet_Pressure_bar": 2.05,
        "Outlet_Pressure_bar": 1.90,
        "Differential_Pressure_bar": 0.15,
        "Filter_Usage_days": 10.0,
    }
    is_valid, err, sanitized = validate_sensor_reading(sample)
    assert is_valid is True
    assert err == ""
    assert sanitized["TDS_ppm"] == 350.0


def test_validate_sensor_reading_missing_field():
    sample = {
        "TDS_ppm": 350.0,
        "Turbidity_NTU": 1.5,
        # Flow_Rate_L_min missing
        "Inlet_Pressure_bar": 2.05,
        "Outlet_Pressure_bar": 1.90,
        "Differential_Pressure_bar": 0.15,
        "Filter_Usage_days": 10.0,
    }
    is_valid, err, sanitized = validate_sensor_reading(sample)
    assert is_valid is False
    assert "Missing required sensor feature" in err


def test_validate_sensor_reading_inconsistent_pressure():
    sample = {
        "TDS_ppm": 350.0,
        "Turbidity_NTU": 1.5,
        "Flow_Rate_L_min": 2.1,
        "Inlet_Pressure_bar": 2.00,
        "Outlet_Pressure_bar": 1.50,
        "Differential_Pressure_bar": 1.20,  # Inlet - Outlet = 0.50, deviation 0.70 bar > 0.05 bar
        "Filter_Usage_days": 10.0,
    }
    is_valid, err, sanitized = validate_sensor_reading(sample)
    assert is_valid is False
    assert "Inconsistent pressure readings" in err


def test_validate_sensor_reading_outlet_exceeds_inlet():
    sample = {
        "TDS_ppm": 350.0,
        "Turbidity_NTU": 1.5,
        "Flow_Rate_L_min": 2.1,
        "Inlet_Pressure_bar": 1.80,
        "Outlet_Pressure_bar": 2.20,  # Physical impossibility in passive filter
        "Differential_Pressure_bar": 0.00,
        "Filter_Usage_days": 10.0,
    }
    is_valid, err, sanitized = validate_sensor_reading(sample)
    assert is_valid is False
    assert "Outlet pressure" in err and "cannot exceed inlet" in err


def test_validate_sensor_reading_out_of_bounds():
    sample = {
        "TDS_ppm": 99999.0,  # Out of range
        "Turbidity_NTU": 1.5,
        "Flow_Rate_L_min": 2.1,
        "Inlet_Pressure_bar": 2.00,
        "Outlet_Pressure_bar": 1.80,
        "Differential_Pressure_bar": 0.20,
        "Filter_Usage_days": 10.0,
    }
    is_valid, err, sanitized = validate_sensor_reading(sample)
    assert is_valid is False
    assert "outside physical bounds" in err
