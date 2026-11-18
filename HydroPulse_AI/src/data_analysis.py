"""
HydroPulse AI - Phase 1: Dataset Inspection and Audit Module
Author: HydroPulse ML Engineering Team
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any


EXPECTED_COLUMNS = [
    "Filter_Cycle_ID",
    "Sample_Index",
    "TDS_ppm",
    "Turbidity_NTU",
    "Flow_Rate_L_min",
    "Inlet_Pressure_bar",
    "Outlet_Pressure_bar",
    "Differential_Pressure_bar",
    "Filter_Usage_days",
    "Processed_Water_Litres",
    "Filter_Condition",
    "RUL_days",
]

SELECTED_FEATURES = [
    "TDS_ppm",
    "Turbidity_NTU",
    "Flow_Rate_L_min",
    "Inlet_Pressure_bar",
    "Outlet_Pressure_bar",
    "Differential_Pressure_bar",
    "Filter_Usage_days",
]


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load dataset from CSV file and verify columns."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    # Verify column presence
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    return df


def audit_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform comprehensive dataset inspection and audit.
    Identifies dimensions, types, missing values, duplicates,
    filter cycles, class distributions, and feature statistics.
    """
    num_rows, num_cols = df.shape

    # Data types and missing values
    col_dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    missing_values = {col: int(df[col].isna().sum()) for col in df.columns}
    missing_pct = {col: float(df[col].isna().mean() * 100) for col in df.columns}

    # Duplicate check
    duplicate_count = int(df.duplicated().sum())

    # Filter cycles analysis
    unique_cycles = sorted(df["Filter_Cycle_ID"].dropna().unique().tolist())
    num_unique_cycles = len(unique_cycles)
    cycle_sample_counts = df["Filter_Cycle_ID"].value_counts().to_dict()
    samples_per_cycle = {
        "min": int(min(cycle_sample_counts.values())) if cycle_sample_counts else 0,
        "max": int(max(cycle_sample_counts.values())) if cycle_sample_counts else 0,
        "mean": float(np.mean(list(cycle_sample_counts.values()))) if cycle_sample_counts else 0.0,
    }

    # Classification Target: Filter_Condition
    class_counts = df["Filter_Condition"].value_counts().to_dict()
    class_pct = (df["Filter_Condition"].value_counts(normalize=True) * 100).round(2).to_dict()

    # Regression Target: RUL_days
    rul = df["RUL_days"]
    rul_stats = {
        "count": int(rul.count()),
        "mean": float(rul.mean()),
        "std": float(rul.std()),
        "min": float(rul.min()),
        "p25": float(rul.quantile(0.25)),
        "median": float(rul.median()),
        "p75": float(rul.quantile(0.75)),
        "max": float(rul.max()),
        "skewness": float(rul.skew()),
        "negative_values_count": int((rul < 0).sum()),
        "zero_values_count": int((rul == 0).sum()),
    }

    # Numerical statistics for selected features
    feature_stats = {}
    for col in SELECTED_FEATURES:
        series = df[col]
        feature_stats[col] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "p25": float(series.quantile(0.25)),
            "median": float(series.median()),
            "p75": float(series.quantile(0.75)),
            "max": float(series.max()),
            "skewness": float(series.skew()),
        }

    # Pressure consistency check: Differential = Inlet - Outlet
    pressure_diff = df["Inlet_Pressure_bar"] - df["Outlet_Pressure_bar"]
    pressure_discrepancy = (df["Differential_Pressure_bar"] - pressure_diff).abs()
    pressure_integrity = {
        "max_discrepancy_bar": float(pressure_discrepancy.max()),
        "mean_discrepancy_bar": float(pressure_discrepancy.mean()),
        "strictly_consistent_within_0_01bar": bool((pressure_discrepancy <= 0.015).all()),
    }

    # Feature correlations with RUL_days
    correlations_with_rul = {
        col: float(df[col].corr(df["RUL_days"])) for col in SELECTED_FEATURES
    }

    # Feature statistics grouped by Filter_Condition class
    condition_group_means = {}
    for col in SELECTED_FEATURES:
        condition_group_means[col] = df.groupby("Filter_Condition")[col].mean().to_dict()

    audit_summary = {
        "dimensions": {
            "num_rows": num_rows,
            "num_columns": num_cols,
        },
        "columns": list(df.columns),
        "column_dtypes": col_dtypes,
        "missing_values": missing_values,
        "missing_percentages": missing_pct,
        "duplicate_records": duplicate_count,
        "filter_cycles": {
            "num_unique_cycles": num_unique_cycles,
            "unique_cycles_list": unique_cycles,
            "samples_per_cycle": samples_per_cycle,
        },
        "classification_target_distribution": {
            "counts": class_counts,
            "percentages": class_pct,
        },
        "rul_target_distribution": rul_stats,
        "selected_feature_statistics": feature_stats,
        "pressure_integrity": pressure_integrity,
        "correlations_with_rul": correlations_with_rul,
        "feature_means_by_condition": condition_group_means,
    }

    return audit_summary


def save_audit_report(audit_summary: Dict[str, Any], output_path: str) -> None:
    """Save the audit report to a formatted JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=4)
    print(f"[Phase 1] Audit report successfully saved to {output_path}")


def get_default_dataset_path() -> str:
    """Resolve dataset path prioritizing the user's downloads folder directly."""
    direct_download_path = r"c:\Users\shyam\Downloads\Atomberg Filter\HydroPulse_synthetic_dataset.csv"
    if os.path.exists(direct_download_path):
        return direct_download_path
    local_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "HydroPulse_synthetic_dataset.csv")
    if os.path.exists(local_path):
        return os.path.abspath(local_path)
    return "HydroPulse_synthetic_dataset.csv"


def run_inspection(
    csv_path: str = None,
    report_path: str = "reports/dataset_audit_report.json",
) -> Dict[str, Any]:
    """Execute end-to-end dataset inspection and print key findings."""
    if csv_path is None:
        csv_path = get_default_dataset_path()
    print("==================================================")
    print("   HydroPulse AI - Phase 1: Dataset Audit")
    print(f"   Dataset Source: {csv_path}")
    print("==================================================")
    df = load_dataset(csv_path)
    summary = audit_dataset(df)
    save_audit_report(summary, report_path)

    print(f"Total Rows: {summary['dimensions']['num_rows']}")
    print(f"Total Columns: {summary['dimensions']['num_columns']}")
    print(f"Unique Filter Cycles: {summary['filter_cycles']['num_unique_cycles']}")
    print(f"Samples per Cycle: {summary['filter_cycles']['samples_per_cycle']}")
    print(f"Duplicates: {summary['duplicate_records']}")
    print(f"Missing Values: {sum(summary['missing_values'].values())}")
    print("\nFilter Condition Distribution:")
    for cond, cnt in summary["classification_target_distribution"]["counts"].items():
        pct = summary["classification_target_distribution"]["percentages"][cond]
        print(f"  - {cond:<10}: {cnt:>5} samples ({pct:>5.1f}%)")

    print("\nRUL Summary (Days):")
    for k, v in summary["rul_target_distribution"].items():
        print(f"  - {k:<25}: {v}")

    print("\nSelected Feature Correlations with RUL_days:")
    for feat, corr in sorted(summary["correlations_with_rul"].items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  - {feat:<28}: {corr:+.4f}")

    print("==================================================")
    return summary


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csv_file = os.path.join(base_dir, "dataset", "HydroPulse_synthetic_dataset.csv")
    report_file = os.path.join(base_dir, "reports", "dataset_audit_report.json")
    run_inspection(csv_file, report_file)
