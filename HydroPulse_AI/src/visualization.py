"""
HydroPulse AI - Phase 2: Exploratory Data Analysis and Visualization Module
Author: HydroPulse ML Engineering Team
"""

import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional


# Professional aesthetics
PALETTE = {"Normal": "#2ca02c", "Warning": "#ff7f0e", "Clogged": "#d62728"}
FEATURE_COLUMNS = [
    "TDS_ppm",
    "Turbidity_NTU",
    "Flow_Rate_L_min",
    "Inlet_Pressure_bar",
    "Outlet_Pressure_bar",
    "Differential_Pressure_bar",
    "Filter_Usage_days",
]


def setup_plot_style():
    """Apply consistent styling across all figures."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 300,
    })


def plot_filter_condition_distribution(df: pd.DataFrame, output_dir: str):
    """Plot 1: Filter Condition Distribution (Bar & Donut Chart)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    counts = df["Filter_Condition"].value_counts()
    colors = [PALETTE[c] for c in counts.index]

    # Bar plot
    bars = axes[0].bar(counts.index, counts.values, color=colors, edgecolor="black", alpha=0.85)
    axes[0].set_title("Filter Condition Frequency Count", fontweight="bold")
    axes[0].set_xlabel("Filter State")
    axes[0].set_ylabel("Number of Samples")
    for bar in bars:
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2.0, h + 25, f"{h} ({h/len(df)*100:.1f}%)",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Donut chart
    axes[1].pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
        textprops={"fontsize": 11, "fontweight": "bold"},
    )
    axes[1].set_title("Class Imbalance Overview", fontweight="bold")

    plt.tight_layout()
    path = os.path.join(output_dir, "01_filter_condition_distribution.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  [Saved] {path}")


def plot_rul_distribution(df: pd.DataFrame, output_dir: str):
    """Plot 2: RUL Distribution (Histogram + Boxplot)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(df["RUL_days"], kde=True, ax=axes[0], color="#1f77b4", bins=30, edgecolor="black")
    axes[0].axvline(df["RUL_days"].mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {df['RUL_days'].mean():.1f} d")
    axes[0].axvline(df["RUL_days"].median(), color="green", linestyle="-.", linewidth=2, label=f"Median: {df['RUL_days'].median():.1f} d")
    axes[0].set_title("Remaining Useful Life (RUL) Histogram & KDE", fontweight="bold")
    axes[0].set_xlabel("RUL (Days)")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    sns.boxplot(x="Filter_Condition", y="RUL_days", data=df, ax=axes[1], palette=PALETTE, order=["Normal", "Warning", "Clogged"])
    axes[1].set_title("RUL Distribution Grouped by Filter Condition", fontweight="bold")
    axes[1].set_xlabel("Filter Condition")
    axes[1].set_ylabel("RUL (Days)")

    plt.tight_layout()
    path = os.path.join(output_dir, "02_rul_distribution.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  [Saved] {path}")


def plot_feature_correlation_heatmap(df: pd.DataFrame, output_dir: str):
    """Plot 3: Feature Correlation Heatmap."""
    cols_to_corr = FEATURE_COLUMNS + ["RUL_days"]
    corr = df[cols_to_corr].corr()

    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.8,
        cbar_kws={"label": "Pearson Correlation Coefficient"},
    )
    plt.title("Sensor Features and Target Correlation Matrix", fontweight="bold", pad=15)
    plt.tight_layout()
    path = os.path.join(output_dir, "03_feature_correlation_heatmap.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  [Saved] {path}")


def plot_differential_pressure_vs_usage(df: pd.DataFrame, output_dir: str):
    """Plot 4: Differential Pressure vs Filter Usage Days."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x="Filter_Usage_days",
        y="Differential_Pressure_bar",
        hue="Filter_Condition",
        palette=PALETTE,
        data=df,
        alpha=0.75,
        s=40,
    )
    plt.axhline(0.6, color="orange", linestyle="--", label="Warning Threshold (~0.6 bar)")
    plt.axhline(1.0, color="red", linestyle="--", label="Clogged Threshold (~1.0 bar)")
    plt.title("Differential Pressure (\u0394P) vs Filter Usage Across Lifecycle", fontweight="bold")
    plt.xlabel("Filter Usage (Days)")
    plt.ylabel("Differential Pressure \u0394P (bar)")
    plt.legend(loc="upper left")
    plt.tight_layout()
    path = os.path.join(output_dir, "04_differential_pressure_vs_usage.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  [Saved] {path}")


def plot_flow_rate_vs_usage(df: pd.DataFrame, output_dir: str):
    """Plot 5: Flow Rate vs Filter Usage Days."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x="Filter_Usage_days",
        y="Flow_Rate_L_min",
        hue="Filter_Condition",
        palette=PALETTE,
        data=df,
        alpha=0.75,
        s=40,
    )
    plt.title("Permeate Flow Rate Decay vs Filter Usage", fontweight="bold")
    plt.xlabel("Filter Usage (Days)")
    plt.ylabel("Flow Rate (L/min)")
    plt.legend(loc="lower left")
    plt.tight_layout()
    path = os.path.join(output_dir, "05_flow_rate_vs_usage.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  [Saved] {path}")


def plot_rul_vs_usage(df: pd.DataFrame, output_dir: str):
    """Plot 6: RUL vs Filter Usage Days."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x="Filter_Usage_days",
        y="RUL_days",
        hue="Filter_Condition",
        palette=PALETTE,
        data=df,
        alpha=0.75,
        s=40,
    )
    plt.title("Remaining Useful Life (RUL) Trajectory vs Filter Service Days", fontweight="bold")
    plt.xlabel("Filter Usage (Days)")
    plt.ylabel("RUL (Days)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    path = os.path.join(output_dir, "06_rul_vs_usage.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  [Saved] {path}")


def plot_feature_distributions(df: pd.DataFrame, output_dir: str):
    """Plot 7: Distribution histograms for all 7 input features."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for idx, col in enumerate(FEATURE_COLUMNS):
        sns.histplot(df[col], kde=True, ax=axes[idx], color="#2b5c8f", bins=25, edgecolor="black")
        axes[idx].set_title(f"Distribution: {col}", fontweight="bold", fontsize=11)
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel("Count")

    # Hide extra subplots
    axes[7].axis("off")
    axes[8].axis("off")

    plt.tight_layout()
    path = os.path.join(output_dir, "07_feature_distributions.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  [Saved] {path}")


def plot_condition_vs_sensor_readings(df: pd.DataFrame, output_dir: str):
    """Plot 8: Filter Condition vs Important Sensor Readings (Violin & Boxplots)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    order = ["Normal", "Warning", "Clogged"]

    sns.boxplot(x="Filter_Condition", y="Differential_Pressure_bar", data=df, ax=axes[0, 0], palette=PALETTE, order=order)
    axes[0, 0].set_title("Differential Pressure by Condition", fontweight="bold")
    axes[0, 0].set_ylabel("\u0394P (bar)")

    sns.boxplot(x="Filter_Condition", y="Flow_Rate_L_min", data=df, ax=axes[0, 1], palette=PALETTE, order=order)
    axes[0, 1].set_title("Flow Rate by Condition", fontweight="bold")
    axes[0, 1].set_ylabel("Flow Rate (L/min)")

    sns.boxplot(x="Filter_Condition", y="Filter_Usage_days", data=df, ax=axes[1, 0], palette=PALETTE, order=order)
    axes[1, 0].set_title("Usage Days by Condition", fontweight="bold")
    axes[1, 0].set_ylabel("Usage Days")

    sns.boxplot(x="Filter_Condition", y="Turbidity_NTU", data=df, ax=axes[1, 1], palette=PALETTE, order=order)
    axes[1, 1].set_title("Turbidity by Condition", fontweight="bold")
    axes[1, 1].set_ylabel("Turbidity (NTU)")

    plt.tight_layout()
    path = os.path.join(output_dir, "08_condition_vs_sensor_readings.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  [Saved] {path}")


def generate_all_eda_visualizations(
    csv_path: str = "dataset/HydroPulse_synthetic_dataset.csv",
    output_dir: str = "visualizations",
):
    """Generate and save complete suite of EDA visual figures."""
    print("==================================================")
    print("   HydroPulse AI - Phase 2: Generating EDA Plots")
    print("==================================================")
    setup_plot_style()
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    plot_filter_condition_distribution(df, output_dir)
    plot_rul_distribution(df, output_dir)
    plot_feature_correlation_heatmap(df, output_dir)
    plot_differential_pressure_vs_usage(df, output_dir)
    plot_flow_rate_vs_usage(df, output_dir)
    plot_rul_vs_usage(df, output_dir)
    plot_feature_distributions(df, output_dir)
    plot_condition_vs_sensor_readings(df, output_dir)
    print(f"All 8 EDA visualizations saved successfully to: {output_dir}")
    print("==================================================")


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csv_file = os.path.join(base_dir, "dataset", "HydroPulse_synthetic_dataset.csv")
    vis_dir = os.path.join(base_dir, "visualizations")
    generate_all_eda_visualizations(csv_file, vis_dir)
