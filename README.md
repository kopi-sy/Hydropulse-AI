HydroPulse AI – Intelligent Water Filter Monitoring and Predictive Maintenance System
Python 3.10+ License: MIT Framework: Scikit-Learn API: Flask Tests: Pytest Passed

1. Project Overview
HydroPulse AI is an industrial-grade machine learning and predictive maintenance platform engineered to monitor water purification filters (sediment, activated carbon, and reverse osmosis membranes). By ingesting continuous sensor telemetry, the system performs two simultaneous inference tasks:

Filter Condition Classification (Multi-Class): Classifies filter operational health into Normal, Warning, or Clogged.
Remaining Useful Life (RUL) Regression: Predicts the exact number of days remaining (RUL_days) before the filtration cartridge reaches critical hydraulic exhaustion or clogging.
The system is built from scratch, strictly verified using group-based cross-validation across distinct filter lifecycle runs, packaged into a modular Python library, and exposed through a lightweight REST API tailored for edge IoT microcontrollers (e.g., ESP32, STM32, Raspberry Pi Pico).

2. Problem Statement & Industrial Objectives
2.1 The Problem
In residential and commercial water purifiers, filter replacements typically rely on fixed calendar intervals (e.g., "replace every 6 months") or crude volumetric meters. These static heuristics lead to:

Premature Replacement: Discarding filters with 30–50% remaining life, causing unnecessary costs and plastic/membrane waste.
Delayed Replacement: Allowing clogged filters to operate, leading to microbiological breakthrough, membrane fouling, severe pump strain, and water contamination.
2.2 HydroPulse AI Objectives
Dynamic Physics-Informed ML: Predict degradation based on hydrodynamic head loss and flow dynamics (governed by Darcy's Law for porous media).
Proactive Maintenance Scheduling: Alert users during the Warning phase (typically 15–45 days of RUL remaining) before catastrophic blockage occurs.
Zero-Leakage Group Validation: Prevent data leakage by ensuring observations from the same operational filter cycle never leak between training and testing sets.
Edge Deployment Ready: Deliver sub-millisecond inference via a production Flask API.
3. Dataset Description & Statistical Characteristics
The system was developed and validated using the supplied synthetic telemetry dataset (HydroPulse_synthetic_dataset.csv).

Metric / Dimension	Dataset Value	Notes
Total Observations	3,000 records	100 observations per filter run
Total Features & Targets	12 columns	7 features, 2 targets, 3 meta-columns
Unique Filter Cycles	30 (FC001 – FC030)	Full continuous lifecycles
Samples per Cycle	Exactly 100 samples/cycle	Uniform sampling interval
Missing Values	0 (0.0%)	Clean sensor telemetry
Duplicate Records	0 (0.0%)	Verified unique rows
Pressure Integrity	$	\Delta P - (P_{in} - P_{out})
3.1 Target Distributions & Class Imbalance
A. Classification Target: Filter_Condition
Normal: 1,777 samples (59.23%) — Clean membrane operating under low differential pressure.
Warning: 853 samples (28.43%) — Early cake formation; hydraulic resistance rising.
Clogged: 370 samples (12.33%) — Severe pore occlusion; critically low flow rate and high 
Δ
P
.
Class Imbalance Note: The dataset exhibits natural degradation imbalance (filters spend more time healthy than failing). Consequently, model evaluation prioritizes Macro F1-Score and class-specific metrics for Warning and Clogged rather than raw accuracy alone.

B. Regression Target: RUL_days
Range: 0.00 to 238.85 days
Mean ± Std: 
103.59
±
60.42
 days
Median (IQR): 102.25 days (
52.36
 to 
151.37
 days)
Zero RUL Records: 49 samples (end of filter life)
Negative Values: 0
4. Feature Selection & Physical Rationale
Seven physical sensor features are selected as the input vector 
x
∈
R
7
:

x = [ TDS_ppm, Turbidity_NTU, Flow_Rate_L_min, Inlet_Pressure_bar, Outlet_Pressure_bar, Differential_Pressure_bar, Filter_Usage_days ]
Feature Name	Units	Correlation with RUL	Physical & Chemical Importance
Differential_Pressure_bar	bar	-0.9490	Primary Indicator (Darcy's Law): Measures net pressure loss (
Δ
P
=
P
i
n
−
P
o
u
t
). As microscopic pores clog with foulants, hydraulic resistance increases dramatically, driving 
Δ
P
 upward.
Flow_Rate_L_min	L/min	+0.9253	Direct indicator of membrane permeability. Clean filters deliver 
∼
1.83
 L/min; clogged filters decay below 
0.80
 L/min under constant supply pressure.
Filter_Usage_days	days	-0.9001	Tracks cumulative service age, mechanical fatigue, and continuous exposure duration.
Outlet_Pressure_bar	bar	+0.8103	Measures clean water discharge pressure. Decreases as head loss through the filter medium increases.
Turbidity_NTU	NTU	-0.1717	Measures suspended particulate matter and micro-silt. Elevated turbidity accelerates particulate cake filtration.
TDS_ppm	ppm	+0.0753	Total Dissolved Solids. Influences osmotic pressure resistance across RO membranes.
Inlet_Pressure_bar	bar	+0.0294	Supply line pressure from booster pump/municipal feed (
∼
2.06
 bar baseline). Essential for validating 
Δ
P
.
Excluded Columns:
Filter_Cycle_ID: Retained strictly as the cluster grouping key for GroupKFold cross-validation.
Sample_Index: Sequential sample index within a run; excluded to prevent models from memorizing temporal sequences.
Processed_Water_Litres: Collinear volume integration of usage days; excluded to prevent future leakage.
5. System Architecture
HydroPulse_AI/
├── dataset/
│   └── HydroPulse_synthetic_dataset.csv     # Input sensor telemetry
├── notebooks/
│   └── exploratory_data_analysis.ipynb      # Interactive EDA walkthrough
├── src/
│   ├── __init__.py                          # Package init
│   ├── data_analysis.py                     # Phase 1: Dataset audit & statistics
│   ├── preprocessing.py                     # Phase 2: Sanitization & pipeline
│   ├── visualization.py                     # Phase 2: High-resolution figures
│   ├── classification.py                    # Phase 3: Classification models
│   ├── regression.py                        # Phase 4: RUL regression models
│   ├── model_evaluation.py                  # Evaluation metrics & reporting
│   ├── model_optimization.py                # Phase 6: Grid search & tuning
│   └── prediction.py                        # Phase 7: Production inference engine
├── models/
│   ├── filter_condition_classifier.pkl      # Champion classifier (Random Forest)
│   ├── rul_regressor.pkl                    # Champion regressor (Gradient Boosting)
│   ├── preprocessing_pipeline.pkl           # Scikit-learn feature pipeline
│   └── model_metadata.json                  # Model hyperparams & metrics schema
├── reports/
│   ├── dataset_audit_report.json            # Full statistical dataset audit
│   ├── classification_evaluation_report.json# Cross-validation classification scores
│   ├── regression_evaluation_report.json    # Cross-validation regression scores
│   └── model_comparison_summary.md          # Model tuning & selection summary
├── visualizations/
│   ├── 01_filter_condition_distribution.png
│   ├── 02_rul_distribution.png
│   ├── 03_feature_correlation_heatmap.png
│   ├── 04_differential_pressure_vs_usage.png
│   ├── 05_flow_rate_vs_usage.png
│   ├── 06_rul_vs_usage.png
│   ├── 07_feature_distributions.png
│   ├── 08_condition_vs_sensor_readings.png
│   ├── 09_confusion_matrices.png
│   └── 10_rul_error_analysis.png
├── tests/
│   ├── conftest.py
│   ├── test_preprocessing.py                # Unit tests for preprocessing
│   └── test_api.py                          # Integration tests for Flask endpoints
├── app.py                                   # Production Flask REST API
├── requirements.txt                         # Dependency specifications
└── README.md                                # Comprehensive documentation
6. Preprocessing Methodology
Zero Data Leakage Pipeline: All feature transformations are encapsulated within a custom Scikit-learn SensorFeaturePreprocessor pipeline.
Physical Boundary Validation: Sensor inputs are validated against strict physical operating ranges:
P
i
n
,
P
o
u
t
,
Δ
P
∈
[
0.0
,
10.0
]
 bar
P
o
u
t
≤
P
i
n
 (passive filtration cannot generate pressure)
Flow Rate 
∈
[
0.0
,
20.0
]
 L/min, TDS 
∈
[
0
,
2000
]
 ppm, Turbidity 
∈
[
0
,
100
]
 NTU
Hydrodynamic Consistency Check: Verifies that 
Δ
P
≈
P
i
n
−
P
o
u
t
 within an instrumentation noise tolerance of 
±
0.05
 bar. Inconsistent readings trigger immediate rejection (HTTP 422).
No Silent Row Removal: Outlier analysis is conducted via Interquartile Range (IQR) for observability, but no real physical operating extremes are discarded.
7. Validation Methodology: GroupKFold Cross-Validation
Standard random 
K
-fold cross-validation or train/test splits cause severe data leakage in time-series telemetry because sequential rows from the same filter cartridge appear in both train and test splits, causing models to merely interpolate timestamps.

HydroPulse AI enforces GroupKFold (
K
=
5
) grouped by Filter_Cycle_ID:

In every fold, 6 entire filter runs (600 observations) are held out as unseen test data.
The remaining 24 filter runs (2,400 observations) are used for model training.
Zero cycle overlap: 
C
y
c
l
e
s
t
r
a
i
n
∩
C
y
c
l
e
s
t
e
s
t
=
∅
.
Every filter cycle spans the complete lifecycle (Normal 
→
 Warning 
→
 Clogged), ensuring all three classes are represented in every validation test fold.
All baseline and optimized algorithms are evaluated against the exact same cross-validation splits.
8. Model Evaluation & Benchmark Results
8.1 Filter Condition Classification Comparison
All models evaluated via 5-Fold GroupKFold out-of-fold cross-validation:

Classifier	Mean Accuracy	Macro F1-Score	Weighted F1	Warning F1	Clogged F1	CV Variation (
σ
)
Decision Tree	94.77%	0.9333	0.9479	0.9103	0.9227	
±
0.0253
Gradient Boosting	96.37%	0.9506	0.9632	0.9348	0.9332	
±
0.0072
Random Forest (Baseline)	96.80%	0.9574	0.9678	0.9442	0.9445	
±
0.0012
Random Forest (Optimized) ★	97.10%	0.9618	0.9708	0.9491	0.9502	
±
0.0010
★ Champion Model: Optimized Random Forest (n_estimators=100, max_depth=10, min_samples_split=5).

Delivers the lowest variance across unseen filter cartridges (
σ
=
0.0010
).
Exceptional Clogged class F1 (
0.9502
) and Warning class F1 (
0.9491
), ensuring filters in critical danger are detected reliably.
8.2 Remaining Useful Life (RUL) Regression Comparison
Evaluated via 5-Fold GroupKFold out-of-fold cross-validation:

Regressor	MAE (days)	RMSE (days)	
R
2
 Score	Median AE (days)	Mean Bias (days)
Decision Tree	15.79 ± 1.87	22.40 ± 2.40	0.8599	9.99	+0.02
Random Forest	13.16 ± 1.75	18.67 ± 2.08	0.9023	7.82	+0.01
Gradient Boosting (Baseline)	14.13 ± 1.24	18.51 ± 1.40	0.9047	10.20	+0.04
Gradient Boosting (Optimized) ★	13.13 ± 1.15	17.82 ± 1.28	0.9114	8.42	+0.01
★ Champion Model: Optimized Gradient Boosting Regressor (learning_rate=0.05, max_depth=5, n_estimators=100).

Achieved highest overall 
R
2
 (
0.9114
) with the lowest cross-cycle variation (
σ
=
1.15
 days).
Critical Phase Accuracy (
R
U
L
≤
30
 days): Mean Absolute Error drops to 
∼
4.2
 days, providing high precision when filters are close to failure.
Zero negative RUL predictions generated across the entire 3,000-sample test set.
9. Flask REST API Specification
9.1 Base URL & Endpoints
Base: http://localhost:5000
GET /health — Health check & sensor specification.
GET /metadata — Model metadata & validation performance.
POST /predict — Real-time filter condition and RUL inference.
9.2 POST /predict Request & Response Examples
Example 1: Healthy Filter (Normal)
Request:

curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "TDS_ppm": 375.0,
    "Turbidity_NTU": 0.82,
    "Flow_Rate_L_min": 2.38,
    "Inlet_Pressure_bar": 2.05,
    "Outlet_Pressure_bar": 1.95,
    "Differential_Pressure_bar": 0.10,
    "Filter_Usage_days": 5.0
  }'
Response (HTTP 200 OK):

{
  "success": true,
  "error": null,
  "prediction": {
    "filter_condition": "Normal",
    "condition_code": 0,
    "class_probabilities": {
      "Normal": 1.0,
      "Warning": 0.0,
      "Clogged": 0.0
    },
    "estimated_rul_days": 211.2,
    "maintenance_urgency": "NORMAL: Filter operating within healthy parameters.",
    "recommended_action": "Routine continuous sensor telemetry monitoring."
  }
}
Example 2: Degrading Filter (Warning)
Request:

curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "TDS_ppm": 310.0,
    "Turbidity_NTU": 3.20,
    "Flow_Rate_L_min": 1.25,
    "Inlet_Pressure_bar": 2.08,
    "Outlet_Pressure_bar": 1.35,
    "Differential_Pressure_bar": 0.73,
    "Filter_Usage_days": 130.0
  }'
Response (HTTP 200 OK):

{
  "success": true,
  "error": null,
  "prediction": {
    "filter_condition": "Warning",
    "condition_code": 1,
    "class_probabilities": {
      "Normal": 0.0253,
      "Warning": 0.9735,
      "Clogged": 0.0012
    },
    "estimated_rul_days": 77.79,
    "maintenance_urgency": "WARNING: Filter performance degrading.",
    "recommended_action": "Order replacement cartridge; schedule maintenance within 7-14 days."
  }
}
Example 3: Inconsistent Sensor Pressure Rejection
Request:

curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "TDS_ppm": 350.0,
    "Turbidity_NTU": 1.5,
    "Flow_Rate_L_min": 2.0,
    "Inlet_Pressure_bar": 2.00,
    "Outlet_Pressure_bar": 1.50,
    "Differential_Pressure_bar": 1.20,
    "Filter_Usage_days": 10.0
  }'
Response (HTTP 422 Unprocessable Entity):

{
  "success": false,
  "error": "Inconsistent pressure readings: Differential_Pressure_bar (1.200 bar) deviates by 0.700 bar from Inlet (2.000) - Outlet (1.500) = 0.500 bar (tolerance: 0.05 bar).",
  "prediction": null
}
10. Installation & Execution Guide
10.1 Environment Setup
# 1. Navigate to project directory
cd HydroPulse_AI

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
10.2 Reproduce Pipeline Phases
# Phase 1: Run Dataset Audit & Integrity Check
python src/data_analysis.py

# Phase 2: Generate All 8 Exploratory Visualizations
python src/visualization.py

# Phase 3: Train and Evaluate Classification Models (GroupKFold)
python src/classification.py

# Phase 4: Train and Evaluate RUL Regression Models (GroupKFold)
python src/regression.py

# Phase 6: Hyperparameter Optimization and Model Serialization
python src/model_optimization.py

# Phase 7: Run Standalone Prediction Engine Demo
python src/prediction.py

# Run Test Suite
pytest tests/
10.3 Run Production Flask Server
# Start Flask API
python app.py
11. Important Scientific Limitations
[!WARNING] Synthetic Dataset Notice The current models were developed, trained, and evaluated on a synthetic telemetry dataset.

Synthetic datasets often exhibit smoother degradation curves, deterministic noise patterns, and uniform cycle lengths (
N
=
100
) that may not capture real-world operational anomalies.
High validation performance on synthetic data does not constitute real-world validation.
Real municipal water systems exhibit complex phenomena such as water hammer, air bubbles in pressure transducers, sensor calibration drift, biofilm biofouling surges, and seasonal temperature-induced viscosity shifts.
Pre-deployment requirement: Before flashing models to production ESP32 edge filters, a formal calibration study must be conducted against physical laboratory sensor benches using real water filtration cartridges.
12. Future Enhancements
Edge TinyML Compilation: Export the Scikit-learn Random Forest and Gradient Boosting decision trees to C++ header files using micromlgen or m2cgen for direct flash execution on ESP32 microcontrollers without an external Python server.
Dynamic Temperature Compensation: Integrate water temperature telemetry (
∘
C
) to account for viscosity changes that alter Darcy flow calculations.
Anomaly & Sensor Fault Detection: Add an unsupervised Autoencoder to detect detached pressure hoses, dry-run conditions (pump running without water), or broken turbine flowmeters.
MQTT / LoRaWAN Telemetry Stream: Provide native MQTT endpoints to stream readings directly from home purifiers to cloud analytics dashboards.
13. License
This project is licensed under the MIT License. Developed for intelligent water purification and predictive maintenance engineering.
