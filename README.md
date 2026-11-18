# HydroPulse AI – Intelligent Water Filter Monitoring & Predictive Maintenance

**Python 3.10+ | Scikit-Learn | Flask | Pytest | MIT License**

## 1. Project Overview

HydroPulse AI is a machine learning-based platform that monitors water filter conditions and predicts maintenance requirements.

It performs two tasks:
- **Filter Classification:** Identifies filter health as Normal, Warning, or Clogged.
- **Remaining Useful Life (RUL):** Estimates the number of days remaining before filter replacement is required.

It supports sediment, activated carbon, and RO filters.

## 2. Problem Statement

Traditional filter replacement relies on fixed schedules, causing:
- Premature replacement and unnecessary costs.
- Delayed replacement and reduced filtration performance.
- Increased maintenance and equipment damage.

**Solution:** HydroPulse AI uses sensor data and machine learning to enable condition-based maintenance.

## 3. Dataset

- Dataset: `HydroPulse_synthetic_dataset.csv`
- Total records: 3,000
- Filter cycles: 30
- Samples per cycle: 100
- Missing values: 0
- Duplicate records: 0

**Input Features:**
- TDS
- Turbidity
- Flow Rate
- Inlet Pressure
- Outlet Pressure
- Differential Pressure
- Filter Usage Days

## 4. Technology Stack

- Python
- Pandas and NumPy
- Scikit-Learn
- Flask REST API
- Matplotlib and Seaborn
- Pytest

## 5. Machine Learning Models

**Classification:** Optimized Random Forest
- Accuracy: 97.10%
- Macro F1-Score: 0.9618

**RUL Prediction:** Optimized Gradient Boosting
- MAE: 13.13 days
- RMSE: 17.82 days
- R² Score: 0.9114

Models are evaluated using 5-fold GroupKFold cross-validation to prevent filter-cycle data leakage.

## 6. System Architecture

```text
HydroPulse_AI/
├── dataset/
├── notebooks/
├── src/
│   ├── data_analysis.py
│   ├── preprocessing.py
│   ├── visualization.py
│   ├── classification.py
│   ├── regression.py
│   ├── model_evaluation.py
│   ├── model_optimization.py
│   └── prediction.py
├── models/
├── reports/
├── visualizations/
├── tests/
├── app.py
├── requirements.txt
└── README.md
```

## 7. API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check API status |
| GET | `/metadata` | View model information |
| POST | `/predict` | Predict filter condition and RUL |

The API validates sensor readings and rejects inconsistent pressure values.

## 8. Installation and Execution

```bash
# Clone or navigate to the project
cd HydroPulse_AI

# Create virtual environment
python -m venv .venv

# Activate environment (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start API
python app.py
```

Run tests:

```bash
pytest tests/
```

## 9. Limitations

The current models are trained and evaluated using synthetic data. Real-world performance must be verified using physical filtration systems and actual sensor readings before deployment.

## 10. Future Enhancements

- ESP32-based TinyML deployment.
- Real-time IoT sensor monitoring.
- MQTT/LoRaWAN integration.
- Temperature compensation.
- Sensor fault and anomaly detection.

## 11. License

Released under the MIT License.
