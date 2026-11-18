# HydroPulse AI - Model Optimization & Comparison Report

## 1. Selected Best Classification Model
- **Algorithm**: `RandomForestClassifier`
- **Best Parameters**: `{'class_weight': None, 'max_depth': 10, 'min_samples_split': 5, 'n_estimators': 100}`
- **GroupKFold Mean Accuracy**: `0.9713 ± 0.0007`
- **GroupKFold Macro F1**: `0.9618 ± 0.0021`
- **Warning Class F1**: `0.9497`
- **Clogged Class F1**: `0.9498`

## 2. Selected Best RUL Regression Model
- **Algorithm**: `GradientBoostingRegressor`
- **Best Parameters**: `{'learning_rate': 0.05, 'max_depth': 5, 'min_samples_split': 2, 'n_estimators': 100}`
- **GroupKFold Mean MAE**: `13.131 ± 2.025 days`
- **GroupKFold Mean RMSE**: `18.294 ± 2.572 days`
- **GroupKFold Mean R²**: `0.9054 ± 0.0256`
- **Median Absolute Error**: `8.181 days`
