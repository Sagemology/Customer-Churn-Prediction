# Customer Churn Analytics — Project Summary

## Project overview

Customer Churn Analytics is a Python machine-learning project that explores customer churn in the public IBM Telco Customer Churn dataset. It pairs a Streamlit dashboard with a reproducible scikit-learn classification pipeline and an interactive form for estimating churn risk from customer service and billing details.

## What it includes

- **Data preparation:** loads the real IBM Telco CSV, converts `TotalCharges` to numeric, handles missing values in the training pipeline, and excludes `customerID` from model inputs.
- **Model:** a class-weighted Logistic Regression baseline with numeric median imputation and scaling, categorical most-frequent imputation and one-hot encoding.
- **Evaluation:** stratified train/test split, holdout classification metrics, confusion matrix, and training-only stratified cross-validation.
- **Dashboard:** dataset-based customer KPIs and interactive churn, contract, tenure, monthly-charge, internet-service, and payment-method charts; model metrics; and customer-level probability estimates with risk bands.
- **Interface:** responsive Light, Dark, and System appearance options. The project documents that Plotly charts use a light palette in System mode because the server cannot read the browser's color preference.
- **Testing:** pytest coverage for data loading, preprocessing, training, prediction, and theme behavior. Synthetic records are only test fixtures, not reported business data.

## Holdout results

These figures are from the model metrics report already present in the project. They are not hardcoded dashboard values.

| Metric | Result |
|---|---:|
| Holdout rows | 1,409 |
| Accuracy | 0.738 |
| Precision (Churn=Yes) | 0.504 |
| Recall (Churn=Yes) | 0.783 |
| F1 score (Churn=Yes) | 0.614 |
| ROC-AUC | 0.841 |
| 5-fold training CV F1 | 0.628 ± 0.023 |

Confusion matrix at the model's 0.50 classification threshold (actual rows: No, Yes; predicted columns: No, Yes):

```text
[[747, 288],
 [ 81, 293]]
```

The dataset's observed churn prevalence is 26.5%. Accuracy should not be read alone: churn-class precision and recall show the trade-off between false alarms and missed churners.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# Place the IBM CSV at data/WA_Fn-UseC_-Telco-Customer-Churn.csv if retraining.
python train_model.py
pytest -q
streamlit run app.py
```

## Limitations

The model is trained on a historical public sample and may not represent a current customer base. A predicted probability is an estimate, not certainty or a causal explanation. Validate calibration, fairness, data drift, and decision costs before using predictions to guide real customer actions. Any suggested follow-up is for human review, not a proven retention intervention.
