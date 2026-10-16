# Customer Churn Prediction

A reproducible IBM Telco Customer Churn project with an interpretable Logistic Regression baseline, leakage-safe preprocessing, holdout evaluation, and a Streamlit analytics and prediction dashboard. The application does not include fabricated customer records or precomputed performance claims. Metrics appear only after training on the supplied public CSV.

## 1. Get the dataset

Obtain the **IBM Telco Customer Churn** CSV (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) from the public [Kaggle dataset page](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). The filename on download may differ; rename the CSV to the expected name and save it as:

```text
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Expected columns include `customerID` and `Churn` (`Yes`/`No`). Blank `TotalCharges` values are converted to missing numeric values and median-imputed inside the training pipeline. `customerID` is excluded from training. The CSV is ignored by git.

## 2. Install

From the project root on Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Python 3.10 or newer is recommended. If your environment cannot install packages from PyPI, use an environment with network access and run the same `pip install` command there.

## 3. Train and evaluate

```bash
python train_model.py
```

Optional paths and faster training without cross-validation:

```bash
python train_model.py --data /path/to/telco.csv --output models/churn_pipeline.joblib
python train_model.py --no-cv
```

The command prints holdout accuracy, precision, recall, F1, ROC-AUC when calculable, confusion matrix, class prevalence, and training-only stratified CV F1 where possible. It saves the fitted preprocessing-plus-classifier bundle to `models/churn_pipeline.joblib` and metrics to `models/churn_pipeline.metrics.json`. The test split is stratified and held out for evaluation; the saved pipeline is then refit on all labeled rows. Re-train when the dataset changes.

## 4. Test

```bash
pytest -q
```

Tests use only a small synthetic fixture to check code paths; those values are not real customer data and are not used as model performance claims.

## 5. Launch the dashboard

After adding the CSV and training:

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit. The responsive dashboard provides **Light**, **Dark**, and **System** themes in the sidebar's **Appearance** selector. The choice is stored in Streamlit session state and survives app reruns. Light and Dark themes style the page, sidebar, metrics, forms, tabs and all Plotly chart elements with explicit colors. System mode follows `prefers-color-scheme` for Streamlit-owned UI and updates when the browser/OS preference changes. **System mode limitation:** Streamlit does not expose the browser's color preference to Python, while Plotly figures are constructed server-side. Charts therefore keep the light Plotly palette in System mode for readable labels; choose explicit Dark to synchronize Plotly charts to a dark palette.

The **Overview** tab shows data-derived customer KPIs and six charts: churn share, contract, tenure ranges, monthly-charge distribution, internet service and payment method. **Model Performance** displays the actual holdout metrics, confusion matrix and saved ROC/precision–recall curves. Curve points are generated from holdout probabilities during training; an existing older model bundle without those points continues to show its stored metrics and prompts that curves need a new training run. **Predict Customer** groups feature inputs by customer/tenure, services, and contract/billing, with choices and numeric bounds taken from the loaded CSV. It shows the model probability and risk band, plus directional Logistic Regression terms when available. Follow-up ideas are suggestions for human review, not proven interventions. If the CSV or model is absent, the app gives the next step instead of inventing data/results.

## Model notes and limitations

- Logistic Regression with `class_weight="balanced"` is the baseline. Numeric features use median imputation and scaling; categorical fields use most-frequent imputation and one-hot encoding with unknown-category support.
- Preprocessing is fitted within the sklearn pipeline, including during cross-validation. The target is the actual `Churn` column; the customer identifier is removed.
- Accuracy can obscure missed churners when churn is less common. Consider precision, recall, F1, ROC-AUC, the confusion matrix, and threshold choice together. A probability score is a model estimate and is not a guarantee or certainty.
- The public sample is historical and may not represent current customers, policies, or deployment conditions. Validate calibration, fairness, drift, decision costs, and operational usefulness before production use. No causal inference is implied.
