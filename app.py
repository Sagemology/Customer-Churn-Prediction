"""Customer Churn Analytics dashboard built from the IBM Telco data."""
from pathlib import Path
from inspect import signature

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import DEFAULT_DATA_PATH, DataFileMissingError, load_telco_data
from src.model import risk_category, split_features_target
from src.theme import (PALETTES, THEME_OPTIONS, initialize_theme_state, plotly_mode,
                       style_plotly_figure, theme_css)

ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT / "models" / "churn_pipeline.joblib"
CHURN_COLORS = {
    "light": {"No": "#267389", "Yes": "#c95f4c"},
    "dark": {"No": "#72c5d2", "Yes": "#f19a83"},
}


def top_model_indicators(pipeline, customer: pd.DataFrame) -> list[str]:
    """Return leading positive Logistic Regression terms, if available."""
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["classifier"]
        if not hasattr(classifier, "coef_"):
            return []
        transformed = preprocessor.transform(customer)
        values = transformed.toarray() if hasattr(transformed, "toarray") else transformed
        contributions = np.asarray(values)[0] * classifier.coef_[0]
        names = preprocessor.get_feature_names_out()
        indicators = []
        for i in np.argsort(contributions)[::-1]:
            if contributions[i] <= 0 or len(indicators) >= 3:
                break
            name = str(names[i])
            for prefix in ("numeric__", "categorical__", "onehot__"):
                name = name.replace(prefix, "")
            indicators.append(f"{name} contributes toward a higher model score")
        return indicators
    except (KeyError, ValueError, AttributeError, IndexError):
        return []


def show_plotly_chart(fig):
    """Use the current Streamlit width API and retain compatibility with older versions."""
    if "width" in signature(st.plotly_chart).parameters:
        return st.plotly_chart(fig, width="stretch")
    return st.plotly_chart(fig, use_container_width=True)


st.set_page_config(page_title="Customer Churn Analytics", page_icon="◉", layout="wide")
initialize_theme_state(st.session_state)
with st.sidebar:
    st.subheader("Appearance")
    theme_selection = st.selectbox("Theme", THEME_OPTIONS, key="theme_mode")
    st.caption("System follows your browser/OS preference for Streamlit UI. Plotly uses its light palette in System mode because Streamlit does not expose that browser preference to Python.")
    st.divider()
    st.subheader("Data source")
    st.caption(f"`{DEFAULT_DATA_PATH.relative_to(ROOT)}`")
    st.caption("Customer counts and charts use this loaded dataset.")
# st.html injects the CSS into the app document instead of relying on Markdown's
# HTML renderer, so changing the selectbox reliably replaces the active tokens.
if hasattr(st, "html"):
    st.html(theme_css(theme_selection))
else:  # Streamlit versions before st.html
    st.markdown(theme_css(theme_selection), unsafe_allow_html=True)
plot_theme = plotly_mode(theme_selection)
chart_colors = CHURN_COLORS[plot_theme]
st.markdown('<div class="dash-title"><h1>Customer Churn Analytics</h1><p>Explore retention patterns and estimate customer churn risk.</p></div>', unsafe_allow_html=True)

try:
    data = load_telco_data(DEFAULT_DATA_PATH)
except DataFileMissingError as exc:
    st.info(str(exc))
    st.markdown("See [README.md](README.md) for the dataset setup steps.")
    st.stop()
except (ValueError, OSError) as exc:
    st.error(f"Unable to load dataset: {exc}")
    st.stop()

monthly = pd.to_numeric(data.get("MonthlyCharges", pd.Series(dtype=float)), errors="coerce")
tenure = pd.to_numeric(data.get("tenure", pd.Series(dtype=float)), errors="coerce")
churn_rate = float(data["Churn"].eq("Yes").mean())
kpis = st.columns(4)
kpis[0].metric("Total customers", f"{len(data):,}")
kpis[1].metric("Churn rate", f"{churn_rate:.1%}", help="Share of the loaded data labeled Churn=Yes.")
kpis[2].metric("Median monthly charges", f"${monthly.median():,.2f}" if monthly.notna().any() else "—")
kpis[3].metric("Median customer tenure", f"{tenure.median():.0f} months" if tenure.notna().any() else "—")

overview, performance, prediction = st.tabs(["Overview", "Model Performance", "Predict Customer"])

with overview:
    st.markdown('<div class="small-note">All counts, distributions and summaries below are calculated from the loaded IBM Telco CSV.</div>', unsafe_allow_html=True)
    row1a, row1b = st.columns(2, gap="medium")
    with row1a:
        counts = data["Churn"].value_counts().reindex(["No", "Yes"], fill_value=0).rename_axis("Status").reset_index(name="Customers")
        fig = px.pie(counts, names="Status", values="Customers", hole=.58, title="Churn vs. retained customers",
                     color="Status", color_discrete_map=chart_colors)
        fig.update_traces(textposition="inside", textinfo="percent",
                          insidetextfont=dict(color="#192329" if plot_theme == "dark" else "#ffffff"),
                          hovertemplate="%{label}: %{value:,} customers (%{percent})<extra></extra>")
        show_plotly_chart(style_plotly_figure(fig, theme_selection))
    with row1b:
        if "Contract" in data:
            summary = data.groupby(["Contract", "Churn"], observed=True).size().rename("Customers").reset_index()
            fig = px.bar(summary, x="Contract", y="Customers", color="Churn", barmode="group", title="Churn by contract type",
                         color_discrete_map=chart_colors, labels={"Contract":"Contract type", "Churn":"Status"})
            show_plotly_chart(style_plotly_figure(fig, theme_selection))
        else:
            st.info("Contract column is not present in the loaded CSV.")
    row2a, row2b = st.columns(2, gap="medium")
    with row2a:
        if "tenure" in data:
            tenure_group = pd.cut(tenure, bins=[-1, 12, 24, 48, np.inf], labels=["0–12 months", "13–24 months", "25–48 months", "49+ months"])
            summary = pd.DataFrame({"Tenure range":tenure_group, "Churn":data["Churn"]}).dropna().groupby(["Tenure range", "Churn"], observed=False).size().rename("Customers").reset_index()
            fig = px.bar(summary, x="Tenure range", y="Customers", color="Churn", barmode="group", title="Churn by tenure range",
                         color_discrete_map=chart_colors, labels={"Tenure range":"Customer tenure", "Churn":"Status"})
            show_plotly_chart(style_plotly_figure(fig, theme_selection))
        else:
            st.info("Tenure column is not present in the loaded CSV.")
    with row2b:
        if "MonthlyCharges" in data:
            fig = px.histogram(data, x="MonthlyCharges", color="Churn", barmode="overlay", opacity=.68, nbins=30,
                               marginal="box", title="Monthly charges by churn status", color_discrete_map=chart_colors,
                               labels={"MonthlyCharges":"Monthly charges ($)", "Churn":"Status"})
            show_plotly_chart(style_plotly_figure(fig, theme_selection))
        else:
            st.info("MonthlyCharges column is not present in the loaded CSV.")
    row3a, row3b = st.columns(2, gap="medium")
    for col, feature, title in [(row3a, "InternetService", "Churn by internet service"), (row3b, "PaymentMethod", "Churn by payment method")]:
        with col:
            if feature in data:
                summary = data.groupby([feature, "Churn"], observed=True).size().rename("Customers").reset_index()
                fig = px.bar(summary, x=feature, y="Customers", color="Churn", barmode="group", title=title,
                             color_discrete_map=chart_colors, labels={feature:feature, "Churn":"Status"})
                fig.update_layout(xaxis_tickangle=-10)
                show_plotly_chart(style_plotly_figure(fig, theme_selection))
            else:
                st.info(f"{feature} column is not present in the loaded CSV.")
    with st.expander("Dataset sample"):
        st.dataframe(data.head(15), use_container_width=True, hide_index=True)

with performance:
    if not ARTIFACT.is_file():
        st.warning("No fitted model artifact was found. Run `python train_model.py` from the project root.")
    else:
        try:
            bundle = joblib.load(ARTIFACT)
            metrics = bundle["metrics"]
            st.caption(f"Stratified holdout · {metrics['test_rows']:,} customers · positive class: Churn=Yes · {metrics['model']}")
            metric_columns = st.columns(5)
            for col, label, key in zip(metric_columns, ["Accuracy", "Precision", "Recall", "F1 score", "ROC-AUC"], ["accuracy", "precision", "recall", "f1", "roc_auc"]):
                value = metrics.get(key)
                col.metric(label, f"{value:.3f}" if value is not None else "N/A")
            if metrics.get("cv_f1_mean") is not None:
                st.caption(f"Training-only {metrics.get('cv_folds', 'CV')}-fold F1: {metrics['cv_f1_mean']:.3f} ± {metrics['cv_f1_std']:.3f} · overall churn prevalence: {metrics['positive_rate']:.1%}")
            c1, c2 = st.columns(2, gap="medium")
            with c1:
                cm = metrics["confusion_matrix"]
                palette = PALETTES[plot_theme]
                heatmap = go.Heatmap(z=cm, x=["Predicted: No", "Predicted: Yes"], y=["Actual: No", "Actual: Yes"],
                                     showscale=False, colorscale=[[0,palette["heat_low"]],[1,palette["heat_high"]]],
                                     hovertemplate="%{y}<br>%{x}<br>Customers: %{z}<extra></extra>")
                fig = go.Figure(heatmap)
                maximum = max(max(row) for row in cm) or 1
                for row_index, row in enumerate(cm):
                    for col_index, value in enumerate(row):
                        text_color = palette["heat_text"] if plot_theme == "dark" or value > maximum * .48 else palette["heat_text_dark"]
                        fig.add_annotation(x=col_index, y=row_index, text=str(value), showarrow=False,
                                           font=dict(color=text_color, size=15))
                fig.update_layout(title="Confusion matrix · holdout", xaxis_title="Prediction", yaxis_title="Actual", yaxis_autorange="reversed")
                show_plotly_chart(style_plotly_figure(fig, theme_selection))
            with c2:
                curves = metrics.get("curves")
                if curves and curves.get("roc_fpr") and curves.get("pr_recall"):
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=curves["roc_fpr"], y=curves["roc_tpr"], mode="lines", name="ROC", line=dict(color=chart_colors["No"], width=2.5), xaxis="x", yaxis="y"))
                    fig.add_trace(go.Scatter(x=curves["pr_recall"], y=curves["pr_precision"], mode="lines", name="Precision–recall", line=dict(color=chart_colors["Yes"], width=2.5), xaxis="x2", yaxis="y2"))
                    fig.update_layout(title="Holdout ROC and precision–recall", xaxis=dict(title="False positive rate", domain=[0,.46]), yaxis=dict(title="True positive rate"),
                                      xaxis2=dict(title="Recall", domain=[.56,1]), yaxis2=dict(title="Precision", anchor="x2"))
                    show_plotly_chart(style_plotly_figure(fig, theme_selection))
                else:
                    st.info("Curve points are not in this saved model bundle. The current training workflow saves them for a future model run; existing holdout metrics remain unchanged.")
            st.info("Accuracy can hide missed churners when classes are imbalanced. Review churn-class precision, recall, F1, ROC-AUC and the confusion matrix together. Metrics describe this holdout and do not guarantee future performance.")
        except (KeyError, OSError, ValueError, TypeError) as exc:
            st.error(f"Could not read the saved model bundle. Details: {exc}")

with prediction:
    if not ARTIFACT.is_file():
        st.info("Run `python train_model.py` to create the model bundle and enable predictions.")
    else:
        try:
            bundle = joblib.load(ARTIFACT)
            pipeline = bundle["pipeline"]
            X, _ = split_features_target(data)
            st.write("Enter a customer's details. Category options and numeric bounds come from the loaded dataset.")
            feature_groups = {"Customer & tenure":[], "Services":[], "Contract & billing":[]}
            for feature in X.columns:
                lower = feature.lower()
                group = "Customer & tenure" if lower in {"tenure", "seniorcitizen", "partner", "dependents"} else (
                    "Contract & billing" if any(token in lower for token in ("contract", "payment", "charge", "paperless", "monthly", "total")) else "Services")
                feature_groups[group].append(feature)
            form_values = {}
            with st.form("predict_customer"):
                form_cols = st.columns(3, gap="medium")
                for i, (group, features) in enumerate(feature_groups.items()):
                    with form_cols[i]:
                        st.markdown(f"**{group}**")
                        for feature in features:
                            values = X[feature]
                            if pd.api.types.is_numeric_dtype(values):
                                median = float(values.median()) if values.notna().any() else 0.
                                minimum = float(values.min()) if values.notna().any() else median
                                maximum = float(values.max()) if values.notna().any() else median
                                step = 1. if pd.api.types.is_integer_dtype(values) else max((maximum-minimum)/100, .01)
                                form_values[feature] = st.number_input(feature, min_value=minimum, max_value=maximum,
                                    value=min(max(median, minimum), maximum), step=step)
                            else:
                                options = sorted(values.dropna().astype(str).unique().tolist())
                                if options:
                                    form_values[feature] = st.selectbox(feature, options)
                submitted = st.form_submit_button("Estimate churn risk", type="primary")
            if submitted:
                customer = pd.DataFrame([form_values], columns=X.columns)
                probability = float(pipeline.predict_proba(customer)[0, 1])
                label = int(pipeline.predict(customer)[0])
                st.subheader("Prediction estimate")
                result1, result2 = st.columns(2)
                result1.metric("Estimated churn probability", f"{probability:.1%}")
                result2.metric("Risk category", risk_category(probability))
                st.progress(probability, text=f"Predicted class: {'Churn' if label else 'No churn'}")
                indicators = top_model_indicators(pipeline, customer)
                if indicators:
                    st.markdown("**Relevant model indicators**")
                    st.caption("Directional Logistic Regression terms for this customer, not causal explanations.")
                    for indicator in indicators:
                        st.markdown(f"- {indicator}")
                    st.markdown("**Possible follow-up**")
                    st.caption("Use these indicators to guide a human review of customer needs. Any retention offer is a suggestion to evaluate, not a proven intervention.")
                st.caption("Probability is a model estimate from historical data, not certainty or a guarantee of future behavior.")
        except (KeyError, OSError, ValueError, TypeError, AttributeError) as exc:
            st.error(f"Prediction is unavailable: {exc}")
