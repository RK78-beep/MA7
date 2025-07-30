import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import os
from utils.helpers import parse_pdf, parse_excel, recommend_deal, generate_pdf_report, gpt_commentary

st.set_page_config(page_title="M&A Deal Analyzer+", layout="wide")
st.title("🤝 M&A Deal Analyzer + AI Commentary")

uploaded_file = st.file_uploader("Upload Financial File (CSV, Excel, PDF)", type=["csv", "xlsx", "pdf"])

if uploaded_file:
    filename = uploaded_file.name
    if filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif filename.endswith(".xlsx"):
        df = parse_excel(uploaded_file)
    elif filename.endswith(".pdf"):
        df = parse_pdf(uploaded_file)
    else:
        st.error("Unsupported file format.")
        st.stop()

    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    model = joblib.load("model.pkl")
    X = df.select_dtypes(include=['number']).fillna(0)
    preds = model.predict(X)
    df['Prediction'] = preds

    # SHAP
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    st.subheader("🔎 SHAP Feature Importance")
    fig, ax = plt.subplots()
    shap.plots.bar(shap_values, max_display=5, show=False)
    st.pyplot(fig)

    # Plotting
    st.subheader("📊 Deal Success Distribution")
    st.plotly_chart(px.histogram(df, x="Prediction", color="Prediction"))

    # GPT-style Commentary
    st.subheader("🧠 AI Commentary")
    commentaries = df.apply(gpt_commentary, axis=1)
    df['GPT_Commentary'] = commentaries
    st.dataframe(df[['Prediction', 'GPT_Commentary']])

    # Recommendations
    df['Recommendation'] = df.apply(recommend_deal, axis=1)
    st.subheader("✅ Final Recommendation")
    st.dataframe(df[['Prediction', 'Recommendation']])

    # PDF Report
    if st.button("📥 Generate Report"):
        path = generate_pdf_report(df)
        with open(path, "rb") as f:
            st.download_button("📄 Download Report", f, "deal_report.pdf", mime="application/pdf")
