
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.config import load_config
from src.data_loading import read_csv, coerce_datetime
from src.modeling import load_pipeline
from src.nlp_text import join_text_columns
from src.visualization import plot_ticket_trends

st.set_page_config(page_title="Customer Satisfaction Prediction", layout="wide")

st.title("📊 Customer Satisfaction Prediction (End-to-End)")

cfg = load_config("config/config.yaml")

st.sidebar.header("Settings")
data_source = st.sidebar.selectbox("Data source", ["Sample CSV", "Upload CSV"])

if data_source == "Sample CSV":
    data_path = "data/sample_customer_support_tickets_small.csv"
    df = read_csv(data_path)
else:
    up = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
    else:
        st.info("Please upload a CSV to proceed.")
        st.stop()

st.subheader("Data Preview")
st.dataframe(df.head(50))

# EDA: Trends
with st.expander("Ticket Trends Over Time"):
    if "Date of Purchase" in df.columns:
        fig = plot_ticket_trends(df, "Date of Purchase")
        st.pyplot(fig)
    else:
        st.warning("'Date of Purchase' column not found.")

# Predict
st.header("🔮 Predict Satisfaction")
try:
    pipe = load_pipeline(cfg.paths.model_dir)
except Exception as e:
    st.error("Model not found. Please run training first (python train.py).")
    st.stop()

X = df.copy()
X = coerce_datetime(X, cfg.columns.datetime)
X = join_text_columns(X, cfg.columns.text)
preds = pipe.predict(X)
st.write("First 20 predictions:")
st.dataframe(pd.DataFrame({"Predicted Satisfaction": preds}).head(20))

# Download predictions
pred_df = df.copy()
pred_df["Predicted Satisfaction"] = preds
st.download_button("Download predictions.csv", pred_df.to_csv(index=False), file_name="predictions.csv")
