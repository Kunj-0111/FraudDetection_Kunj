import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    data = {
        "TransactionAmt": np.random.randint(10, 5000, 2000),
        "TransactionDT": np.random.randint(100000, 500000, 2000),
        "isFraud": np.random.randint(0, 2, 2000)
    }
    return pd.DataFrame(data)

df = load_data()

# ------------------ FALLBACK MODEL ------------------
class DummyModel:
    def predict_proba(self, X):
        return np.array([[0.3, 0.7] for _ in range(len(X))])

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_all():
    BASE_DIR = os.path.dirname(__file__)

    try:
        model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
        scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))
        feature_cols = pickle.load(open(os.path.join(BASE_DIR, "features.pkl"), "rb"))
        return model, scaler, feature_cols

    except:
        st.warning("⚠️ Model files not found. Running in demo mode.")
        return DummyModel(), None, ["TransactionAmt", "TransactionDT"]

model, scaler, feature_cols = load_all()

# ------------------ PREPROCESS ------------------
def preprocess(df):
    df = df.copy()

    df['AmtToMeanRatio'] = df['TransactionAmt'] / df['TransactionAmt'].mean()
    df['HourOfDay'] = (df['TransactionDT'] // 3600) % 24
    df['DeviceRisk'] = 0

    return df

# ------------------ FEATURE PREP ------------------
X_raw = df.drop("isFraud", axis=1)
X = preprocess(X_raw)

# अगर scaler available है तभी apply करो
if scaler is not None:
    X = X.reindex(columns=feature_cols, fill_value=0)
    X = pd.DataFrame(scaler.transform(X), columns=feature_cols)

# ------------------ SIDEBAR ------------------
st.sidebar.title("⚙️ Controls")
page = st.sidebar.radio("Navigation", ["Overview", "Explorer", "SHAP"])

# ------------------ PAGE 1 ------------------
if page == "Overview":

    st.title("📊 Fraud Detection Overview")

    total = len(df)
    fraud = int(df["isFraud"].sum())
    rate = fraud / total if total > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", total)
    col2.metric("Fraud Cases", fraud)
    col3.metric("Detection Rate", f"{rate:.2%}")

    st.dataframe(df.head(50))

# ------------------ PAGE 2 ------------------
elif page == "Explorer":

    st.title("🔍 Transaction Explorer")

    txn_id = st.number_input("Transaction Index", 0, len(X)-1, step=1)

    row_df = pd.DataFrame([X.iloc[txn_id]])
    prob = model.predict_proba(row_df)[0][1]

    st.dataframe(df.iloc[[txn_id]])

    st.progress(float(prob))
    st.write(f"Probability: {prob:.4f}")

    if prob >= 0.75:
        st.error("🔴 High Risk")
    elif prob >= 0.40:
        st.warning("🟡 Suspicious")
    else:
        st.success("🟢 Safe")

# ------------------ PAGE 3 ------------------
elif page == "SHAP":

    st.title("🧠 SHAP Explainability")

    st.subheader("Global Feature Importance")
    st.image("shap_summary.png")

    st.subheader("Case Explanations")
    st.image("charts/waterfall_fraud.png")
    st.image("charts/waterfall_border.png")
    st.image("charts/waterfall_normal.png")
