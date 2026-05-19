import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    return pd.read_csv("../data/train_transaction.csv")

df = load_data()
df = df.sample(2000, random_state=42)  # LOWER = safer

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_all():
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    feature_cols = pickle.load(open("features.pkl", "rb"))
    return model, scaler, feature_cols

model, scaler, feature_cols = load_all()

# ------------------ PREPROCESS ------------------
def preprocess(df):
    df = df.copy()

    num_cols = df.select_dtypes(include=['int64','float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    df['AmtToMeanRatio'] = df['TransactionAmt'] / df['TransactionAmt'].mean()
    df['HourOfDay'] = (df['TransactionDT'] // 3600) % 24

    if 'DeviceInfo' in df.columns:
        df['DeviceRisk'] = df['DeviceInfo'].astype(str).str.contains('mobile', case=False).astype(int)
    else:
        df['DeviceRisk'] = 0

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()

    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    return df

# ------------------ FEATURE PREP ------------------
X_raw = df.drop("isFraud", axis=1)
X = preprocess(X_raw)
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

    st.metric("Total Transactions", total)
    st.metric("Fraud Cases", fraud)
    st.metric("Detection Rate", f"{rate:.2%}")

    st.write("### Sample Data")
    st.dataframe(df.head(50))

# ------------------ PAGE 2 ------------------
elif page == "Explorer":

    st.title("🔍 Transaction Explorer")

    txn_id = st.number_input("Transaction Index", 0, len(X)-1)

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

    st.image("shap_summary.png")
    st.image("charts/waterfall_fraud.png")
    st.image("charts/waterfall_border.png")
    st.image("charts/waterfall_normal.png")
