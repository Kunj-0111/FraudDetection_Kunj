import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv("../data/train_transaction.csv")

df = load_data()
df = df.sample(5000, random_state=42)  # FIXED

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_all():
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    feature_cols = pickle.load(open("features.pkl", "rb"))
    return model, scaler, feature_cols

model, scaler, feature_cols = load_all()


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


X_raw = df.drop("isFraud", axis=1)
X = preprocess(X_raw)
X = X.reindex(columns=feature_cols, fill_value=0)
X = pd.DataFrame(scaler.transform(X), columns=feature_cols)


st.sidebar.title("⚙️ Controls")

page = st.sidebar.radio("Navigation", ["Overview", "Explorer", "SHAP"])

st.sidebar.write(f"📊 Total Sample Loaded: {len(df)}")

min_amt = st.sidebar.slider(
    "Minimum Transaction Amount",
    0,
    int(df["TransactionAmt"].max()),
    0
)

df = df[df["TransactionAmt"] >= min_amt]

st.sidebar.write(f"📉 After Filter: {len(df)}")

if page == "Overview":

    st.title("📊 Fraud Detection Overview")
    st.markdown("### 📈 Real-Time Fraud Detection Dashboard")

    total = len(df)
    fraud = int(df["isFraud"].sum())
    rate = fraud / total
    avg_amt = df[df["isFraud"] == 1]["TransactionAmt"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", total)
    c2.metric("Fraud Cases", fraud)
    c3.metric("Detection Rate", f"{rate:.2%}")
    c4.metric("Avg Fraud Amount", f"${avg_amt:.2f}")

    fig = px.histogram(df, x="TransactionAmt", color="isFraud", log_y=True)
    st.plotly_chart(fig, use_container_width=True)


elif page == "Explorer":

    st.title("🔍 Transaction Explorer")
    st.markdown("### 🔎 Analyze Individual Transaction")

    txn_id = st.number_input("Transaction Index", 0, len(X)-1)

    row_df = pd.DataFrame([X.iloc[txn_id]])
    prob = model.predict_proba(row_df)[0][1]

    st.subheader("📄 Transaction Details")
    st.dataframe(df.iloc[[txn_id]])

    st.subheader("⚠️ Risk Score")
    st.progress(float(prob))
    st.write(f"Probability: **{prob:.4f}**")

    if prob >= 0.75:
        st.error("🔴 High Risk Transaction")
    elif prob >= 0.40:
        st.warning("🟡 Suspicious Transaction")
    else:
        st.success("🟢 Safe Transaction")


elif page == "SHAP":

    st.title("🧠 SHAP Explainability")
    st.markdown("### 🧠 Model Decision Breakdown")

    txn_id = st.number_input("Transaction Index", 0, len(X)-1)

    row_df = pd.DataFrame([X.iloc[txn_id]])

    pred = model.predict(row_df)[0]
    prob = model.predict_proba(row_df)[0][1]

    st.subheader("Prediction")
    st.write("Fraud" if pred == 1 else "Legitimate")

    st.subheader("Probability")
    st.write(f"{prob:.4f}")

    # SHAP (CORRECT)
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    st.subheader("📊 SHAP Waterfall")

    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[txn_id], show=False)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("📝 Explanation")

    if prob >= 0.75:
        st.error("High risk due to strong fraud indicators.")
    elif prob >= 0.40:
        st.warning("Borderline case with mixed signals.")
    else:
        st.success("Transaction appears normal.")
