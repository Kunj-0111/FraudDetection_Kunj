LIVE URL:- https://fraud-detection-kunj.streamlit.app/

"Due to file size limitations, full dataset is not included. Sample/demo data used for deployment."

# 💳 Real-Time Fraud Detection System with Explainable AI

## 📌 Overview
This project builds a **real-time fraud detection system** using machine learning and Explainable AI (SHAP).  
It detects fraudulent transactions, handles imbalanced data, and provides clear explanations through an interactive dashboard.

---

## 🎯 Problem Statement
Financial fraud causes huge losses globally. Traditional systems fail to detect new patterns and lack transparency.

This project aims to:
- Detect fraud in real-time  
- Handle imbalanced datasets  
- Provide explainable predictions  
- Build a live dashboard for analysis  

---

## 📊 Dataset
Dataset: IEEE-CIS Fraud Detection (Kaggle)

- ~590,000 transactions  
- 433 features  
- ~3.5% fraud rate  

Files used:
- `train_transaction.csv`
- `train_identity.csv`

---

## ⚙️ Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost, LightGBM  
- SHAP  
- Matplotlib, Seaborn, Plotly  
- Streamlit  

---

## 🧠 ML Pipeline
1. Data Cleaning & Missing Value Handling  
2. Feature Engineering  
3. Handling Class Imbalance (SMOTE)  
4. Feature Scaling  
5. Model Training & Evaluation  
6. Threshold Optimization  
7. SHAP Explainability  

---

## 🤖 Models Used
- LightGBM  
- XGBoost ⭐ (Best Model)  
- Isolation Forest  

---

## 📈 Features
- Real-time fraud prediction  
- Risk segmentation (Critical, Suspicious, Clear)  
- SHAP explainability  
- Interactive dashboard  
- Threshold optimization  

---

## 📊 Visualizations
- SHAP Summary Plot  
- Fraud Rate by Hour  
- Transaction Amount Distribution  
- Risk Tier Chart  
- Precision-Recall Curve  

---

## 🧠 Key Insights
- High transaction amounts increase fraud risk  
- Fraud occurs more during late-night hours  
- Suspicious devices show higher fraud probability  

---

## 🚨 Risk Segmentation
- 🔴 Critical Risk → ≥ 0.75  
- 🟡 Suspicious → 0.40 – 0.74  
- 🟢 Clear → < 0.40  

---

## 💡 Business Recommendations
- Block high-risk transactions in real-time  
- Use multi-factor authentication for suspicious transactions  

---

## 🖥️ Dashboard

Run locally:

```bash
cd dashboard
streamlit run app.py
