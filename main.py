import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import plotly.express as px

# Load data
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("creditcard.csv")

# Preprocess
df['Hour'] = df['Time'] // 3600
LABELS = ["Normal", "Fraud"]

st.title("📊 Credit Card Fraud Detection Dashboard with Prediction")

# Sidebar controls
st.sidebar.header("🔧 Filter & Model Settings")
model_choice = st.sidebar.selectbox("Select Model", ["Isolation Forest", "Local Outlier Factor", "One-Class SVM"])
amount_range = st.sidebar.slider("Transaction Amount Range", float(df['Amount'].min()), float(df['Amount'].max()), (0.0, 500.0))
hour_range = st.sidebar.slider("Hour of Transaction", int(df['Hour'].min()), int(df['Hour'].max()), (0, 24))

# Filtered data
df_filtered = df[(df['Amount'] >= amount_range[0]) & (df['Amount'] <= amount_range[1])]
df_filtered = df_filtered[(df_filtered['Hour'] >= hour_range[0]) & (df_filtered['Hour'] <= hour_range[1])]

# Preview
st.subheader("🗂️ Data Preview")
st.dataframe(df_filtered.head(100))

# Class distribution
st.subheader("🔍 Class Distribution")
fig, ax = plt.subplots()
sns.countplot(x='Class', data=df_filtered, ax=ax)
ax.set_xticklabels(LABELS)
st.pyplot(fig)

# Amount distribution
st.subheader("💵 Transaction Amount Distribution by Class")
fig, ax = plt.subplots()
sns.histplot(data=df_filtered, x='Amount', hue='Class', bins=50, kde=True, ax=ax)
st.pyplot(fig)

# Hourly Fraud Rate
st.subheader("⏰ Hourly Fraud Rate")
hourly_stats = df.groupby("Hour")["Class"].mean().reset_index()
fig = px.line(hourly_stats, x='Hour', y='Class', title='Average Fraud Rate by Hour')
st.plotly_chart(fig)

# Correlation Heatmap
st.subheader("📌 Correlation Heatmap")
corr = df_filtered.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, cmap='coolwarm', annot=False, ax=ax)
st.pyplot(fig)

# Prediction block
st.subheader("🚨 Fraud Detection Prediction")
X = df_filtered.drop(['Class', 'Time'], axis=1)
y = df_filtered['Class']

if model_choice == "Isolation Forest":
    model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
elif model_choice == "Local Outlier Factor":
    model = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
elif model_choice == "One-Class SVM":
    model = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.01)

if model_choice == "Local Outlier Factor":
    y_pred = model.fit_predict(X)
else:
    model.fit(X)
    y_pred = model.predict(X)

# Convert predictions: 1 -> Normal, -1 -> Fraud
y_pred = [1 if i == 1 else 0 for i in y_pred]
df_filtered['Predicted'] = y_pred

# Summary report
st.markdown("### 📈 Prediction Summary")
st.text(classification_report(y, y_pred, target_names=LABELS))

# Confusion matrix
st.subheader("📉 Confusion Matrix")
cm = confusion_matrix(y, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
st.pyplot(fig)

# Result distribution
st.subheader("📌 Prediction Results Distribution")
fig, ax = plt.subplots()
sns.countplot(x='Predicted', data=df_filtered, ax=ax)
ax.set_xticklabels(LABELS)
st.pyplot(fig)

# Export option
st.sidebar.markdown("---")
if st.sidebar.button("📥 Download Predicted Data"):
    st.sidebar.download_button(
        label="Download CSV",
        data=df_filtered.to_csv(index=False).encode('utf-8'),
        file_name='fraud_predictions.csv',
        mime='text/csv'
    )

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")
