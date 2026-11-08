import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)

TARGET = "deposit"
MODEL_DIR = "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

st.set_page_config(page_title="Bank Deposit Classifier", page_icon="🏦", layout="wide")
st.title("🏦 Bank Term-Deposit Prediction: Model Comparison")
st.caption("Upload test data, pick a saved model, and view its evaluation on that data. No retraining happens here.")


@st.cache_resource
def load_model(fname):
    return joblib.load(os.path.join(MODEL_DIR, fname))


def encode_target(y, classes):
    """Align the uploaded target values with the labels the model was trained on."""
    y = pd.Series(y)
    classes = list(classes)
    if y.dtype == object and not isinstance(classes[0], str):
        y = y.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0})
    elif y.dtype == object:
        y = y.astype(str).str.strip()
    return y


# ---------- Sidebar ----------
st.sidebar.header("Controls")
uploaded = st.sidebar.file_uploader("Upload test CSV (with target column)", type=["csv"])
model_name = st.sidebar.selectbox("Select model", list(MODEL_FILES.keys()))

if uploaded is None:
    st.info("👈 Upload your test CSV from the sidebar to begin. It must contain all input features and the 'deposit' column.")
    st.stop()

df = pd.read_csv(uploaded)
target_col = next((c for c in df.columns if c.lower() == TARGET), None)
if target_col is None:
    st.error(f"Target column '{TARGET}' not found. Columns present: {list(df.columns)}")
    st.stop()

with st.expander("Preview uploaded data", expanded=False):
    st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    st.dataframe(df.head(20))

X = df.drop(columns=[target_col])
model = load_model(MODEL_FILES[model_name])
classes = model.classes_
y_true = encode_target(df[target_col], classes)

if y_true.isna().any():
    st.error("Could not map some target values to the model's classes. Check the 'deposit' column values.")
    st.stop()

try:
    y_pred = model.predict(X)
except Exception as e:
    st.error(f"Prediction failed. Uploaded columns may not match training features.\n\n{e}")
    st.stop()

# ---------- Metrics ----------
binary = len(classes) == 2
avg = "binary" if binary else "weighted"
pos = classes[1]

if hasattr(model, "predict_proba"):
    proba = model.predict_proba(X)
else:
    proba = None

try:
    if binary:
        auc = roc_auc_score(y_true, proba[:, 1] if proba is not None else model.decision_function(X))
    else:
        auc = roc_auc_score(y_true, proba, multi_class="ovr", average="weighted")
except Exception:
    auc = float("nan")

kw = {"pos_label": pos} if binary else {}
metrics = {
    "Accuracy": accuracy_score(y_true, y_pred),
    "AUC": auc,
    "Precision": precision_score(y_true, y_pred, average=avg, zero_division=0, **kw),
    "Recall": recall_score(y_true, y_pred, average=avg, zero_division=0, **kw),
    "F1": f1_score(y_true, y_pred, average=avg, zero_division=0, **kw),
    "MCC": matthews_corrcoef(y_true, y_pred),
}

st.subheader(f"Results: {model_name}")
cols = st.columns(6)
for col, (k, v) in zip(cols, metrics.items()):
    col.metric(k, "N/A" if np.isnan(v) else f"{v:.4f}")

# ---------- Confusion matrix + report ----------
left, right = st.columns(2)
with left:
    st.markdown("**Confusion Matrix**")
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.markdown("**Classification Report**")
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(rep).transpose().round(3))

# ---------- All-model comparison on uploaded data ----------
st.subheader("Compare all models on this data")
if st.button("Run all 5 models"):
    rows = []
    for name, f in MODEL_FILES.items():
        m = load_model(f)
        yp = m.predict(X)
        yt = encode_target(df[target_col], m.classes_)
        try:
            if binary:
                s = m.predict_proba(X)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(X)
                a = roc_auc_score(yt, s)
            else:
                a = roc_auc_score(yt, m.predict_proba(X), multi_class="ovr", average="weighted")
        except Exception:
            a = float("nan")
        k2 = {"pos_label": m.classes_[1]} if binary else {}
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(yt, yp),
            "AUC": a,
            "Precision": precision_score(yt, yp, average=avg, zero_division=0, **k2),
            "Recall": recall_score(yt, yp, average=avg, zero_division=0, **k2),
            "F1": f1_score(yt, yp, average=avg, zero_division=0, **k2),
            "MCC": matthews_corrcoef(yt, yp),
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model").round(4))
