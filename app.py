
import streamlit as st
import pandas as pd
import numpy as np

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(page_title="CardioLens", page_icon="❤️", layout="wide")

st.title("❤️ CardioLens")
st.subheader("Exploring Cardiovascular Health Patterns Through Machine Learning")
st.write("A machine-learning demonstration using the UCI Heart Disease dataset.")

@st.cache_data
def load_data():
    heart = fetch_ucirepo(id=45)
    data = heart.data.original.copy()
    data = data.replace("?", np.nan)

    for c in data.columns:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    data = data.dropna(subset=["num"])
    data["target"] = (data["num"] > 0).astype(int)
    data = data.drop(columns=["num"])
    return data

@st.cache_resource
def train_model():
    data = load_data()
    X = data.drop(columns=["target"])
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1 Score": f1_score(y_test, predictions, zero_division=0)
    }

    return model, X, results

try:
    data = load_data()
    model, X, results = train_model()

    st.success("Model trained successfully!")

    st.markdown("### Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(data))
    col2.metric("Total Features", len(X.columns))
    col3.metric("Model", "Logistic Regression")

    st.markdown("### Model Performance")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{results['Accuracy']*100:.2f}%")
    c2.metric("Precision", f"{results['Precision']*100:.2f}%")
    c3.metric("Recall", f"{results['Recall']*100:.2f}%")
    c4.metric("F1 Score", f"{results['F1 Score']*100:.2f}%")

    st.bar_chart(pd.DataFrame({"Score": results}))

    st.markdown("---")
    st.markdown("## Live Prediction Demo")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 20, 90, 50)
        sex = st.selectbox("Sex", ["Female", "Male"])
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
        trestbps = st.slider("Resting Blood Pressure", 80, 200, 130)
        chol = st.slider("Cholesterol", 100, 600, 240)
        fbs = st.selectbox("Fasting Blood Sugar > 120", ["No", "Yes"])
        restecg = st.selectbox("Resting ECG", [0, 1, 2])

    with col2:
        thalach = st.slider("Maximum Heart Rate", 60, 220, 150)
        exang = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
        oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0, 0.1)
        slope = st.selectbox("Slope", [0, 1, 2])
        ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3, 4])
        thal = st.selectbox("Thal", [0, 1, 2, 3])

    if st.button("Run CardioLens Prediction", type="primary"):

        sample = pd.DataFrame([{
            "age": age,
            "sex": 1 if sex == "Male" else 0,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": 1 if fbs == "Yes" else 0,
            "restecg": restecg,
            "thalach": thalach,
            "exang": 1 if exang == "Yes" else 0,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal
        }])

        prediction = model.predict(sample)[0]
        probability = model.predict_proba(sample)[0][1]

        st.markdown("### Prediction Result")

        if prediction == 1:
            st.warning("Classification: Positive class")
        else:
            st.success("Classification: Negative class")

        st.metric("Positive-Class Probability", f"{probability*100:.2f}%")

        st.info("This is an educational demonstration using a historical dataset. It is not a medical diagnosis or a prediction of future heart events.")

    st.markdown("---")
    st.markdown("### About CardioLens")
    st.write("""
    - Data preparation
    - Missing value handling
    - Machine-learning training
    - Model evaluation
    - Interactive classification
    """)

    st.caption("Dataset: UCI Heart Disease dataset.")

except Exception as e:
    st.error("Something went wrong.")
    st.exception(e)
