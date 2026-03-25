import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ================= OUTLIER FUNCTION (ADD HERE) =================
def iqr_capping(data):
    data = pd.DataFrame(data).copy()
    for col in data.columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        data[col] = np.where(data[col] < lower, lower, data[col])
        data[col] = np.where(data[col] > upper, upper, data[col])
    return data


# ================= LOAD MODEL =================
model = joblib.load("model.pkl")

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Student Performance Predictor", layout="wide")

st.title("🎓 Student Performance Predictor (G3)")
st.write("Predict student final grade using Machine Learning")

# ================= DEBUG (FEATURE CHECK) =================
if st.checkbox("🔍 Show Model Features"):
    st.write(model.feature_names_in_)

# ================= SIDEBAR INPUTS =================
st.sidebar.header("📊 Enter Student Details")

# -------- NUMERIC FEATURES --------
age = st.sidebar.slider("Age", 15, 22, 17)
studytime = st.sidebar.slider("Study Time", 1, 4, 2)
failures = st.sidebar.slider("Past Failures", 0, 4, 0)
absences = st.sidebar.slider("Absences", 0, 100, 5)

# -------- CATEGORICAL FEATURES --------
school = st.sidebar.selectbox("School", ["GP", "MS"])
sex = st.sidebar.selectbox("Gender", ["M", "F"])
famsize = st.sidebar.selectbox("Family Size", ["LE3", "GT3"])

schoolsup = st.sidebar.selectbox("School Support", ["yes", "no"])
famsup = st.sidebar.selectbox("Family Support", ["yes", "no"])
paid = st.sidebar.selectbox("Paid Classes", ["yes", "no"])
activities = st.sidebar.selectbox("Extra Activities", ["yes", "no"])
higher = st.sidebar.selectbox("Higher Education", ["yes", "no"])
internet = st.sidebar.selectbox("Internet Access", ["yes", "no"])

# ================= CREATE INPUT DATA =================
input_data = pd.DataFrame([{
    "school": school,
    "sex": sex,
    "age": age,
    "famsize": famsize,
    "studytime": studytime,
    "failures": failures,
    "schoolsup": schoolsup,
    "famsup": famsup,
    "paid": paid,
    "activities": activities,
    "higher": higher,
    "internet": internet,
    "absences": absences
}])

# ================= SHOW INPUT =================
st.subheader("📋 Input Data")
st.dataframe(input_data)

# ================= PREDICTION =================
if st.button("🚀 Predict G3 Score"):
    
    prediction = model.predict(input_data)[0]
    
    st.subheader("🎯 Prediction Result")
    st.success(f"Predicted Final Grade: {prediction:.2f}")

    # Performance Level
    if prediction >= 15:
        st.balloons()
        st.success("🌟 Excellent Performance")
    elif prediction >= 10:
        st.info("👍 Average Performance")
    else:
        st.error("⚠️ Poor Performance - Needs Improvement")
