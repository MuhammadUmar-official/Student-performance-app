import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ================= LOAD MODEL & FEATURES =================
model = pickle.load(open("model.pkl", "rb"))  # full pipeline
features = pickle.load(open("features.pkl", "rb"))

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Student Performance Predictor", layout="wide")

st.title("🎓 Student Performance Predictor (G3)")
st.write("Predict final grade using ML model")

# ================= SIDEBAR =================
st.sidebar.header("📊 Enter Student Details")

numeric_cols = ['age', 'studytime', 'failures', 'absences', 'G1', 'G2']

categorical_cols = [
    'school', 'sex', 'famsize',
    'schoolsup', 'famsup', 'paid',
    'activities', 'higher', 'internet'
]

user_input = {}

# ================= NUMERIC INPUTS =================
user_input['age'] = st.sidebar.slider("Age", 15, 22, 17)
user_input['studytime'] = st.sidebar.slider("Study Time", 1, 4, 2)
user_input['failures'] = st.sidebar.slider("Failures", 0, 4, 0)
user_input['absences'] = st.sidebar.slider("Absences", 0, 100, 5)

user_input['G1'] = st.sidebar.slider("G1 Score", 0, 20, 10)
user_input['G2'] = st.sidebar.slider("G2 Score", 0, 20, 10)

# ================= FEATURE ENGINEERING =================
user_input['G_avg'] = (user_input['G1'] + user_input['G2']) / 2
user_input['G_ratio'] = user_input['G2'] / (user_input['G1'] + 1)

# ================= CATEGORICAL INPUTS =================
user_input['school'] = st.sidebar.selectbox("School", ["GP", "MS"])
user_input['sex'] = st.sidebar.selectbox("Gender", ["M", "F"])
user_input['famsize'] = st.sidebar.selectbox("Family Size", ["LE3", "GT3"])
user_input['schoolsup'] = st.sidebar.selectbox("School Support", ["yes", "no"])
user_input['famsup'] = st.sidebar.selectbox("Family Support", ["yes", "no"])
user_input['paid'] = st.sidebar.selectbox("Paid Classes", ["yes", "no"])
user_input['activities'] = st.sidebar.selectbox("Activities", ["yes", "no"])
user_input['higher'] = st.sidebar.selectbox("Higher Education", ["yes", "no"])
user_input['internet'] = st.sidebar.selectbox("Internet Access", ["yes", "no"])

# ================= DATAFRAME =================
input_df = pd.DataFrame([user_input])

# 🔥 IMPORTANT: enforce correct feature order
input_df = input_df[features]

# ================= SHOW INPUT =================
st.subheader("📋 Input Data")
st.dataframe(input_df)

# ================= PREDICTION =================
if st.button("🚀 Predict G3 Score"):
    try:
        prediction = model.predict(input_df)[0]

        st.subheader("🎯 Result")
        st.success(f"Predicted G3 Score: {prediction:.2f}")

        # Performance labels
        if prediction >= 15:
            st.balloons()
            st.success("🌟 Excellent Performance")
        elif prediction >= 10:
            st.info("👍 Average Performance")
        else:
            st.error("⚠️ Needs Improvement")

    except Exception as e:
        st.error(f"Prediction Error: {e}")

# ================= DEBUG =================
if st.checkbox("🔍 Show Features"):
    st.write(features)
