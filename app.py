import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ================= LOAD MODEL =================
model = pickle.load(open("model.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Student Performance AI",
    layout="wide",
    page_icon="🎓"
)

# ================= TITLE =================
st.title("🎓 Student Performance AI Dashboard (PRO)")
st.write("AI system that predicts student performance + explains results like a real product")

# ================= SIDEBAR INPUT =================
st.sidebar.header("📊 Student Input Panel")

user_input = {}

user_input['age'] = st.sidebar.slider("Age", 15, 22, 17)
user_input['studytime'] = st.sidebar.slider("Study Time", 1, 4, 2)
user_input['failures'] = st.sidebar.slider("Failures", 0, 4, 0)
user_input['absences'] = st.sidebar.slider("Absences", 0, 100, 5)
user_input['G1'] = st.sidebar.slider("G1 Score", 0, 20, 10)
user_input['G2'] = st.sidebar.slider("G2 Score", 0, 20, 10)

# Feature engineering
user_input['G_avg'] = (user_input['G1'] + user_input['G2']) / 2
user_input['G_ratio'] = user_input['G2'] / (user_input['G1'] + 1)

# Categorical
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
input_df = input_df[features]

# ================= MODEL PREDICTION =================
if st.button("🚀 Generate AI Report"):
    
    prediction = model.predict(input_df)[0]

    st.divider()

    # ================= REPORT CARD =================
    st.subheader("📊 Performance Report Card")

    if prediction >= 15:
        grade = "A (Excellent)"
        emoji = "🌟"
        color_box = "#0f5132"
        remark = "Outstanding performance. Student is highly capable."
    elif prediction >= 10:
        grade = "B (Good)"
        emoji = "👍"
        color_box = "#084298"
        remark = "Good performance, but room for improvement."
    else:
        grade = "C (Weak)"
        emoji = "⚠️"
        color_box = "#842029"
        remark = "Student needs serious attention."

    st.markdown(f"""
    <div style="
        background-color:{color_box};
        padding:25px;
        border-radius:15px;
        color:white;
        text-align:center;
    ">
        <h1>{emoji} Predicted Score: {prediction:.2f}</h1>
        <h2>🏆 Grade: {grade}</h2>
        <p style="font-size:18px;">{remark}</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ================= AI EXPLANATION =================
    st.subheader("🧠 AI Explanation (Why this result?)")

    reasons = []

    if user_input['G2'] > 15:
        reasons.append("Strong previous performance (G2 is high)")
    if user_input['studytime'] >= 3:
        reasons.append("Good study time habits")
    if user_input['failures'] > 0:
        reasons.append("Past academic failures affected prediction")
    if user_input['absences'] > 20:
        reasons.append("High absences reduced score")
    if user_input['G1'] < 10:
        reasons.append("Weak foundation in earlier exam (G1 low)")

    if len(reasons) == 0:
        reasons.append("Balanced academic profile with no strong weaknesses")

    for i, r in enumerate(reasons, 1):
        st.write(f"🔹 {i}. {r}")

    st.divider()

    # ================= IMPROVEMENT PLAN =================
    st.subheader("💡 Improvement Plan")

    if prediction < 10:
        st.error("📌 Focus on basics + daily study routine")
        st.error("📌 Reduce absences immediately")
        st.error("📌 Improve past exam preparation")

    elif prediction < 15:
        st.info("📌 Improve consistency in study")
        st.info("📌 Practice past papers regularly")
        st.info("📌 Increase revision time")

    else:
        st.success("📌 Maintain current performance")
        st.success("📌 Focus on advanced concepts")
        st.success("📌 Help other students to strengthen knowledge")

    st.divider()

    # ================= SUMMARY =================
    st.subheader("📌 Final Summary")

    st.markdown(f"""
    - 🎯 Final Score: **{prediction:.2f}**
    - 🏆 Grade: **{grade}**
    - 🧠 Key Insight: Model analyzed academic history + behavior patterns
    """)
st.write("This AI system Provides a comprehensive analysis of student performance")

# Footer
st.markdown("Made By Muhammad Umer")
