# ================= IMPORTS =================
import streamlit as st
import pandas as pd
import joblib

# ================= LOAD MODEL & FEATURES =================
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")

# ================= APP CONFIG =================
st.set_page_config(page_title="Student Performance Predictor", layout="wide")

st.title("🎓 Student Performance Predictor")
st.write("Predict final grade (G3) based on student information.")

# ================= USER INPUT =================
st.sidebar.header("Input Student Data")

def user_input_features():
    # Numeric sliders
    age = st.sidebar.slider("Age", 15, 22, 17)
    studytime = st.sidebar.slider("Weekly Study Time (1–5)", 1, 5, 3)
    failures = st.sidebar.slider("Past Failures", 0, 4, 0)
    absences = st.sidebar.slider("Absences", 0, 50, 5)
    G1 = st.sidebar.slider("Grade G1 (first period)", 0, 20, 10)
    G2 = st.sidebar.slider("Grade G2 (second period)", 0, 20, 10)

    # Engineered features (same as training)
    G_avg = (G1 + G2) / 2
    G_ratio = G2 / (G1 + 1)

    # Categorical inputs
    school = st.sidebar.selectbox("School", ("GP", "MS"))
    sex = st.sidebar.selectbox("Sex", ("M", "F"))
    famsize = st.sidebar.selectbox("Family Size", ("LE3", "GT3"))
    schoolsup = st.sidebar.selectbox("Extra School Support?", ("yes", "no"))
    famsup = st.sidebar.selectbox("Family Support?", ("yes", "no"))
    paid = st.sidebar.selectbox("Paid Classes?", ("yes", "no"))
    activities = st.sidebar.selectbox("Extra-curricular Activities?", ("yes", "no"))
    higher = st.sidebar.selectbox("Wants Higher Education?", ("yes", "no"))
    internet = st.sidebar.selectbox("Internet Access?", ("yes", "no"))

    data = {
        'school': school,
        'sex': sex,
        'age': age,
        'famsize': famsize,
        'studytime': studytime,
        'failures': failures,
        'schoolsup': schoolsup,
        'famsup': famsup,
        'paid': paid,
        'activities': activities,
        'higher': higher,
        'internet': internet,
        'absences': absences,
        'G1': G1,
        'G2': G2,
        'G_avg': G_avg,
        'G_ratio': G_ratio
    }

    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# ================= PREDICTION =================
prediction = model.predict(input_df)[0]

st.subheader("🎯 Predicted Final Grade (G3)")
st.success(f"{prediction:.2f} / 20")

# ================= FEATURE IMPORTANCE =================
st.subheader("📊 Top Feature Importance")

try:
    # Get model and preprocessor
    rf = model.named_steps['regressor']
    preprocessor = model.named_steps['preprocessor']

    # Feature names
    numeric_cols = ['age','studytime','failures','absences','G1','G2','G_avg','G_ratio']

    categorical_cols = [
        'school','sex','famsize',
        'schoolsup','famsup','paid','activities',
        'higher','internet'
    ]

    cat_features = preprocessor.named_transformers_['cat'] \
        .named_steps['encoder'] \
        .get_feature_names_out(categorical_cols)

    all_features = list(numeric_cols) + list(cat_features)

    # Importance
    importances = rf.feature_importances_

    feature_importance_df = pd.DataFrame({
        'feature': all_features,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    top_features = feature_importance_df.head(10)

    st.table(top_features)
    st.bar_chart(top_features.set_index('feature')['importance'])

except Exception as e:
    st.warning(f"Feature importance not available: {e}")

# ================= FOOTER =================
st.write("🔥 Model: Random Forest (R² ≈ 0.85)")