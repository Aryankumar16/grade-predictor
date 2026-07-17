import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# 1. PAGE SETUP
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Student Performance Predictor")
st.caption("Predict academic outcomes using historical data or lifestyle habits.")
st.divider()


def classify(score):
    if score >= 16:
        return "Excellent"
    elif score >= 12:
        return "Good"
    elif score >= 9:
        return "Average"
    else:
        return "Poor"

def score_badge(score: float) -> str:
    if score >= 85:
        return "Distinction"
    elif score >= 70:
        return "Merit"
    elif score >= 55:
        return "Pass"
    else:
        return "Below Pass"

# Color mappings matching the style
PERF_COLORS = {"Excellent": "#2ecc71", "Good": "#3498db", "Average": "#f39c12", "Poor": "#e74c3c"}
BADGE_COLORS = {"Distinction": "#2ecc71", "Merit": "#3498db", "Pass": "#f39c12", "Below Pass": "#e74c3c"}

# 3. LOAD PRE-TRAINED MODELS
@st.cache_resource
def load_models():
    model1 = joblib.load("dt.pkl")
    model2 = joblib.load("xgb.pkl")
    encoders2 = joblib.load("encoders.pkl")
    features2 = joblib.load("features.pkl")
    return model1, model2, encoders2, features2

try:
    model1, model2, encoders2, feature_cols2 = load_models()
except Exception as e:
    st.error("Error loading model files. Please ensure all .pkl files are in the same directory.")
    st.stop()

tab1, tab2 = st.tabs([" Model 1: Grade Classifier", " Model 2: Early Score Predictor"])


# TAB 1: GRADE PERFORMANCE CLASSIFIER

with tab1:
    st.markdown("### Predict final grade category based on current progress.")
    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.subheader(" Behavior & Habits")
        studytime = st.selectbox("Weekly Study Time", [1, 2, 3, 4], 
                                 format_func=lambda x: ["< 2 hrs", "2-5 hrs", "5-10 hrs", "> 10 hrs"][x-1])
        failures = st.slider("Past Class Failures", 0, 3, 0)
        absences = st.slider("School Absences", 0, 93, 5)
        goout = st.slider("Frequency of Going Out (1-5)", 1, 5, 3)
        activities = st.radio("Extra-Curricular Activities?", ["Yes", "No"], horizontal=True)
        internet = st.radio("Home Internet Access?", ["Yes", "No"], horizontal=True)

    with col_r:
        st.subheader(" Period Grades (0 - 20)")
        G1 = st.slider("First Period Grade (G1)", 0, 20, 10)
        G2 = st.slider("Second Period Grade (G2)", 0, 20, 10)

    if st.button(" Predict Performance Band", type="primary", use_container_width=True):
        input_data = np.array([[
            studytime, failures, 
            1 if activities == "Yes" else 0, 
            1 if internet == "Yes" else 0, 
            absences, goout, G1, G2
        ]])

        prediction = model1.predict(input_data)[0]  # This yields "Excellent", "Good", etc.
        proba = model1.predict_proba(input_data)[0]
        classes = model1.classes_

        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric(label="Predicted Performance Category", value=prediction)
            if prediction in ["Excellent", "Good"]:
                st.success(f"The student is tracking towards a **{prediction}** grade.")
            else:
                st.warning(f"Attention needed. The student is tracking towards an **{prediction}** grade.")

        with res_col2:
            fig = px.bar(x=classes, y=proba * 100, labels={'x': 'Category', 'y': 'Confidence (%)'}, title="Prediction Confidence")
            st.plotly_chart(fig, use_container_width=True)


# TAB 2: EARLY EXAM SCORE PREDICTOR

with tab2:
    st.markdown("### Forecast exact exam score based purely on lifestyle habits.")
    col_a, col_b, col_c = st.columns(3, gap="medium")

    with col_a:
        st.markdown("** Engagement**")
        study_hours = st.slider("Daily Study (hours)", 0.0, 12.0, 3.0, 0.5)
        attendance_pct = st.slider("Attendance Rate (%)", 0.0, 100.0, 80.0, 1.0)
        extracurricular = st.radio("Co-Curricular Activity?", ["Yes", "No"], horizontal=True)

    with col_b:
        st.markdown("** Well-being**")
        mental_health = st.slider("Mental Health Rating (1-10)", 1, 10, 6)
        sleep_hours = st.slider("Nightly Sleep (hours)", 0.0, 12.0, 7.0, 0.5)
        exercise_freq = st.slider("Weekly Exercise Sessions", 0, 7, 3)

    with col_c:
        st.markdown("** Environment**")
        internet_quality = st.selectbox("Internet Quality", ["Poor", "Average", "Good"], index=1)
        social_media_hrs = st.slider("Daily Social Media (hours)", 0.0, 12.0, 2.0, 0.5)
        netflix_hrs = st.slider("Daily Streaming/TV (hours)", 0.0, 12.0, 1.5, 0.5)

    distraction = social_media_hrs + netflix_hrs
    study_vs_dist = study_hours - distraction
    study_mental = study_hours * mental_health

    st.divider()
    ind_col1, ind_col2, ind_col3 = st.columns(3)
    ind_col1.metric(" Total Distraction Time", f"{distraction:.1f} hrs/day")
    ind_col2.metric(" Study vs. Distraction Balance", f"{study_vs_dist:.1f} hrs", 
                    delta="Study Dominant" if study_vs_dist >= 0 else "Distraction Dominant")
    ind_col3.metric(" Focus Quality Score", f"{study_mental:.1f}")
    st.divider()

    if st.button(" Generate Exam Forecast", type="primary", use_container_width=True):
        internet_encoded = encoders2["internet_quality"].transform([internet_quality])[0]
        extracurricular_encoded = encoders2["extracurricular_participation"].transform([extracurricular])[0]

        row = {
            "study_hours_per_day": study_hours,
            "mental_health_rating": mental_health,
            "exercise_frequency": exercise_freq,
            "sleep_hours": sleep_hours,
            "social_media_hours": social_media_hrs,
            "netflix_hours": netflix_hrs,
            "attendance_percentage": attendance_pct,
            "internet_quality": internet_encoded,
            "extracurricular_participation": extracurricular_encoded,
            "distraction_hours": distraction,
            "study_vs_distraction": study_vs_dist,
            "study_mental_interaction": study_mental,
        }

        input_df = pd.DataFrame([row])[feature_cols2]
        predicted_score = max(0.0, min(100.0, float(model2.predict(input_df)[0])))
        
        badge = score_badge(predicted_score)
        color = BADGE_COLORS[badge]

        res_a, res_b = st.columns(2)
        with res_a:
            st.metric(label="Projected Examination Score", value=f"{predicted_score:.1f} / 100")
            st.caption(f"Performance Standing: **{badge}**")
            
            st.markdown("####  Personalised Suggestions:")

            suggestion_given = False
            if study_hours < 2:
                st.info(" Try increasing study sessions closer to 3 hours daily.")
                suggestion_given=True
            if distraction >= 4:
                st.warning(" Reducing digital distraction times could optimize performance.")
                suggestion_given=True
            if sleep_hours < 6.5:
                st.error(" Aim for 7+ hours of sleep to improve daytime cognitive focus.")
                suggestion_given=True
            if study_hours >= 4 and distraction < 3 and sleep_hours >= 7:
                st.success(" Great balance! Keep maintaining this routine.")
                suggestion_given=True
            if not suggestion_given:
                st.success(" Your lifestyle indicators are fairly balanced. Focus on maintaining consistency in your routines for optimal results.")
        with res_b:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=predicted_score,
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': color}},
                title={'text': f"Trajectory Status: {badge}"}
            ))
            fig_gauge.update_layout(height=250, margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

# 4. FOOTER
st.divider()
st.caption(" Indicative projections intended strictly for educational guidance.")

