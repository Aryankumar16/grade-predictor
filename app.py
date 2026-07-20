import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime


# 1. PAGE CONFIGURATION 
st.set_page_config(
    page_title="Student Performance Prediction System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS
st.markdown("""
<style>

h1{
    color: #2C3E50 !important;
    font-weight: 700;
}


h2{
    color: #EDEADE !important;
}


h3{
    color: #EDEADE !important;
}

</style>
""", unsafe_allow_html=True)

# Formal color mappings
COLOR_SUCCESS = "#2E8B57"  # Sea Green
COLOR_WARNING = "#DAA520"  # Goldenrod
COLOR_DANGER = "#CD5C5C"   # Indian Red
COLOR_PRIMARY = "#4682B4"  # Steel Blue
COLOR_TEAL = "#16A085"     # Sea Teal for secondary metrics

# 2. CORE LOGIC & DEFINITIONS
def classify_grade(score):
    if score >= 16: return "Excellent"
    if score >= 12: return "Good"
    if score >= 9: return "Average"
    return "Poor"

def get_performance_standing(score: float) -> str:
    if score >= 85: return "Distinction"
    if score >= 70: return "Merit"
    if score >= 55: return "Pass"
    return "Below Pass"

@st.cache_resource
def load_models():
    """Loads pre-trained models and encoders."""
    model1 = joblib.load("dt.pkl")
    model2 = joblib.load("xgb.pkl")
    encoders2 = joblib.load("encoders.pkl")
    features2 = joblib.load("features.pkl")
    return model1, model2, encoders2, features2

try:
    model1, model2, encoders2, feature_cols2 = load_models()
except Exception as e:
    st.error(f"System Error: Model assets could not be located or loaded. ({e})")
    st.stop()


# Session state — in-session prediction history


if "history_a" not in st.session_state:
    st.session_state.history_a = []
if "history_b" not in st.session_state:
    st.session_state.history_b = []


# 3. HEADER

st.markdown("<h1 style='text-align: center;'>Student Performance Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #7F8C8D; font-weight: 300;'>Predictive modeling for student outcomes and behavioral impact.</p>", unsafe_allow_html=True)
st.divider()

tab1, tab2 = st.tabs(["Historical Grade Classification", "Behavioral Score Forecasting"])


# 4. TAB 1: HISTORICAL GRADE CLASSIFICATION

t1_col_input, t1_col_results = tab1.columns([1, 1], gap="large")

t1_col_input.markdown("### Input Parameters")
studytime = t1_col_input.selectbox("Weekly Study Duration", [1, 2, 3, 4], format_func=lambda x: ["Under 2 hours", "2 to 5 hours", "5 to 10 hours", "Over 10 hours"][x-1])
failures = t1_col_input.slider("Historical Class Failures", 0, 3, 0)
absences = t1_col_input.number_input("Recorded Absences", min_value=0, max_value=100, value=5)
goout = t1_col_input.slider("Social Frequency Index (1-5)", 1, 5, 3)
activities = t1_col_input.selectbox("Extracurricular Participation", ["Yes", "No"], key="extra_t1")
internet = t1_col_input.selectbox("Home Internet Access", ["Yes", "No"], key="net_t1")

t1_col_input.markdown("#### Period Assessments")
G1 = t1_col_input.number_input("Term 1 Grade (0-20)", min_value=0, max_value=20, value=10)
G2 = t1_col_input.number_input("Term 2 Grade (0-20)", min_value=0, max_value=20, value=10)

analyze_t1 = t1_col_input.button("Run Classification Analysis", type="primary", use_container_width=True)

t1_col_results.markdown("### Analytical Output")
if analyze_t1:
    input_data = np.array([[
        studytime, failures, 
        1 if activities == "Yes" else 0, 
        1 if internet == "Yes" else 0, 
        absences, goout, G1, G2
    ]])

    prediction = model1.predict(input_data)[0]
    probabilities = model1.predict_proba(input_data)[0]
    classes = model1.classes_

    t1_col_results.metric(label="Projected Final Category", value=prediction)
    
    if prediction in ["Excellent", "Good"]:
        t1_col_results.success(f"Status: Favorable. The trajectory indicates a final grade of {prediction}.")
    else:
        t1_col_results.warning(f"Status: At Risk. The trajectory indicates a final grade of {prediction}.")

    # --- Right Column: Graph 1 (Confidence Bar Chart) ---
    fig1 = px.bar(
        x=classes, y=probabilities * 100, 
        labels={'x': 'Classification', 'y': 'Probability (%)'}, 
        title="Model Confidence Distribution",
        color_discrete_sequence=[COLOR_PRIMARY]
    )
    fig1.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Montserrat")
    )
    t1_col_results.plotly_chart(fig1, use_container_width=True)
    
    # --- Right Column: Graph 2 (Academic Trajectory Plot) ---
    proxy_scores = {"Excellent": 18, "Good": 14, "Average": 10.5, "Poor": 6}
    final_proxy = proxy_scores.get(prediction, 10)

    fig2 = go.Figure()
    
    # Background Context Bands
    fig2.add_hrect(y0=16, y1=20, fillcolor=COLOR_SUCCESS, opacity=0.1, line_width=0, layer="below")
    fig2.add_hrect(y0=12, y1=16, fillcolor=COLOR_PRIMARY, opacity=0.1, line_width=0, layer="below")
    fig2.add_hrect(y0=9, y1=12, fillcolor=COLOR_WARNING, opacity=0.1, line_width=0, layer="below")
    fig2.add_hrect(y0=0, y1=9, fillcolor=COLOR_DANGER, opacity=0.1, line_width=0, layer="below")

    # Actual historical grades
    fig2.add_trace(go.Scatter(
        x=["Term 1", "Term 2"], y=[G1, G2],
        mode="lines+markers", name="Actual",
        line=dict(color="#2C3E50", width=3),
        marker=dict(size=10, color="#2C3E50")
    ))

    # Projected future grade trajectory
    fig2.add_trace(go.Scatter(
        x=["Term 2", "Final (Projected)"], y=[G2, final_proxy],
        mode="lines+markers", name="Projected",
        line=dict(color="#2C3E50", width=3, dash="dash"),
        marker=dict(size=12, symbol="star", color="#2C3E50")
    ))

    fig2.update_layout(
        title="Academic Trajectory Projection",
        yaxis_title="Grade (0-20 scale)",
        yaxis=dict(range=[0, 20], dtick=4),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Montserrat"),
        showlegend=False,
        margin=dict(t=40, b=20, l=20, r=20)
    )
    t1_col_results.plotly_chart(fig2, use_container_width=True)

    # --- Left Column: Graph 3 (Behavioral Feature Importance) ---
    # This plots under the Run button to balance the layout
    try:
        profile_features = [
        "Study Habit Strength", 
        "Failure Avoidance", 
        "Extracurriculars", 
        "Connectivity", 
        "Attendance Profile", 
        "Social Activity"]

        profile_vals = [
        (studytime / 4) * 100,                 # 1-4 scale converted to percentage
        100 - (failures / 3 * 100),            # 0-3 scale (Inverted: 0 failures = 100 score)
        100 if activities == "Yes" else 0,
        100 if internet == "Yes" else 0,
        max(0, 100 - absences),                # 0 absences = 100 score
        (goout / 5) * 100   ]                   # 1-5 scale converted to percentage

        fig3 = px.bar(
        x=profile_vals, 
        y=profile_features, 
        orientation='h',
        title="Normalized Behavioral Profile",
        labels={'x': 'Index Score (0-100)', 'y': ''},
        color_discrete_sequence=[COLOR_TEAL]
    )
        fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Montserrat"),
        xaxis=dict(range=[0, 100]),
        margin=dict(t=40, b=20, l=20, r=20)
    )
        fig3.update_yaxes(autorange="reversed")
        t1_col_input.plotly_chart(fig3, use_container_width=True)

        
    except AttributeError:
        pass

    # Session history
    record_a = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "studytime": studytime,
        "failures": failures,
        "activities": activities,
        "internet": internet,
        "absences": absences,
        "goout": goout,
        "G1": G1,
        "G2": G2,
        "prediction": prediction,
    }
    st.session_state.history_a.append(record_a)

else:
    t1_col_results.info("Awaiting input. Please run the analysis to view projections.")

# History tracking and CSV export
if st.session_state.history_a:
    with tab1.expander(f"Session history ({len(st.session_state.history_a)} predictions)"):
        df_a = pd.DataFrame(st.session_state.history_a)
        st.dataframe(df_a, use_container_width=True)
        st.download_button(
            "Download history as CSV",
            data=df_a.to_csv(index=False).encode('utf-8'),
            file_name=f"classification_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

# 5. TAB 2: BEHAVIORAL SCORE FORECASTING

tab2.markdown("### Behavioral Indicators")
t2_col_a, t2_col_b, t2_col_c = tab2.columns(3, gap="medium")


t2_col_a.markdown("#### Academic Engagement")
study_hours = t2_col_a.number_input("Daily Study (Hours)", min_value=0.0, max_value=12.0, value=3.0, step=0.5)
attendance_pct = t2_col_a.slider("Attendance Rate (%)", 0.0, 100.0, 85.0, 1.0)
extracurricular = t2_col_a.selectbox("Co-Curricular Activity", ["Yes", "No"], key="extra_t2")


t2_col_b.markdown("#### Health & Well-being")
mental_health = t2_col_b.slider("Well-being Index (1-10)", 1, 10, 7)
sleep_hours = t2_col_b.number_input("Nightly Rest (Hours)", min_value=0.0, max_value=12.0, value=7.5, step=0.5)
exercise_freq = t2_col_b.slider("Weekly Exercise Sessions", 0, 7, 3)


t2_col_c.markdown("#### Digital Environment")
internet_quality = t2_col_c.selectbox("Connectivity Standard", ["Poor", "Average", "Good"], index=1, key="net_t2")
social_media_hrs = t2_col_c.number_input("Social Media (Hours)", min_value=0.0, max_value=12.0, value=1.5, step=0.5)
netflix_hrs = t2_col_c.number_input("Streaming Media (Hours)", min_value=0.0, max_value=12.0, value=1.0, step=0.5)


distraction = social_media_hrs + netflix_hrs
study_vs_dist = study_hours - distraction
study_mental = study_hours * mental_health

# Baseline Assumptions for Deltas
DISTRACTION_BASELINE = 3.0  # Assumed healthy max baseline
FOCUS_BASELINE = 15.0       # Assumed baseline (3 hrs study * 5 well-being)

tab2.divider()
tab2.markdown("### Derived Behavioral Metrics")
met_col1, met_col2, met_col3 = tab2.columns(3)

met_col1.metric(
    label="Total Distraction Load", 
    value=f"{distraction:.1f} hrs", 
    delta=f"{distraction - DISTRACTION_BASELINE:.1f} hrs vs baseline",
    delta_color="inverse"
)

met_col2.metric(
    label="Study-to-Distraction Balance", 
    value=f"{study_vs_dist:.1f} hrs", 
    delta=f"{study_vs_dist:.1f} net positive hours", 
    delta_color="normal"
)

met_col3.metric(
    label="Cognitive Focus Index", 
    value=f"{study_mental:.1f}",
    delta=f"{study_mental - FOCUS_BASELINE:.1f} pts vs baseline",
    delta_color="normal"
)
tab2.divider()

analyze_t2 = tab2.button("Run Forecasting Engine", type="primary", use_container_width=True)

if analyze_t2:
    if "internet_quality" in encoders2:
        internet_encoded = encoders2["internet_quality"].transform([internet_quality])[0]
    else:
        internet_encoded = {"Poor": 0, "Average": 1, "Good": 2}.get(internet_quality, 1)

    if "extracurricular_participation" in encoders2:
        extracurricular_encoded = encoders2["extracurricular_participation"].transform([extracurricular])[0]
    else:
        extracurricular_encoded = {"No": 0, "Yes": 1}.get(extracurricular, 0)

    
    row_data = {
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

    input_df = pd.DataFrame([row_data])[feature_cols2]
    predicted_score = max(0.0, min(100.0, float(model2.predict(input_df)[0])))
    standing = get_performance_standing(predicted_score)

    res_col1, res_col2 = tab2.columns([1, 1], gap="large")
    
    # Left Results: Formal Executive Summary
    res_col1.markdown("### Executive Summary")
    res_col1.metric(label="Forecasted Examination Score", value=f"{predicted_score:.1f} / 100")
    res_col1.caption(f"Academic Standing: **{standing}**")
    
    suggestion_given = False
    if study_hours < 2: 
        res_col1.info("Observation: Study duration is below optimal thresholds. Increasing to 3+ hours daily is recommended.")
        suggestion_given = True
    if distraction >= 4: 
        res_col1.warning("Risk Factor: High digital distraction load identified. Reduction is advised to optimize focus.")
        suggestion_given = True
    if sleep_hours < 6.5: 
        res_col1.error("Risk Factor: Inadequate rest may negatively impact cognitive function and retention.")
        suggestion_given = True
    if attendance_pct < 75:
        res_col1.warning("Risk Factor: Attendance rates fall below standard academic compliance.")
        suggestion_given = True

    if not suggestion_given:
        res_col1.success("Evaluation: Core behavioral metrics are stable and well-balanced. Maintain current routines.")

    
    gauge_color = COLOR_SUCCESS if predicted_score >= 55 else COLOR_DANGER
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=predicted_score,
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': gauge_color}},
        title={'text': f"Trajectory Status", 'font': {'family': 'Lora'}}
    ))
    fig_gauge.update_layout(
        height=300, 
        margin=dict(t=50, b=20, l=20, r=20), 
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Montserrat")
    )
    res_col2.plotly_chart(fig_gauge, use_container_width=True)

    # LIFESTYLE & BEHAVIORAL VISUALIZATIONS

    tab2.divider()
    tab2.markdown("### Lifestyle & Behavioral Insights")
    
    chart_col1, chart_col2 = tab2.columns([1, 1], gap="large")
    
    # --- Left Chart: 24-Hour Time Allocation (Donut) ---
    unallocated_time = max(0, 24.0 - (sleep_hours + study_hours + social_media_hrs + netflix_hrs))
    time_labels = ["Sleep", "Study", "Social Media", "Streaming", "Other/Unallocated"]
    time_values = [sleep_hours, study_hours, social_media_hrs, netflix_hrs, unallocated_time]
    
    fig_donut = go.Figure(data=[go.Pie(
        labels=time_labels, 
        values=time_values, 
        hole=0.6,
        marker_colors=["#34495E", COLOR_PRIMARY, COLOR_DANGER, COLOR_WARNING, "#BDC3C7"]
    )])
    fig_donut.update_layout(
        title="24-Hour Time Allocation",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Montserrat"),
        margin=dict(t=40, b=20, l=20, r=20)
    )
    chart_col1.plotly_chart(fig_donut, use_container_width=True)
    
    # --- Right Chart: "High Achiever" Comparison (Radar) ---
    # Normalize inputs to a 0-100 scale for radar visualization
    norm_attendance = attendance_pct
    norm_mental = mental_health * 10
    norm_sleep = min(100, (sleep_hours / 8.0) * 100)  # Assuming 8 hours is optimal
    norm_study = min(100, (study_hours / 4.0) * 100)  # Assuming 4 hours is optimal
    norm_discipline = max(0, 100 - (distraction * 10)) # Scales down as distraction goes up
    
    categories = ['Attendance', 'Mental Health', 'Sleep Adequacy', 'Study Volume', 'Digital Discipline']
    student_profile = [norm_attendance, norm_mental, norm_sleep, norm_study, norm_discipline]
    ideal_profile = [95, 85, 100, 85, 80] # The hypothetical "High Achiever" baseline
    
    fig_radar = go.Figure()
    
    # Baseline Trace
    fig_radar.add_trace(go.Scatterpolar(
        r=ideal_profile + [ideal_profile[0]], # Close the loop
        theta=categories + [categories[0]],
        fill='toself',
        name='High Achiever Baseline',
        line=dict(color="#95A5A6", dash="dot"),
        fillcolor="rgba(149, 165, 166, 0.2)"
    ))
    
    # Student Trace
    fig_radar.add_trace(go.Scatterpolar(
        r=student_profile + [student_profile[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Your Profile',
        line=dict(color=COLOR_PRIMARY),
        fillcolor="rgba(70, 130, 180, 0.4)"
    ))
    
    fig_radar.update_layout(
        title="Performance Profile vs. Baseline",
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Montserrat"),
        margin=dict(t=40, b=20, l=20, r=20),
        legend=dict(orientation="h", y=-0.2)
    )
    chart_col2.plotly_chart(fig_radar, use_container_width=True)

    # Session history
    record_b = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "study_hours_per_day": study_hours,
        "mental_health_rating": mental_health,
        "exercise_frequency": exercise_freq,
        "sleep_hours": sleep_hours,
        "social_media_hours": social_media_hrs,
        "netflix_hours": netflix_hrs,
        "attendance_percentage": attendance_pct,
        "internet_quality": internet_quality,
        "extracurricular_participation": extracurricular,
        "predicted_score": round(predicted_score, 2),
        "band": standing,
    }
    st.session_state.history_b.append(record_b)
if st.session_state.history_b:
    with tab2.expander(f"Session history ({len(st.session_state.history_b)} predictions)"):
        df_b = pd.DataFrame(st.session_state.history_b)
        st.dataframe(df_b, use_container_width=True)
        st.download_button(
            "Download history as CSV",
            data=df_b.to_csv(index=False).encode('utf-8'),
            file_name=f"score_prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )


# 6. FOOTER

st.divider()
st.markdown("<p style='text-align: center; color: #95A5A6; font-size: 0.85em;'>Proprietary Academic Forecasting Model. For advisory purposes only.</p>", unsafe_allow_html=True)
