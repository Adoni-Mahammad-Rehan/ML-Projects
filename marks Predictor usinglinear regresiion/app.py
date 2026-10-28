import streamlit as st
import numpy as np
import joblib
from pathlib import Path

# Keep these files in the same folder as app.py
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_score_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"

FEATURES = [
    "hours_studied",
    "sleep_hours",
    "attendance_percent",
    "previous_scores",
]

st.set_page_config(
    page_title="Student Exam Score Predictor",
    page_icon="🎓",
    layout="centered",
)

# Simple styling
st.markdown(
    """
    <style>
    .block-container {max-width: 850px; padding-top: 2rem; padding-bottom: 3rem;}
    .hero {
        padding: 1.5rem 1.6rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #102a43 0%, #243b53 55%, #334e68 100%);
        color: white;
        margin-bottom: 1.3rem;
    }
    .hero h1 {color: white; margin-bottom: .35rem; font-size: 2.1rem;}
    .hero p {color: #d9e2ec; margin-bottom: 0; font-size: 1rem;}
    .result {
        padding: 1.25rem;
        border: 1px solid rgba(128,128,128,.28);
        border-radius: 16px;
        text-align: center;
        margin-top: .7rem;
    }
    .score {font-size: 2.8rem; font-weight: 750; line-height: 1.2;}
    .muted {color: #829ab1; font-size: .9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🎓 Student Exam Score Predictor</h1>
        <p>Enter the student's study habits and previous performance to estimate an exam score.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH.name}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH.name}")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

try:
    model, scaler = load_artifacts()
except Exception as exc:
    st.error(
        "I couldn't load the model files. Make sure `student_score_model.pkl` "
        "and `scaler.pkl` are in the same folder as `app.py`."
    )
    st.code(str(exc))
    st.stop()

st.subheader("Student details")
st.caption("Use the same units and feature definitions as the dataset used to train the model.")

with st.form("score_prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        hours_studied = st.number_input(
            "Hours studied",
            min_value=0.0,
            max_value=24.0,
            value=5.0,
            step=0.5,
            help="Number of hours spent studying.",
        )
        sleep_hours = st.number_input(
            "Sleep hours per day",
            min_value=0.0,
            max_value=24.0,
            value=7.0,
            step=0.5,
        )

    with col2:
        attendance_percent = st.number_input(
            "Attendance (%)",
            min_value=0.0,
            max_value=100.0,
            value=85.0,
            step=1.0,
        )
        previous_scores = st.number_input(
            "Previous score",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=1.0,
            help="Previous exam score, using the same scale as your dataset.",
        )

    submitted = st.form_submit_button(
        "Predict exam score",
        type="primary",
        use_container_width=True,
    )

if submitted:
    # The scaler was fitted on these four columns in this exact order.
    input_data = np.array(
        [[hours_studied, sleep_hours, attendance_percent, previous_scores]],
        dtype=float,
    )
    scaled_input = scaler.transform(input_data)
    prediction = float(np.asarray(model.predict(scaled_input)).reshape(-1)[0])

    # Display the model's raw prediction; do not silently clip it to 0–100.
    st.markdown('<div class="result">', unsafe_allow_html=True)
    st.markdown("### Estimated exam score")
    st.markdown(f'<div class="score">{prediction:.2f}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="muted">Prediction from your trained Linear Regression model</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if prediction < 0 or prediction > 100:
        st.warning(
            "The model returned a score outside 0–100. Linear regression can produce "
            "out-of-range predictions; this is a model limitation, not a UI error."
        )

    st.info(
        "This is an estimate based on patterns in the training data, not a guaranteed result."
    )

with st.expander("How this app works"):
    st.write(
        "The app collects four inputs, applies the saved StandardScaler, and sends "
        "the scaled values to the saved Linear Regression model. The scaler and model "
        "must be the same fitted artifacts used during training."
    )
