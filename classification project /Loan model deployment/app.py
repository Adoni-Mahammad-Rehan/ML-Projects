import streamlit as st
import pandas as pd
import joblib

# Keep these files in the same folder as app.py.
MODEL_PATH = "Svm_Loan_model.pkl"
SCALER_PATH = "Scaler_Loan_model.pkl"
COLUMNS_PATH = "Columns_Loan_model.pkl"

st.set_page_config(page_title="Loan Status Predictor", page_icon="🏦", layout="wide")

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    model_columns = joblib.load(COLUMNS_PATH)
    return model, scaler, model_columns

st.title("🏦 Loan Status Predictor")
st.caption("Enter applicant and loan details to estimate the model's predicted loan status.")
st.info(
    "This is an educational machine-learning demo, not a lending decision or financial advice. "
    "The prediction may be wrong and should not be used as the sole basis for approving or rejecting a loan."
)

try:
    model, scaler, model_columns = load_artifacts()
except FileNotFoundError as exc:
    st.error(
        f"Could not find {exc.filename}. Put Svm_Loan_model.pkl, "
        "Scaler_Loan_model.pkl, and Columns_Loan_model.pkl in the same folder as app.py."
    )
    st.stop()
except Exception as exc:
    st.error(f"Could not load the saved model files: {exc}")
    st.stop()

with st.form("loan_form"):
    st.subheader("Applicant details")
    left, right = st.columns(2)

    with left:
        person_age = st.number_input("Age (years)", min_value=18, max_value=100, value=25, step=1)
        person_income = st.number_input("Annual income", min_value=1.0, value=71948.0, step=1000.0)
        person_gender = st.selectbox("Gender", ["female", "male"])
        person_education = st.selectbox(
            "Education",
            ["High School", "Associate", "Bachelor", "Master", "Doctorate"]
        )
        person_home_ownership = st.selectbox(
            "Home ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"]
        )

    with right:
        cb_person_cred_hist_length = st.number_input(
            "Credit history length (years)", min_value=0, max_value=60, value=3, step=1
        )
        cb_person_default_on_file = st.selectbox(
            "Previous default on file?", ["N", "Y"]
        )
        loan_intent = st.selectbox(
            "Loan purpose",
            [
                "PERSONAL", "EDUCATION", "MEDICAL", "VENTURE",
                "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"
            ]
        )
        loan_amnt = st.number_input("Loan amount", min_value=0.0, value=35000.0, step=500.0)
        loan_int_rate = st.number_input(
            "Loan interest rate (%)", min_value=0.0, max_value=100.0, value=16.02, step=0.1
        )

    loan_percent_income = loan_amnt / person_income if person_income > 0 else 0.0
    st.metric("Loan amount / annual income", f"{loan_percent_income:.2f}")
    st.caption("This value is calculated automatically as loan amount ÷ annual income.")

    submitted = st.form_submit_button("Predict loan status", type="primary", use_container_width=True)

if submitted:
    # Raw fields must use the same original names as the notebook's dataset.
    raw_input = pd.DataFrame([{
        "person_age": person_age,
        "person_income": person_income,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
        "person_gender": person_gender,
        "person_education": person_education,
        "person_home_ownership": person_home_ownership,
        "loan_intent": loan_intent,
        "cb_person_default_on_file": cb_person_default_on_file,
    }])

    # Match the notebook's pd.get_dummies approach and restore the exact training columns.
    categorical_columns = [
        "person_gender",
        "person_education",
        "person_home_ownership",
        "loan_intent",
        "cb_person_default_on_file",
    ]
    encoded_input = pd.get_dummies(raw_input, columns=categorical_columns, dtype=int)
    encoded_input = encoded_input.reindex(columns=model_columns, fill_value=0)

    try:
        # Use transform, not fit_transform: the saved training scaler must be reused.
        scaled_input = scaler.transform(encoded_input)
        prediction = model.predict(scaled_input)[0]

        st.subheader("Prediction result")
        if int(prediction) == 1:
            st.warning("Loan status = Accepted")
        else:
            st.success("Loan status = Rejected")

    

        if hasattr(model, "decision_function"):
            st.caption("The SVM decision score is not a calibrated probability.")
            try:
                score = float(model.decision_function(scaled_input)[0])
                st.write(f"SVM decision score: `{score:.4f}`")
            except Exception:
                pass

        with st.expander("Show encoded model input"):
            st.dataframe(encoded_input, use_container_width=True)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.write(
            "Check that the model, scaler, and saved columns were all created from the same "
            "training run, and that the input feature names match the notebook."
        )
