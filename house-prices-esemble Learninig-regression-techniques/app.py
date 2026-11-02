
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model
bundle = joblib.load("house_price_model.pkl")

model = bundle["model"]
numeric_cols = bundle["numeric_cols"]
categorical_cols = bundle["categorical_cols"]
numeric_means = bundle["numeric_means"]
categorical_options = bundle["categorical_options"]
feature_columns = bundle["feature_columns"]

# Page configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 House Price Prediction")
st.write("Enter the house details to estimate its selling price.")

st.divider()

# Input form
with st.form("house_prediction_form"):

    input_data = {}

    st.subheader("Numerical Features")

    num_col1, num_col2 = st.columns(2)

    for i, col in enumerate(numeric_cols):

        with (num_col1 if i % 2 == 0 else num_col2):

            input_data[col] = st.number_input(
                label=col.replace("_", " ").title(),
                value=float(numeric_means[col])
            )

    st.divider()

    st.subheader("Categorical Features")

    cat_col1, cat_col2 = st.columns(2)

    for i, col in enumerate(categorical_cols):

        options = categorical_options[col]

        with (cat_col1 if i % 2 == 0 else cat_col2):

            if options:
                input_data[col] = st.selectbox(
                    label=col.replace("_", " ").title(),
                    options=options
                )
            else:
                input_data[col] = ""

    submit = st.form_submit_button(
        "Predict House Price",
        use_container_width=True
    )

# Prediction
if submit:

    input_df = pd.DataFrame([input_data])

    # One-hot encoding (same as training)
    input_df = pd.get_dummies(
        input_df,
        columns=categorical_cols,
        dtype=int
    )

    # Match the exact training feature columns
    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Predict
    prediction = model.predict(input_df)[0]

    st.divider()

    st.success("Prediction completed!")

    st.metric(
        label="Estimated House Selling Price",
        value=f"${prediction:,.2f}"
    )

    st.caption(
        "This is an estimated price based on the trained XGBoost model."
    )
