import streamlit as st
import pandas as pd
import joblib

# Load the trained model
saved_model = joblib.load("titanic_model.pkl")

# Support both saved model formats
if isinstance(saved_model, dict):
    model = saved_model["model"]
    feature_columns = saved_model.get(
        "feature_columns",
        ["pclass", "sex", "age", "fare"]
    )
else:
    model = saved_model
    feature_columns = ["pclass", "sex", "age", "fare"]

# Page settings
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)

st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger details to predict survival.")

# Passenger inputs
pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.number_input(
    "Age",
    min_value=0.1,
    max_value=100.0,
    value=25.0
)

fare = st.number_input(
    "Ticket Fare",
    min_value=0.0,
    max_value=600.0,
    value=30.0
)

# Convert gender into the same encoding used during training
sex = 1 if gender == "Female" else 0

# Prediction button
if st.button("Predict Survival"):

    input_data = pd.DataFrame(
        [[pclass, sex, age, fare]],
        columns=["pclass", "sex", "age", "fare"]
    )

    input_data = input_data[feature_columns]

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("🎉 Passenger is predicted to Survive!")
    else:
        st.error("Passenger is predicted Not to Survive.")

    # Show prediction probabilities
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        classes = list(model.classes_)

        survived_index = classes.index(1)
        not_survived_index = classes.index(0)

        st.write(
            f"Survival Probability: "
            f"{probabilities[survived_index] * 100:.2f}%"
        )

        st.write(
            f"Not Survived Probability: "
            f"{probabilities[not_survived_index] * 100:.2f}%"
        )