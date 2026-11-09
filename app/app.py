import joblib
import pandas as pd
import streamlit as st#/home/sonu-nitu/Machine_defect_m/Machine defect_model/app.py
from pathlib import Path


# ----------------------------
# Page configuration
# ----------------------------
st.set_page_config(
    page_title="Machine Failure Prediction",
    page_icon="⚙️",
    layout="centered",
)

st.title("⚙️ Machine Failure Prediction")
st.write(
    "Enter the machine sensor readings below to estimate the likelihood "
    "of a machine failure."
)

# ----------------------------
# Load the saved model artifact
# ----------------------------
MODEL_PATH = Path(__file__).resolve().parent.parent / "Machine_defect_model.joblib"



@st.cache_resource
def load_artifact(path):
    return joblib.load(path)

try:
    artifact = load_artifact(MODEL_PATH)
    model = artifact["pipeline"]
    threshold = float(artifact["threshold"])
    features = artifact["features"]
except FileNotFoundError:
    st.error(
        "Model file not found. Make sure "
        "'Machine_defect_model.joblib' exists in the project directory."
    )
    st.stop()
except Exception as exc:
    st.error(f"Could not load the model artifact: {exc}")
    st.stop()

# ----------------------------
# Sensor input form
# ----------------------------
st.subheader("Sensor readings")

with st.form("sensor_input_form"):
    col1, col2 = st.columns(2)

    with col1:
        temperature = st.number_input(
            "Temperature",
            value=75.5,
            help="Enter the machine temperature reading.",
        )
        pressure = st.number_input(
            "Pressure",
            value=0.90,
            format="%.4f",
            help="Enter the machine pressure reading.",
        )
        vibration = st.number_input(
            "Vibration Level",
            value=0.80,
            format="%.4f",
            help="Enter the machine vibration reading.",
        )

    with col2:
        humidity = st.number_input(
            "Humidity",
            value=0.80,
            format="%.4f",
            help="Enter the machine humidity reading.",
        )
        power = st.number_input(
            "Power Consumption",
            value=0.125,
            format="%.4f",
            help="Enter the machine power consumption reading.",
        )

    submitted = st.form_submit_button(
        "Predict machine status",
        type="primary",
        use_container_width=True,
    )

# ----------------------------
# Prediction
# ----------------------------
if submitted:
    input_values = {
        "Temperature": temperature,
        "Pressure": pressure,
        "Vibration_Level": vibration,
        "Humidity": humidity,
        "Power_Consumption": power,
    }

    # Keep exactly the feature names and order used during training.
    input_df = pd.DataFrame([input_values])[features]

    try:
        failure_probability = float(model.predict_proba(input_df)[0, 1])
        prediction = int(failure_probability >= threshold)

        st.divider()
        st.subheader("Prediction result")
        st.metric(
            "Estimated failure probability",
            f"{failure_probability:.2%}",
        )
        st.caption(f"Decision threshold: {threshold:.2f}")

        if prediction == 1:
            st.error("⚠️ Prediction: Failure")
            st.write("The model's estimated failure probability is at or above the decision threshold.")
        else:
            st.success("✅ Prediction: No failure")
            st.write("The model's estimated failure probability is below the decision threshold.")

        with st.expander("View submitted sensor values"):
            st.dataframe(input_df, use_container_width=True)

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

st.divider()
st.caption(
    "Prototype only: this result is a model prediction, not a guarantee of machine condition. "
    "Validate the model on representative real-world data before using it for maintenance decisions."
)
