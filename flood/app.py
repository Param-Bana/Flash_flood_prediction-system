
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "sih26192_flash_flood_rf_prototype.joblib"

bundle = joblib.load(MODEL_PATH)
model = bundle["pipeline"]
FEATURES = bundle["features"]

st.set_page_config(
    page_title="SIH26192 Flash-Flood Prototype",
    page_icon="🌧️",
    layout="wide"
)

st.title("SIH26192 — Flash-Flood Prediction Prototype")
st.caption(
    "Random Forest prototype using INDOFLOODS event-scale precipitation and "
    "catchment characteristics. Streamlit inputs simulate field/sensor data."
)

st.warning(
    "Prototype limitation: the current INDOFLOODS target is 'Severe Flood' "
    "versus 'Flood'. This is a development proxy, not a validated flash-flood "
    "occurrence label or an operational warning system."
)

st.subheader("Pre-event precipitation")

cols = st.columns(5)
rain = {}
for i in range(1, 11):
    with cols[(i - 1) % 5]:
        rain[f"T{i}d"] = st.number_input(
            f"T{i}d (mm)",
            min_value=0.0,
            value=0.0,
            step=1.0,
            help=(
                "T1d = precipitation one day before the flood start. "
                "T2d–T10d = cumulative precipitation over the previous "
                f"{i} days."
            )
        )

st.subheader("Catchment characteristics")

c1, c2, c3, c4 = st.columns(4)

with c1:
    stream_order = st.number_input("Stream Order", min_value=1.0, value=3.0, step=1.0)
with c2:
    drainage_area = st.number_input("Drainage Area", min_value=0.0, value=1000.0, step=10.0)
with c3:
    catchment_relief = st.number_input("Catchment Relief", min_value=0.0, value=1000.0, step=10.0)
with c4:
    catchment_length = st.number_input("Catchment Length", min_value=0.0, value=50000.0, step=100.0)

c5, c6, c7 = st.columns(3)

with c5:
    drainage_density = st.number_input("Drainage Density", min_value=0.0, value=0.001, format="%.6f")
with c6:
    ruggedness_number = st.number_input("Ruggedness Number", min_value=0.0, value=1.0, format="%.6f")
with c7:
    annual_precipitation = st.number_input("Annual Precipitation (mm)", min_value=0.0, value=1500.0, step=10.0)

row = {
    **rain,
    "Stream Order": stream_order,
    "Drainage Area": drainage_area,
    "Catchment Relief": catchment_relief,
    "Catchment Length": catchment_length,
    "Drainage Density": drainage_density,
    "Ruggedness Number": ruggedness_number,
    "Annual Precipitation": annual_precipitation,
}

input_df = pd.DataFrame([row], columns=FEATURES)

if st.button("Predict Flood Severity", type="primary", use_container_width=True):
    probability = float(model.predict_proba(input_df)[0, 1])
    prediction = int(model.predict(input_df)[0])

    st.subheader("Prototype result")

    m1, m2 = st.columns(2)
    with m1:
        st.metric("Severe-Flood Probability", f"{probability * 100:.1f}%")
    with m2:
        st.metric(
            "Predicted Class",
            "Severe Flood" if prediction == 1 else "Flood"
        )

    st.progress(min(max(probability, 0.0), 1.0))

    if prediction == 1:
        st.error(
            "Prototype model output: Severe Flood class. "
            "This is not an operational emergency warning."
        )
    else:
        st.info(
            "Prototype model output: Flood class. "
            "This is not an operational safety assessment."
        )

st.divider()
st.caption(
    "SIH26192 prototype only."
)
