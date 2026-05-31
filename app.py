import streamlit as st
import pickle
import numpy as np

model = pickle.load(
    open("models/stacking_classifier.pkl", "rb")
)

scaler = pickle.load(
    open("models/scaler.pkl", "rb")
)

encoders = pickle.load(
    open("models/encoders.pkl", "rb")
)

st.title("Heart Disease Prediction")

Age = st.number_input("Age", 20, 100, 50)

Sex = st.selectbox(
    "Sex",
    ["M", "F"]
)

ChestPainType = st.selectbox(
    "Chest Pain Type",
    ["ATA", "NAP", "ASY", "TA"]
)

RestingBP = st.number_input(
    "Resting Blood Pressure",
    50,
    250,
    120
)

Cholesterol = st.number_input(
    "Cholesterol",
    50,
    700,
    200
)

FastingBS = st.selectbox(
    "Fasting Blood Sugar",
    [0, 1]
)

RestingECG = st.selectbox(
    "Resting ECG",
    ["Normal", "LVH", "ST"]
)

MaxHR = st.number_input(
    "Maximum Heart Rate",
    50,
    250,
    150
)

ExerciseAngina = st.selectbox(
    "Exercise Angina",
    ["Y", "N"]
)

Oldpeak = st.number_input(
    "Old Peak",
    0.0,
    10.0,
    1.0
)

ST_Slope = st.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"]
)

if st.button("Predict"):

    Age_Cholesterol = Age * Cholesterol

    BP_Age = RestingBP * Age

    MaxHR_Age_Ratio = MaxHR / (Age + 1)

    Sex = encoders["Sex"].transform([Sex])[0]

    ChestPainType = encoders[
        "ChestPainType"
    ].transform([ChestPainType])[0]

    RestingECG = encoders[
        "RestingECG"
    ].transform([RestingECG])[0]

    ExerciseAngina = encoders[
        "ExerciseAngina"
    ].transform([ExerciseAngina])[0]

    ST_Slope = encoders[
        "ST_Slope"
    ].transform([ST_Slope])[0]

    features = np.array([[
        Age,
        Sex,
        ChestPainType,
        RestingBP,
        Cholesterol,
        FastingBS,
        RestingECG,
        MaxHR,
        ExerciseAngina,
        Oldpeak,
        ST_Slope,
        Age_Cholesterol,
        BP_Age,
        MaxHR_Age_Ratio
    ]])

    features = scaler.transform(features)

    prediction = model.predict(features)

    if prediction[0] == 1:

        st.error(
            "High Risk of Heart Disease"
        )

    else:

        st.success(
            "Low Risk of Heart Disease"
        )