import streamlit as st
import pandas as pd
import pickle

# Load trained models
with open("svm_mental_health.pkl", "rb") as f:
    svm_model = pickle.load(f)

with open("rf_mental_health.pkl", "rb") as f:
    rf_model = pickle.load(f)

# Sidebar inputs
st.sidebar.header("User Input Features")

age = st.sidebar.slider("Age", 18, 65, 30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Non-binary"])
education = st.sidebar.selectbox("Education Level", ["High School", "Associate's", "Bachelor's", "Master's/PhD"])
profession = st.sidebar.selectbox("Profession", ["Student", "Engineer", "Teacher", "Doctor", "Finance", "Artist", "Tech", "Unemployed", "Manager", "Healthcare"])
sleep_hours = st.sidebar.slider("Sleep Hours per day", 3.0, 10.0, 7.0, 0.1)
physical_activity = st.sidebar.slider("Physical Activity min/wk", 0, 600, 150)
social_interaction = st.sidebar.selectbox("Social Interactions", ["Low", "Medium", "High"])
screen_time = st.sidebar.slider("Screen Time hrs/day", 0.5, 12.0, 5.0, 0.1)
work_hours = st.sidebar.slider("Work Hours hrs/wk", 0, 80, 40)

# Prepare input dataframe
input_df = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "Education_Level": [education],
    "Profession": [profession],
    "Sleep_Hours": [sleep_hours],
    "Physical_Activity_min/wk": [physical_activity],
    "Social_Interactions": [social_interaction],
    "Screen_Time_hrs/day": [screen_time],
    "Work_Hours_hrs/wk": [work_hours]
})

model_choice = st.sidebar.selectbox("Select Model", ["SVM", "Random forest"])

if st.button("Predict Mental Health Status"):
    if model_choice == "SVM":
        pred = svm_model.predict(input_df)[0]
    else:
        pred = rf_model.predict(input_df)[0]

    # Map back to string
    label_map = {0: "Bad", 1: "Normal", 2: "Good"}
    pred_label = label_map[pred]

    st.subheader("Predicted Mental Health Status:")
    st.success(pred_label)
