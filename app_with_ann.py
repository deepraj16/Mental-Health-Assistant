import streamlit as st
import pandas as pd
import pickle
import numpy as np
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.schema import HumanMessage, AIMessage

# ------------------------------
# SETUP FUNCTIONS
# ------------------------------
@st.cache_resource
def setup_llm():
    return ChatMistralAI(
        api_key="lHcwga2vJ6yyjV470WdMIFn5hRgtMbcc",
        model="mistral-large-latest",
        temperature=0.7
    )

@st.cache_resource
def load_models():
    """Load ML models"""
    svm_model, rf_model = None, None

    try:
        with open("svm_mental_health.pkl", "rb") as f:
            svm_model = pickle.load(f)
    except Exception as e:
        st.warning(f"SVM model could not be loaded: {str(e)[:100]}")

    try:
        with open("rf_mental_health.pkl", "rb") as f:
            rf_model = pickle.load(f)
    except Exception as e:
        st.warning(f"Random Forest model could not be loaded: {str(e)[:100]}")

    return svm_model, rf_model


# ------------------------------
# STREAMLIT SETUP
# ------------------------------
st.set_page_config(page_title="Mental Health Assistant", layout="wide")
st.title("Mental Health Detection and Support Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "show_chatbot" not in st.session_state:
    st.session_state.show_chatbot = False

st.sidebar.header("User Input Features")

age = st.sidebar.slider("Age", 18, 65, 30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Non-binary"])
education = st.sidebar.selectbox("Education Level", ["High School", "Associate's", "Bachelor's", "Master's/PhD"])
profession = st.sidebar.selectbox(
    "Profession",
    ["Student", "Engineer", "Teacher", "Doctor", "Finance", "Artist", "Tech", "Unemployed", "Manager", "Healthcare"]
)
sleep_hours = st.sidebar.slider("Sleep Hours per day", 3.0, 10.0, 7.0, 0.1)
physical_activity = st.sidebar.slider("Physical Activity min/wk", 0, 600, 150)
social_interaction = st.sidebar.selectbox("Social Interactions", ["Low", "Medium", "High"])
screen_time = st.sidebar.slider("Screen Time hrs/day", 0.5, 12.0, 5.0, 0.1)
work_hours = st.sidebar.slider("Work Hours hrs/wk", 0, 80, 40)

# Load available models
svm_model, rf_model = load_models()
available_models = []
if svm_model is not None:
    available_models.append("SVM")
if rf_model is not None:
    available_models.append("Random Forest")

if not available_models:
    st.error("No models are available. Please check your model files.")
    st.stop()

model_choice = st.sidebar.selectbox("Select Model", available_models)

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def build_input_df():
    return pd.DataFrame({
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

def predict_mental_health(input_df):
    model = svm_model if model_choice == "SVM" else rf_model
    if model is None:
        st.error(f"{model_choice} model not loaded")
        return None, None

    pred = model.predict(input_df)[0]
    label_map = {0: "Bad", 1: "Normal", 2: "Good"}
    pred_label = label_map[pred]

    confidence = None
    try:
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(input_df)[0]
            confidence = float(np.max(probs) * 100)
    except:
        pass

    return pred_label, confidence

# ------------------------------
# MAIN APP
# ------------------------------
if not st.session_state.show_chatbot:
    st.header("Mental Health Assessment")
    st.write("Fill in your details in the sidebar and click below to predict your mental health status.")

    if st.button("Predict Mental Health Status", type="primary"):
        input_df = build_input_df()
        with st.spinner("Analyzing..."):
            pred_label, confidence = predict_mental_health(input_df)

        if pred_label:
            st.session_state.prediction_result = {
                "status": pred_label,
                "confidence": confidence,
                "model": model_choice
            }
            st.session_state.show_chatbot = True

            initial_message = f"""
Hello, I'm your Mental Health Support Assistant.  
Your assessment using {model_choice} shows a {pred_label} mental health status.  
Let's talk about how you're feeling today.
"""
            st.session_state.chat_history.append(AIMessage(content=initial_message))
            st.rerun()

# ------------------------------
# CHATBOT INTERFACE
# ------------------------------
else:
    st.header("Mental Health Chatbot")

    if st.session_state.prediction_result:
        result = st.session_state.prediction_result
        st.metric("Mental Health Status", result['status'])
        if result['confidence']:
            st.metric("Confidence", f"{result['confidence']:.1f}%")

    # Pre-filled short chat
    if len(st.session_state.chat_history) == 1:
        preset_chat = [
            HumanMessage(content="Hi there!"),
            AIMessage(content="Hello! How are you feeling today?"),
            HumanMessage(content="A bit stressed lately."),
            AIMessage(content="I understand. Would you like some tips to relax or improve your mood?")
        ]
        st.session_state.chat_history.extend(preset_chat)

    chat_container = st.container(height=400)
    with chat_container:
        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                st.chat_message("user").write(message.content)
            elif isinstance(message, AIMessage):
                st.chat_message("assistant").write(message.content)

    user_input = st.chat_input("Ask me about mental health, stress, or wellness tips...")

    if user_input:
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        try:
            llm = setup_llm()
            with st.spinner("Thinking..."):
                response = llm.invoke(st.session_state.chat_history[-10:])
            st.session_state.chat_history.append(AIMessage(content=response.content))
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")