import streamlit as st
import pandas as pd
import pickle
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.schema import HumanMessage, AIMessage

# ------------------------------
# CACHING HELPERS
# ------------------------------

@st.cache_resource
def setup_llm():
    """Initialize the Mistral AI model."""
    return ChatMistralAI(
        api_key="lHcwga2vJ6yyjV470WdMIFn5hRgtMbcc",
        model="mistral-large-latest",
        temperature=0.7
    )

@st.cache_resource
def load_models():
    """Load trained mental health models."""
    with open("svm_mental_health.pkl", "rb") as f:
        svm_model = pickle.load(f)
    with open("rf_mental_health.pkl", "rb") as f:
        rf_model = pickle.load(f)
    return svm_model, rf_model

# ------------------------------
# SESSION STATE INITIALIZATION
# ------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "show_chatbot" not in st.session_state:
    st.session_state.show_chatbot = False

# ------------------------------
# PAGE CONFIGURATION
# ------------------------------
st.set_page_config(page_title="Mental Health Assistant", layout="wide")
st.title("🧠 Mental Health Detection & Support Assistant")

# ------------------------------
# SIDEBAR INPUTS
# ------------------------------
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
model_choice = st.sidebar.selectbox("Select Model", ["SVM", "Random Forest"])

# Prepare input dataframe for prediction
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


def predict_mental_health():
    """Predict mental health status using selected model."""
    try:
        svm_model, rf_model = load_models()
        model = svm_model if model_choice == "SVM" else rf_model
        pred = model.predict(input_df)[0]
        label_map = {0: "Bad", 1: "Normal", 2: "Good"}
        return label_map[pred]
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None
    
if not st.session_state.show_chatbot:
    st.header("📊 Mental Health Assessment")
    st.write("Fill in your information and click the button to get your mental health status prediction.")

    if st.button("🔍 Predict Mental Health Status", type="primary"):
        pred_label = predict_mental_health()
        if pred_label:
            # Save prediction result
            st.session_state.prediction_result = {
                "status": pred_label,
                "age": age,
                "gender": gender,
                "education": education,
                "profession": profession,
                "sleep": sleep_hours,
                "activity": physical_activity,
                "social": social_interaction,
                "screen_time": screen_time,
                "work_hours": work_hours
            }

            # Display result
            st.subheader("Predicted Mental Health Status:")
            if pred_label == "Good":
                st.success(f"✅ {pred_label}")
            elif pred_label == "Normal":
                st.info(f"ℹ️ {pred_label}")
            else:
                st.warning(f"⚠️ {pred_label}")

            # Automatically show chatbot after prediction
            st.session_state.show_chatbot = True
            initial_message = f"""Hello! I'm your Mental Health Support Assistant. Your assessment shows **{pred_label}** mental health status.

I can provide support, explain results, and give lifestyle tips. How can I assist you today?"""
            st.session_state.chat_history.append(AIMessage(content=initial_message))
            st.rerun()

# ------------------------------
# CHATBOT INTERFACE
# ------------------------------
if st.session_state.show_chatbot:
    

    # Display prediction summary
    if st.session_state.prediction_result:
        result = st.session_state.prediction_result
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mental Health Status", result['status'])
        with col2:
            st.metric("Sleep Hours", f"{result['sleep']} hrs/day")
        with col3:
            st.metric("Physical Activity", f"{result['activity']} min/wk")

    # Chat container
    with st.container(height=400):
        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                st.chat_message("user").write(message.content)
            elif isinstance(message, AIMessage):
                st.chat_message("assistant").write(message.content)

    # Chat input
    user_input = st.chat_input("Ask me about mental health, coping strategies, or your results...")
    if user_input:
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        try:
            llm = setup_llm()
            
            # System prompt with prediction context
            system_context = """You are a compassionate mental health support assistant. Your role:
1. Provide empathetic support
2. Offer coping strategies and wellness tips
3. Explain mental health assessment results
4. Encourage healthy habits
5. Remind users you are not a replacement for professional help."""
            
            if st.session_state.prediction_result:
                result = st.session_state.prediction_result
                system_context += f"\n\nUser's Assessment Result: {result['status']}\nDetails: Age {result['age']}, Sleep {result['sleep']}hrs, Activity {result['activity']}min/wk, Screen {result['screen_time']}hrs/day, Work {result['work_hours']}hrs/wk"
           
            messages = [HumanMessage(content=system_context)] + st.session_state.chat_history[-10:]
            response = llm.invoke(messages)
            st.session_state.chat_history.append(AIMessage(content=response.content))
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Check your API key and internet connection.")


if st.button("🗑️ Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()
