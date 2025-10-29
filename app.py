import streamlit as st
import pandas as pd
import pickle
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.schema import HumanMessage, AIMessage

def load_models():
    with open("svm_mental_health.pkl", "rb") as f:
        svm_model = pickle.load(f)
    with open("rf_mental_health.pkl", "rb") as f:
        rf_model = pickle.load(f)
    return svm_model, rf_model


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "show_chatbot" not in st.session_state:
    st.session_state.show_chatbot = False

st.set_page_config(page_title="Mental Health Assistant", layout="wide")
st.title("🧠 Mental Health Detection & Support Assistant")

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
    try:
        svm_model, rf_model = load_models()
        model = svm_model if model_choice == "SVM" else rf_model
        pred = model.predict(input_df)[0]
        label_map = {0: "Bad", 1: "Normal", 2: "Good"}
        return label_map[pred]
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

# ------------------------------
# PREDICTION INTERFACE
# ------------------------------
if not st.session_state.show_chatbot:
    st.header("📊 Mental Health Assessment")
    st.write("Fill in your information in the sidebar and click the button below to get your mental health status prediction.")

    if st.button("🔍 Predict Mental Health Status", type="primary"):
        input_df = build_input_df()
        pred_label = predict_mental_health(input_df)
        
        if pred_label:
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
            
            st.subheader("Predicted Mental Health Status:")
            if pred_label == "Good":
                st.success(f"✅ {pred_label}")
            elif pred_label == "Normal":
                st.info(f"ℹ️ {pred_label}")
            else:
                st.warning(f"⚠️ {pred_label}")

            st.session_state.show_chatbot = True
            
            # Add initial greeting message
            initial_message = f"""Hello! I'm your Mental Health Support Assistant. Your assessment shows a **{pred_label}** mental health status.

I'm here to help you understand your results and provide support. You can ask me about:
- Understanding your mental health status
- Tips for improving sleep, exercise, or lifestyle
- Stress management and coping strategies
- General mental health advice

How can I assist you today?"""
            st.session_state.chat_history.append(AIMessage(content=initial_message))
            st.rerun()

# ------------------------------
# CHATBOT INTERFACE
# ------------------------------
else:
    st.header("💬 Mental Health Support Chatbot")
    
    # Display metrics
    if st.session_state.prediction_result:
        result = st.session_state.prediction_result
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mental Health Status", result['status'])
        with col2:
            st.metric("Sleep Hours", f"{result['sleep']} hrs/day")
        with col3:
            st.metric("Physical Activity", f"{result['activity']} min/wk")
    
    st.write("---")
    
    # Chat container
    chat_container = st.container(height=400)
    with chat_container:
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
            
            # System context
            system_context = """You are a compassionate mental health support assistant. Your role is to:
1. Provide empathetic support and understanding
2. Offer evidence-based coping strategies and wellness tips
3. Help users understand their mental health assessment results
4. Encourage healthy habits like sleep, exercise, and social connection
5. ALWAYS remind users that you're not a replacement for professional help

Important: If someone expresses severe distress, suicidal thoughts, or crisis, urge them to contact emergency services or a crisis hotline immediately."""
            
            # Add user's prediction context
            if st.session_state.prediction_result:
                result = st.session_state.prediction_result
                context_info = f"""

User's Assessment Results:
- Mental Health Status: {result['status']}
- Age: {result['age']}
- Gender: {result['gender']}
- Education: {result['education']}
- Profession: {result['profession']}
- Sleep: {result['sleep']} hours/day
- Physical Activity: {result['activity']} minutes/week
- Social Interactions: {result['social']}
- Screen Time: {result['screen_time']} hours/day
- Work Hours: {result['work_hours']} hours/week

Use this information to provide personalized advice."""
                system_context += context_info

            # Prepare messages (last 10 for context)
            messages = [HumanMessage(content=system_context)] + st.session_state.chat_history[-10:]
            
            # Get AI response
            with st.spinner("Thinking..."):
                response = llm.invoke(messages)
            
            st.session_state.chat_history.append(AIMessage(content=response.content))
            st.rerun()
            
        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Please check your API key and internet connection.")
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 New Assessment"):
            st.session_state.chat_history = []
            st.session_state.prediction_result = None
            st.session_state.show_chatbot = False
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            # Re-add initial greeting
            if st.session_state.prediction_result:
                result = st.session_state.prediction_result
                initial_message = f"""Hello! I'm your Mental Health Support Assistant. Your assessment shows a **{result['status']}** mental health status.

I'm here to help you understand your results and provide support. You can ask me about:
- Understanding your mental health status
- Tips for improving sleep, exercise, or lifestyle
- Stress management and coping strategies
- General mental health advice

How can I assist you today?"""
                st.session_state.chat_history.append(AIMessage(content=initial_message))
            st.rerun()

# Footer
st.markdown("---")

st.markdown("⚠️ **Disclaimer:** This tool is for informational purposes only and not a substitute for professional mental health care. If you're in crisis, please contact a mental health professional or emergency services immediately.")
