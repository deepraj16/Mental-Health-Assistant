import streamlit as st
import pandas as pd
import pickle
import numpy as np
from tensorflow import keras
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.schema import HumanMessage, AIMessage


@st.cache_resource
def setup_llm():
    return ChatMistralAI(
        api_key="lHcwga2vJ6yyjV470WdMIFn5hRgtMbcc",
        model="mistral-large-latest",
        temperature=0.7
    )

@st.cache_resource
def load_models():
    """Load all ML models and preprocessing objects"""
    svm_model, rf_model, ann_model, label_encoders, scaler = None, None, None, None, None
    
    try:
        with open("svm_mental_health.pkl", "rb") as f:
            svm_model = pickle.load(f)
    except Exception as e:
        st.warning(f"⚠️ SVM model could not be loaded: {str(e)[:100]}")
    
    try:
        with open("rf_mental_health.pkl", "rb") as f:
            rf_model = pickle.load(f)
    except Exception as e:
        st.warning(f"⚠️ Random Forest model could not be loaded: {str(e)[:100]}")
    
    try:
        ann_model = keras.models.load_model('model.keras')
        with open('preprocessing.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    except Exception as e:
        st.error(f"ANN model could not be loaded: {e}")
    
    return svm_model, rf_model, ann_model, label_encoders, scaler


# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "show_chatbot" not in st.session_state:
    st.session_state.show_chatbot = False


st.set_page_config(page_title="Mental Health Assistant", layout="wide")
st.title("Mental Health Detection & Support Assistant")

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

# Get available models
svm_model, rf_model, ann_model, label_encoders, scaler = load_models()
available_models = []
if svm_model is not None:
    available_models.append("SVM")
if rf_model is not None:
    available_models.append("Random Forest")
if ann_model is not None:
    available_models.append("ANN (Neural Network)")

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
    try:
        svm_model, rf_model, ann_model, label_encoders, scaler = load_models()
        
        if model_choice == "ANN (Neural Network)":
            if ann_model is None or label_encoders is None or scaler is None:
                st.error("ANN model or preprocessing objects not loaded")
                return None, None
            
            # Create a copy for preprocessing
            ann_input = pd.DataFrame({
                'Age': [input_df['Age'].values[0]],
                'Gender': [input_df['Gender'].values[0]],
                'Education_Level': [input_df['Education_Level'].values[0]],
                'Profession': [input_df['Profession'].values[0]],
                'Sleep_Hours': [input_df['Sleep_Hours'].values[0]],
                'Physical_Activity_min/wk': [input_df['Physical_Activity_min/wk'].values[0]],
                'Social_Interactions': [input_df['Social_Interactions'].values[0]],
                'Screen_Time_hrs/day': [input_df['Screen_Time_hrs/day'].values[0]],
                'Work_Hours_hrs/wk': [input_df['Work_Hours_hrs/wk'].values[0]]
            })
            
            # Encode categorical columns one by one
            try:
                ann_input.loc[:, 'Gender'] = label_encoders['Gender'].transform(ann_input['Gender'])
            except:
                ann_input.loc[:, 'Gender'] = label_encoders['Gender'].transform(ann_input['Gender'].values.reshape(-1, 1)).ravel()
            
            try:
                ann_input.loc[:, 'Education_Level'] = label_encoders['Education_Level'].transform(ann_input['Education_Level'])
            except:
                ann_input.loc[:, 'Education_Level'] = label_encoders['Education_Level'].transform(ann_input['Education_Level'].values.reshape(-1, 1)).ravel()
            
            try:
                ann_input.loc[:, 'Profession'] = label_encoders['Profession'].transform(ann_input['Profession'])
            except:
                ann_input.loc[:, 'Profession'] = label_encoders['Profession'].transform(ann_input['Profession'].values.reshape(-1, 1)).ravel()
            
            try:
                ann_input.loc[:, 'Social_Interactions'] = label_encoders['Social_Interactions'].transform(ann_input['Social_Interactions'])
            except:
                ann_input.loc[:, 'Social_Interactions'] = label_encoders['Social_Interactions'].transform(ann_input['Social_Interactions'].values.reshape(-1, 1)).ravel()
            
            # Convert to numpy array for scaling
            ann_input_array = ann_input.values
            
            # Scale features
            ann_input_scaled = scaler.transform(ann_input_array)
            
            # Predict
            pred_probs = ann_model.predict(ann_input_scaled, verbose=0)
            
            # Determine predicted class
            if pred_probs.shape[1] > 2:
                pred_class = np.argmax(pred_probs, axis=1)
            else:
                pred_class = (pred_probs > 0.5).astype(int).flatten()
            
            # Convert back to original labels
            print(pred_class)
          
            print("?????????????????????????????")
            try:
                pred_label = label_encoders['target'].inverse_transform(pred_class)[0]
            except:
                # If target encoder expects 2D input
                pred_label = label_encoders['target'].inverse_transform(pred_class.reshape(-1, 1))[0][0] if pred_class.ndim == 1 else label_encoders['target'].inverse_transform(pred_class)[0]
            
            confidence = float(np.max(pred_probs[0]) * 100)
            if pred_label ==0 :
                pred_label="Bad"
            elif pred_label ==1 :
                pred_label="Normal"
            else:
                pred_label="Good"
            return pred_label, confidence
            
        else:
            # SVM or Random Forest prediction
            model = svm_model if model_choice == "SVM" else rf_model
            
            if model is None:
                st.error(f"{model_choice} model not loaded")
                return None, None
            
            pred = model.predict(input_df)[0]
            label_map = {0: "Bad", 1: "Normal", 2: "Good"}
            pred_label = label_map[pred]
            
            # Get probability if available
            confidence = None
            try:
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(input_df)[0]
                    confidence = float(np.max(probs) * 100)
            except:
                pass
            
            return pred_label, confidence
            
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None


if not st.session_state.show_chatbot:
    st.header(" Mental Health Assessment")
    st.write("Fill in your information in the sidebar and click the button below to get your mental health status prediction.")
    
    # Display input summary
    with st.expander("View Your Input Summary"):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Age:** {age}")
            st.write(f"**Gender:** {gender}")
            st.write(f"**Education:** {education}")
            st.write(f"**Profession:** {profession}")
        with col2:
            st.write(f"**Sleep Hours:** {sleep_hours} hrs/day")
            st.write(f"**Physical Activity:** {physical_activity} min/wk")
            st.write(f"**Social Interactions:** {social_interaction}")
            st.write(f"**Screen Time:** {screen_time} hrs/day")
            st.write(f"**Work Hours:** {work_hours} hrs/wk")

    if st.button(" Predict Mental Health Status", type="primary"):
        input_df = build_input_df()
        
        with st.spinner("Analyzing your mental health status..."):
            pred_label, confidence = predict_mental_health(input_df)
        
        if pred_label:
            st.session_state.prediction_result = {
                "status": pred_label,
                "confidence": confidence,
                "model": model_choice,
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
            
            
            col1, col2 = st.columns([2, 1])
            with col1:
                if pred_label == "Good":
                    st.success(f" **{pred_label}**")
                elif pred_label == "Normal":
                    st.info(f" **{pred_label}**")
                else:
                    st.warning(f" **{pred_label}**")
            
            with col2:
                if confidence:
                    st.metric("Confidence", f"{confidence:.1f}%")
                st.caption(f"Model: {model_choice}")

            st.session_state.show_chatbot = True
            
            # Add initial greeting message
            confidence_text = f" with {confidence:.1f}% confidence" if confidence else ""
            initial_message = f"""Hello! I'm your Mental Health Support Assistant. 

Your assessment using **{model_choice}** shows a **{pred_label}** mental health status{confidence_text}.

I'm here to help you understand your results and provide support. You can ask me about:
-  Understanding your mental health status
-  Tips for improving sleep, exercise, or lifestyle
-  Stress management and coping strategies
- General mental health advice
- Interpretation of your specific metrics

How can I assist you today?"""
            st.session_state.chat_history.append(AIMessage(content=initial_message))
            st.rerun()

# ------------------------------
# CHATBOT INTERFACE
# ------------------------------
else:
    st.header("Mental Health Support Chatbot")
    
    # Display metrics
    if st.session_state.prediction_result:
        result = st.session_state.prediction_result
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if result['status'] == "Good":
                st.metric("Mental Health Status", result['status'], delta="Positive")
            elif result['status'] == "Normal":
                st.metric("Mental Health Status", result['status'])
            else:
                st.metric("Mental Health Status", result['status'], delta="Needs Attention", delta_color="inverse")
        with col2:
            st.metric("Sleep Hours", f"{result['sleep']} hrs/day")
        with col3:
            st.metric("Physical Activity", f"{result['activity']} min/wk")
        with col4:
            if result['confidence']:
                st.metric("Confidence", f"{result['confidence']:.1f}%")
            else:
                st.metric("Model", result['model'])
    
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
                confidence_info = f"- Prediction Confidence: {result['confidence']:.1f}%\n" if result['confidence'] else ""
                context_info = f"""

User's Assessment Results (Model: {result['model']}):
- Mental Health Status: {result['status']}
{confidence_info}- Age: {result['age']}
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
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("New Assessment"):
            st.session_state.chat_history = []
            st.session_state.prediction_result = None
            st.session_state.show_chatbot = False
            st.rerun()
    
    with col2:
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            # Re-add initial greeting
            if st.session_state.prediction_result:
                result = st.session_state.prediction_result
                confidence_text = f" with {result['confidence']:.1f}% confidence" if result['confidence'] else ""
                initial_message = f"""Hello! I'm your Mental Health Support Assistant. 

Your assessment using **{result['model']}** shows a **{result['status']}** mental health status{confidence_text}.

I'm here to help you understand your results and provide support. You can ask me about:
-  Understanding your mental health status
-  Tips for improving sleep, exercise, or lifestyle
-  Stress management and coping strategies
-  General mental health advice
-  Interpretation of your specific metrics

How can I assist you today?"""
                st.session_state.chat_history.append(AIMessage(content=initial_message))
            st.rerun()
    
    with col3:
        if st.button("Export Chat"):
            chat_export = "\n\n".join([
                f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
                for msg in st.session_state.chat_history
            ])
            st.download_button(
                label="Download Chat",
                data=chat_export,
                file_name="mental_health_chat.txt",
                mime="text/plain"
            )

st.markdown("---")
