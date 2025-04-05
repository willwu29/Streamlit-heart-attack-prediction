import streamlit as st
import joblib  
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.pipeline import Pipeline


# ###################
# # Create a title
# ###################


# st.set_page_config(page_title="Heart Attack Prediction App", page_icon="💓", layout="centered")

# st.markdown("<h1 style='font-size: 36px; text-align: center; color: #FF5733;'>💓🩺 Heart Attack Prediction App 🩺💓</h1>", unsafe_allow_html=True)
# st.write("<h4 style='text-align: center; color: #555;'>Use this app to predict your heart attack risk!</h4>", unsafe_allow_html=True)

# # Centered message with adjusted styling
# st.markdown("<h4 style='font-size: 20px; text-align: center; color: #555;'>Please fill out your information on the left!</h4>", unsafe_allow_html=True)

# # Introduction section
# st.markdown("<h2 style='text-align: center;'>Welcome to the Heart Attack Prediction App</h2>", unsafe_allow_html=True)
# st.markdown("""
# This application aims to assess your risk of experiencing a heart attack based on various health and lifestyle factors. 
# By filling out the information on the left, the app will use a trained machine learning model to provide a risk assessment. 
# Please ensure to provide accurate information to receive a reliable prediction.

# ### How It Works
# - The model utilizes multiple health indicators to predict heart attack risk.
# - After you fill in the required fields, click on the *Predict Heart Attack Risk* button.
# - The app will calculate the prediction and display the results along with important health guidance.

# ### Disclaimer
# This application is for informational purposes only and should not be considered a substitute for professional medical advice.
# In case of health concerns, consult a qualified healthcare professional.
# """)

###################
# Load Model
###################
@st.cache_resource
def load_model():
    model_path = 'model/pipeline_logreg_final.joblib'
    if not os.path.exists(model_path):
        st.error(f"Model file does not exist at {model_path}.")
        st.stop()
    try:
        model = joblib.load(model_path)
        return model
    except ModuleNotFoundError as e:
        st.error(f"Failed to load model due to a missing module: {str(e)}")
        st.stop()

model = load_model()

# Create columns with adjusted ratios
col1, spacer, col2 = st.columns([1.2, 0.3, 1.2])  # Wider left column

with col1:
    # Personal Information
    st.header("Personal Info")
    sex = st.selectbox("Gender:", ["Male", "Female"])
    race_ethnicity = st.selectbox("Race/Ethnicity:", 
                                ["White", "Hispanic", "Black", "Asian", "Multiracial", "Other", "Unknown"])
    age_category = st.selectbox("Age Category:", 
                                ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", 
                                "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"])
    
    # Health Condition
    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
    st.header("Health Condition")
    bmi_category = st.selectbox("BMI Category:", 
                                ["Underweight", "Healthy", "Overweight", "Obese"],
                                help="Underweight ≤18.4, Healthy 18.5-24.9, Overweight 25.0-29.9, Obese ≥30.0")
    general_health = st.selectbox("How would you rate your Health Condition:", 
                                ["Excellent", "Very good", "Good", "Fair", "Poor", "Unknown"])
    deaf_or_hard_of_hearing = st.selectbox("Hearing Difficulty:", ["No", "Yes", "Unknown"])  
    blind_or_vision_difficulty = st.selectbox("Vision Difficulty (Even when wearing glasses):", ["No", "Yes", "Unknown"])  
    difficulty_walking = st.selectbox("Walking & Climbing stairs Difficulty:", ["No", "Yes", "Unknown"])  
    difficulty_dressing_bathing = st.selectbox("Dressing & Bathing Difficulty:", ["No", "Yes", "Unknown"])

with col2:
    # Habits & Lifestyle
    st.header("Habits & Lifestyle")
    physical_activities = st.selectbox("Any Physical activities in past 30 days:", ["No", "Yes"])  
    alcohol_drinker = st.selectbox("Any Alcohol consumption in past 30 days:", ["No", "Yes", "Unknown"])  
    smoker_status = st.selectbox("Smoking status:", 
                               ["Never", "Former", "Every day smoker", "Some days smoker"])
    
    # Medical History
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)  # Increased spacing
    st.header("Medical History")
    had_depressive_disorder = st.selectbox("Depressive disorder diagnosis:", ["No", "Yes", "Unknown"]) 
    had_diabetes = st.selectbox("Diabetes diagnosis:", ["No", "Yes", "Pre-diabetes", "Gestational-diabetes", "Unknown"])  
    had_kidney_disease = st.selectbox("Kidney disease diagnosis:", ["No", "Yes", "Unknown"])  
    had_angina = st.selectbox("Angina diagnosis:", ["No", "Yes"])  
    had_stroke = st.selectbox("Stroke history:", ["No", "Yes"]) 
    had_copd = st.selectbox("COPD diagnosis:", ["No", "Yes", "Unknown"])  
    had_arthritis = st.selectbox("Arthritis diagnosis:", ["No", "Yes", "Unknown"])

# Spacer column remains empty
with spacer:
    st.empty()

# Prepare input data
input_data = [
    sex,  # Gender
    race_ethnicity,                # Race/Ethnicity
    age_category,                  # Age Category
    bmi_category.lower(),          # BMI Category
    alcohol_drinker,               # Alcohol Drinkers
    general_health,        # Convert to lowercase
    smoker_status,         # Convert to lowercase
    physical_activities,  # Physical Activities
    had_angina,  # Had Angina
    had_stroke,  # Had Stroke
    had_copd,    # Had COPD
    had_diabetes,           # Convert to lowercase
    had_kidney_disease,     # Convert to lowercase
    had_depressive_disorder, # Convert to lowercase
    had_arthritis,           # Convert to lowercase
    deaf_or_hard_of_hearing,  # Hearing Difficulty
    blind_or_vision_difficulty,  # Vision Difficulty
    difficulty_walking,      # Difficulty Walking
    difficulty_dressing_bathing  # Difficulty Dressing/Bathing
]

# Create input column names that match the model input column names and order
input_columns = [
    'sex', 'race_ethnicity_category', 'age_category', 'bmi_category',
    'alcohol_drinkers', 'general_health', 'smoker_status',
    'physical_activities', 'had_angina', 'had_stroke', 'had_copd',
    'had_diabetes', 'had_kidney_disease', 'had_depressive_disorder',
    'had_arthritis', 'deaf_or_hard_of_hearing',
    'blind_or_vision_difficulty', 'difficulty_walking',
    'difficulty_dressing_bathing'
]

# Create input DataFrame
input_df = pd.DataFrame([input_data], columns=input_columns)

##################
# Make Prediction
##################
if st.button('Predict Heart Attack Risk'):
    try:
        proba = model.predict_proba(input_df)[0][1]  # Get probability of the positive class
        threshold = 0.5  # Define a threshold for classification

        prediction = 'High Risk' if proba >= threshold else 'Low Risk'
        
        st.subheader('Results')
        if prediction == 'High Risk':
            st.error("⚠️ Warning! Our assessment indicates you are at high risk for a heart attack. " 
                      "It is crucial that you consult a healthcare professional immediately for further evaluation.")
        else:
            st.success("✅ Good News! Our assessment indicates you are at low risk for a heart attack. Keep up the good work and maintain a healthy lifestyle!")

    except Exception as e:
        st.error(f"An error occurred while making the prediction: {str(e)}")
