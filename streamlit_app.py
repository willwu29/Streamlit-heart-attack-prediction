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

# Sidebar for user input organized into sections
with st.sidebar:
    st.markdown("<h3 style='font-size: 24px;'>Personal Information</h3>", unsafe_allow_html=True)
    sex = st.selectbox("Select your sex:", ["Male", "Female"])
    race_ethnicity = st.selectbox("What is your race/ethnicity", 
                                   ["White", "Hispanic", "Black", "Asian", "Multiracial", "Other", "Unknown"])
    age_category = st.selectbox("Select your age category:", 
                                 ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", 
                                  "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"])

    st.markdown("<h3 style='font-size: 24px;'>Habits & Lifestyle</h3>", unsafe_allow_html=True)
    bmi_category = st.selectbox("BMI Category: (Underweight <= 18.4), (Healthy 18.5-24.9), (Overweight 25.0-29.9), (Obese >= 30.0)", 
                                ["Underweight", "Healthy", "Overweight", "Obese"])
    general_health = st.selectbox("Would you say that in general your health is:", 
                                  ["Excellent", "Very good", "Good", "Fair", "Poor", "Unknown"])
    physical_activities = st.selectbox("In the past month, did you engage in any physical activities or exercises?", ["No", "Yes"])  
    alcohol_drinker = st.selectbox("In the past 30 days, have you consumed at least one alcoholic drink?", 
                                   ["No", "Yes", "Unknown"])  
    smoker_status = st.selectbox("Please describe your smoking habit:", 
                                 ["Never", "Former", "Every day smoker", "Some days smoker"])

    st.markdown("<h3 style='font-size: 24px;'>Medical History</h3>", unsafe_allow_html=True)
    deaf_or_hard_of_hearing = st.selectbox("Do you have serious difficulty hearing?", ["No", "Yes", "Unknown"])  
    blind_or_vision_difficulty = st.selectbox("Do you have serious difficulty seeing, even when wearing glasses?", ["No", "Yes", "Unknown"])  
    difficulty_walking = st.selectbox("Do you have serious difficulty walking or climbing stairs?", ["No", "Yes", "Unknown"])  
    difficulty_dressing_bathing = st.selectbox("Do you have difficulty dressing or bathing?", ["No", "Yes", "Unknown"])  
    had_depressive_disorder = st.selectbox("Have you ever been diagnosed with Depressive Disorder?", ["No", "Yes", "Unknown"]) 
    had_diabetes = st.selectbox("Have you ever been diagnosed with Diabetes?", ["No", "Yes", "Pre-diabetes", "Gestational-diabetes", "Unknown"])  
    had_kidney_disease = st.selectbox("Have you ever been diagnosed with Kidney Disease?", ["No", "Yes", "Unknown"])  
    had_angina = st.selectbox("Have you ever been diagnosed with Angina?", ["No", "Yes"])  
    had_stroke = st.selectbox("Have you ever had a Stroke?", ["No", "Yes"])  
    had_copd = st.selectbox("Have you ever been diagnosed with Chronic Obstructive Pulmonary Disease (COPD)?", ["No", "Yes", "Unknown"])  
    had_arthritis = st.selectbox("Have you ever been diagnosed with Arthritis?", ["No", "Yes", "Unknown"])

# Prepare input data
input_data = [
    1 if sex == "Female" else 0,  # Gender
    race_ethnicity,                # Race/Ethnicity
    age_category,                  # Age Category
    bmi_category.lower(),          # BMI Category
    alcohol_drinker,               # Alcohol Drinkers
    general_health.lower(),        # Convert to lowercase
    smoker_status.lower(),         # Convert to lowercase
    1 if physical_activities == "Yes" else 0,  # Physical Activities
    1 if had_angina == "Yes" else 0,  # Had Angina
    1 if had_stroke == "Yes" else 0,  # Had Stroke
    1 if had_copd == "Yes" else 0,    # Had COPD
    had_diabetes.lower(),           # Convert to lowercase
    had_kidney_disease.lower(),     # Convert to lowercase
    had_depressive_disorder.lower(), # Convert to lowercase
    had_arthritis.lower(),           # Convert to lowercase
    1 if deaf_or_hard_of_hearing == "Yes" else 0,  # Hearing Difficulty
    1 if blind_or_vision_difficulty == "Yes" else 0,  # Vision Difficulty
    1 if difficulty_walking == "Yes" else 0,      # Difficulty Walking
    1 if difficulty_dressing_bathing == "Yes" else 0  # Difficulty Dressing/Bathing
]

# Create input column names that match the model input column name and order
input_columns = [
    'is_female', 'race_ethnicity_category', 'age_category', 'bmi_category',
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
