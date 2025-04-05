import streamlit as st
import joblib  
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.pipeline import Pipeline



###################
# Configure Page
###################
st.set_page_config(page_title="Heart Attack Prediction App", page_icon="💓", layout="centered")

###################
# Header Section
###################
st.markdown("<h1 style='font-size: 36px; text-align: left; color: #FF5733;'>💓🩺 Heart Attack Prediction App 🩺💓</h1>", unsafe_allow_html=True)
st.write("<h4 style='text-align: left; color: #555;'>Use this app to predict your heart attack risk!</h4>", unsafe_allow_html=True)
st.markdown("<h4 style='font-size: 20px; text-align: left; color: #555;'>Please fill out your information below:</h4>", unsafe_allow_html=True)

###################
# Centered Input Form
###################
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    # Split form into two columns
    left_col, right_col = st.columns(2)
    
    with left_col:
        st.markdown("### Personal Information")
        sex = st.selectbox("Gender", ["Male", "Female"])
        age_category = st.selectbox("Age Category", 
                                  ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", 
                                   "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"])
        bmi_category = st.selectbox("BMI Category", 
                                  ["Underweight", "Healthy", "Overweight", "Obese"])
        race_ethnicity = st.selectbox("Race/Ethnicity", 
                                    ["White", "Hispanic", "Black", "Asian", "Multiracial", "Other", "Unknown"])
        
        st.markdown("### Health Habits")
        physical_activities = st.selectbox("Physical Activities", ["No", "Yes"])  
        alcohol_drinker = st.selectbox("Alcohol Consumption", ["No", "Yes", "Unknown"])  
        smoker_status = st.selectbox("Smoking Status", 
                                  ["Never", "Former", "Every day smoker", "Some days smoker"])
        
    with right_col:
        st.markdown("### Health Status")
        general_health = st.selectbox("General Health", 
                                   ["Excellent", "Very good", "Good", "Fair", "Poor", "Unknown"])
        difficulty_walking = st.selectbox("Walking Difficulty", ["No", "Yes", "Unknown"])
        difficulty_dressing_bathing = st.selectbox("Dressing/Bathing Difficulty", ["No", "Yes", "Unknown"])
        had_depressive_disorder = st.selectbox("Depressive Disorder", ["No", "Yes", "Unknown"])
        had_diabetes = st.selectbox("Diabetes", ["No", "Yes", "Pre-diabetes", "Gestational-diabetes", "Unknown"])
        had_kidney_disease = st.selectbox("Kidney Disease", ["No", "Yes", "Unknown"])
        had_angina = st.selectbox("Angina", ["No", "Yes"])
        had_stroke = st.selectbox("Stroke History", ["No", "Yes"])
        had_copd = st.selectbox("COPD", ["No", "Yes", "Unknown"])
        had_arthritis = st.selectbox("Arthritis", ["No", "Yes", "Unknown"])

###################
# Information Sections (Left-Aligned)
###################
st.markdown("<h2 style='text-align: left;'>Welcome to the Heart Attack Prediction App</h2>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: left;'>
This application assesses your heart attack risk using various health indicators. 
Provide accurate information for reliable predictions.

### How It Works
- Fill in your health information above
- Click the prediction button
- Get instant risk assessment with health guidance

### Disclaimer
This tool provides informational estimates only. Always consult medical professionals for health advice.
</div>
""", unsafe_allow_html=True)

###################
# Load Model
###################
@st.cache_resource
def load_model():
    model_path = 'model/pipeline_logreg_final.joblib'
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()
    
model = load_model()

###################
# Prediction Logic
###################
input_data = [
    sex, race_ethnicity, age_category, bmi_category.lower(), alcohol_drinker,
    general_health, smoker_status, physical_activities, had_angina, had_stroke,
    had_copd, had_diabetes, had_kidney_disease, had_depressive_disorder,
    had_arthritis, deaf_or_hard_of_hearing, blind_or_vision_difficulty,
    difficulty_walking, difficulty_dressing_bathing
]

input_columns = [
    'sex', 'race_ethnicity_category', 'age_category', 'bmi_category',
    'alcohol_drinkers', 'general_health', 'smoker_status',
    'physical_activities', 'had_angina', 'had_stroke', 'had_copd',
    'had_diabetes', 'had_kidney_disease', 'had_depressive_disorder',
    'had_arthritis', 'deaf_or_hard_of_hearing', 'blind_or_vision_difficulty',
    'difficulty_walking', 'difficulty_dressing_bathing'
]

if st.button('Predict Heart Attack Risk', type='primary'):
    input_df = pd.DataFrame([input_data], columns=input_columns)
    try:
        proba = model.predict_proba(input_df)[0][1]
        threshold = model.named_steps['logreg'].threshold
        prediction = 'High Risk' if proba >= threshold else 'Low Risk'
        
        st.subheader('Results')
        if prediction == 'High Risk':
            st.error("""
            ⚠️ High Risk Detected! 
            Please consult a healthcare professional immediately.
            """)
        else:
            st.success("""
            ✅ Low Risk Detected!
            Maintain your healthy lifestyle with regular checkups.
            """)
            
        st.write(f"Risk Probability: {proba:.1%}")
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.write("Please check your inputs and try again.")

###################
# Footer
###################
st.markdown("---")
st.markdown("<div style='text-align: left; color: #777;'>This app uses machine learning for risk estimation. Accuracy depends on data quality and model limitations.</div>", unsafe_allow_html=True)
