import streamlit as st
import joblib  # Assuming you saved your model using joblib

st.title('💓🩺💓🩺 Heart Attack Prediction App')
st.write('Use this app to predict your heart attack risk!')

# Create input fields for user input
is_female = st.selectbox("Sex", ["Male", "Female"])
race_ethnicity = st.selectbox("What is your race/ethnicity", 
                               ["White", "Hispanic", "Black", "Asian", "Multiracial", "Other", "Unknown"])
age_category = st.selectbox("Age Category", 
                             ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", 
                              "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"])
bmi_category = st.selectbox("BMI Category: (Underweight <= 18.4), (Healthy 18.5-24.9), (Overweight 25.0-29.9), (Obese >= 30.0)", 
                            ["underweight","healthy", "overweight", "obese"])
general_health = st.selectbox("Would you say that in general your health is:", ["Excellent", "Very good", "Good", "Fair", "Poor", "Unknown"])
physical_activities = st.selectbox("In the past month, aside from your regular job, did you engage in any physical activities or exercises for exercise?", ["0", "1"])
alcohol_drinker = st.selectbox("In the past 30 days, have you consumed at least one alcoholic drink?", ["Yes", "No", "Unknown"])
smoker_status = st.selectbox("Please describe your smoking habit:", ["Never", "Former", "Every day smoker", "Some days smoker"])
deaf_or_hard_of_hearing = st.selectbox("Are you deaf or do you have serious difficulty hearing?", ["Yes", "No", "Unknown"])
blind_or_vision_difficulty = st.selectbox("Are you blind or do you have serious difficulty seeing, even when wearing glasses?", ["Yes", "No", "Unknown"])
difficulty_walking = st.selectbox("Do you have serious difficulty walking or climbing stairs?", ["Yes", "No", "Unknown"])
difficulty_dressing_bathing = st.selectbox("Do you have difficulty dressing or bathing?", ["Yes", "No", "Unknown"])
had_depressive_disorder = st.selectbox("Have you ever been diagnosed with Depressive Disorder?", ["Yes", "No", "Unknown"])
had_diabetes = st.selectbox("Have you ever been diagnosed with Diabetes?", ["Yes", "No", "Pre-diabetes", "Gestational-diabetes", "Unknown"])
had_kidney_disease = st.selectbox("Have you ever been diagnosed with Kidney Disease?", ["Yes", "No", "Unknown"])
had_angina = st.selectbox("Have you ever been diagnosed with Angina, which is a type of chest pain caused by reduced blood flow to the heart?", ["Yes", "No"])
had_stroke = st.selectbox("Have you ever had a Stroke, which is a medical condition where the blood supply to the brain is interrupted or reduced?", ["Yes", "No"])
had_copd = st.selectbox("Have you ever been diagnosed with Chronic Obstructive Pulmonary Disease (COPD), which is a progressive lung disease that makes it difficult to breathe due to airflow blockage?", ["Yes", "No", "Unknown"])
had_arthritis = st.selectbox("Have you ever been diagnosed with Arthritis?", ["Yes", "No", "Unknown"])

# Prepare input data
input_data = [
    1 if is_female == "Female" else 0,  # Gender
    race_ethnicity,
    age_category,
    bmi_category,
    alcohol_drinker,
    general_health,
    smoker_status,
    1 if physical_activities == "1" else 0,  # Physical activities
    1 if had_angina == "Yes" else 0,
    1 if had_stroke == "Yes" else 0,
    1 if had_copd == "Yes" else 0,
    had_diabetes,
    had_kidney_disease,
    had_depressive_disorder,
    had_arthritis,
    deaf_or_hard_of_hearing,
    blind_or_vision_difficulty,
    difficulty_walking,
    difficulty_dressing_bathing
]

# Load the model at the start of the app
@st.cache_resource
def load_model():
    model = joblib.load('heart_attack_model.joblib')  # Update
