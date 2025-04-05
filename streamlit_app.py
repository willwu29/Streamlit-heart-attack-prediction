import streamlit as st
import joblib  
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder  

###################
# Create a title
###################
st.title('💓🩺 Heart Attack Prediction App')
st.write('Use this app to predict your heart attack risk!')


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


###############################
# Create User Input on the App
################################
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
physical_activities = st.selectbox("In the past month, aside from your regular job, did you engage in any physical activities or exercises for exercise?", ["Yes", "No"])
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
    race_ethnicity,                       # race_ethnicity_category
    age_category,                         # age_category
    bmi_category,                         # bmi_category
    alcohol_drinker,                     # alcohol_drinkers
    general_health,                       # general_health
    smoker_status,                        # smoker_status
    1 if physical_activities == "1" else 0,  # Physical activities
    1 if had_angina == "Yes" else 0,     # had_angina
    1 if had_stroke == "Yes" else 0,     # had_stroke
    had_copd,                       # had_copd
    had_diabetes,                         # had_diabetes
    had_kidney_disease,                   # had_kidney_disease
    had_depressive_disorder,              # had_depressive_disorder
    had_arthritis,                        # had_arthritis
    deaf_or_hard_of_hearing,              # deaf_or_hard_of_hearing
    blind_or_vision_difficulty,            # blind_or_vision_difficulty
    difficulty_walking,                    # difficulty_walking
    difficulty_dressing_bathing            # difficulty_dressing_bathing
]

# Create input column name that match the model inpul column name and order
input_columns = [
     'is_female', 'race_ethnicity_category', 'age_category', 'bmi_category',
       'alcohol_drinkers', 'general_health', 'smoker_status',
       'physical_activities', 'had_angina', 'had_stroke', 'had_copd',
       'had_diabetes', 'had_kidney_disease', 'had_depressive_disorder',
       'had_arthritis', 'deaf_or_hard_of_hearing',
       'blind_or_vision_difficulty', 'difficulty_walking',
       'difficulty_dressing_bathing']

# Creae a input df
input_df = pd.DataFrame([input_data], columns=input_columns)


##################
# Make Prediction
##################
# Get prediction probability and apply custom threshold
if st.button('Predict Heart Attack Risk'):
    try:
        # Get probability of positive class
        proba = model.predict_proba(input_df)[0][1]
        
        # Retrieve threshold from the model
        model_threshold = model.named_steps['logreg'].threshold
        
        # Make prediction using custom threshold
        prediction = 'High Risk' if proba >= model_threshold else 'Low Risk'
        
        # Display results
        st.subheader('Results')
        
        # Add interpretation
        if prediction == 'High Risk':
            st.error("⚠️ Warning! Our assessment indicates you are at high risk for a heart attack. " \
            "It is crucial that you consult a healthcare professional immediately for further evaluation.")
        else:
            st.success("✅ Good News! Our assessment indicates you are at low risk for a heart attack. Keep up the good work and maintain a healthy lifestyle!")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please check your input values and try again.")

