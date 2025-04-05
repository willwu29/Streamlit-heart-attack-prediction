import streamlit as st
import joblib  
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.pipeline import Pipeline

###################
# Create a title
###################


st.title('💓🩺 Heart Attack Prediction App 🩺💓')
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
    
model = load_model()



###############################
# Create User Input on the App
################################


# CSS to make the sidebar wider
st.markdown(
    """
    <style>
        .css-1d391kg {
            width: 400px; /* Adjust the width as needed */
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.header("User Input")
st.markdown("Please fill out the following information:")

# Create input fields for user input
# sex = st.selectbox("Sex", ["Male", "Female"])
# race_ethnicity = st.selectbox("What is your race/ethnicity", 
#                                ["White", "Hispanic", "Black", "Asian", "Multiracial", "Other", "Unknown"])
# age_category = st.selectbox("Age Category", 
#                              ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", 
#                               "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"])
# bmi_category = st.selectbox("BMI Category: (Underweight <= 18.4), (Healthy 18.5-24.9), (Overweight 25.0-29.9), (Obese >= 30.0)", 
#                             ["Underweight", "Healthy", "Overweight", "Obese"])
# general_health = st.selectbox("Would you say that in general your health is:", 
#                               ["Excellent", "Very good", "Good", "Fair", "Poor", "Unknown"])
# physical_activities = st.selectbox("In the past month, aside from your regular job, did you engage in any physical activities or exercises for exercise?", ["No", "Yes"])  
# alcohol_drinker = st.selectbox("In the past 30 days, have you consumed at least one alcoholic drink?", 
#                                ["No", "Yes", "Unknown"])  
# smoker_status = st.selectbox("Please describe your smoking habit:", 
#                              ["Never", "Former", "Some days smoker", "Every day smoker"]) 
# deaf_or_hard_of_hearing = st.selectbox("Are you deaf or do you have serious difficulty hearing?", 
#                                        ["No", "Yes", "Unknown"])  
# blind_or_vision_difficulty = st.selectbox("Are you blind or do you have serious difficulty seeing, even when wearing glasses?", 
#                                           ["No", "Yes", "Unknown"])  
# difficulty_walking = st.selectbox("Do you have serious difficulty walking or climbing stairs?", 
#                                   ["No", "Yes", "Unknown"])  
# difficulty_dressing_bathing = st.selectbox("Do you have difficulty dressing or bathing?", 
#                                            ["No", "Yes", "Unknown"])  
# had_depressive_disorder = st.selectbox("Have you ever been diagnosed with Depressive Disorder?", 
#                                        ["No", "Yes", "Unknown"]) 
# had_diabetes = st.selectbox("Have you ever been diagnosed with Diabetes?", 
#                             ["No", "Yes", "Pre-diabetes", "Gestational-diabetes", "Unknown"])  
# had_kidney_disease = st.selectbox("Have you ever been diagnosed with Kidney Disease?", 
#                                   ["No", "Yes", "Unknown"])  
# had_angina = st.selectbox("Have you ever been diagnosed with Angina, which is a type of chest pain caused by reduced blood flow to the heart?", 
#                           ["No", "Yes"])  
# had_stroke = st.selectbox("Have you ever had a Stroke, which is a medical condition where the blood supply to the brain is interrupted or reduced?", 
#                           ["No", "Yes"])  
# had_copd = st.selectbox("Have you ever been diagnosed with Chronic Obstructive Pulmonary Disease (COPD), which is a progressive lung disease that makes it difficult to breathe due to airflow blockage?", 
#                         ["No", "Yes", "Unknown"])  
# had_arthritis = st.selectbox("Have you ever been diagnosed with Arthritis?", 
#                              ["No", "Yes", "Unknown"])  
with st.sidebar:
    with st.sidebar:
    st.markdown("<h2>Sex</h2>", unsafe_allow_html=True)
    sex = st.selectbox("Select your sex:", ["Male", "Female"])

    st.markdown("<h2>Race/Ethnicity</h2>", unsafe_allow_html=True)
    race_ethnicity = st.selectbox("What is your race/ethnicity", 
                                   ["White", "Hispanic", "Black", "Asian", "Multiracial", "Other", "Unknown"])

    st.markdown("<h2>Age Category</h2>", unsafe_allow_html=True)
    age_category = st.selectbox("Select your age category:", 
                                 ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", 
                                  "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"])

    st.markdown("<h2>BMI Category</h2>", unsafe_allow_html=True)
    bmi_category = st.selectbox("BMI Category: (Underweight <= 18.4), (Healthy 18.5-24.9), (Overweight 25.0-29.9), (Obese >= 30.0)", 
                                ["Underweight", "Healthy", "Overweight", "Obese"])
    
    st.markdown("<h2>General Health</h2>", unsafe_allow_html=True)
    general_health = st.selectbox("Would you say that in general your health is:", 
                                  ["Excellent", "Very good", "Good", "Fair", "Poor", "Unknown"])

    st.markdown("<h2>Physical Activities</h2>", unsafe_allow_html=True)
    physical_activities = st.selectbox("In the past month, did you engage in any physical activities or exercises?", ["No", "Yes"])  
   
    st.markdown("<h2>Alcohol Drinker</h2>", unsafe_allow_html=True)
    alcohol_drinker = st.selectbox("In the past 30 days, have you consumed at least one alcoholic drink?", 
                                   ["No", "Yes", "Unknown"])

    st.markdown("<h2>Smoking Habit</h2>", unsafe_allow_html=True)
    smoker_status = st.selectbox("Please describe your smoking habit:", 
                                 ["Never", "Former", "Every day smoker", "Some days smoker"]) 

    st.markdown("<h2>Hearing Difficulty</h2>", unsafe_allow_html=True)
    deaf_or_hard_of_hearing = st.selectbox("Do you have serious difficulty hearing?", ["No", "Yes", "Unknown"])  

    st.markdown("<h2>Vision Difficulty</h2>", unsafe_allow_html=True)
    blind_or_vision_difficulty = st.selectbox("Do you have serious difficulty seeing, even when wearing glasses?", ["No", "Yes", "Unknown"])  

    st.markdown("<h2>Difficulty Walking</h2>", unsafe_allow_html=True)
    difficulty_walking = st.selectbox("Do you have serious difficulty walking or climbing stairs?", ["No", "Yes", "Unknown"])  

    st.markdown("<h2>Difficulty Dressing/Bathing</h2>", unsafe_allow_html=True)
    difficulty_dressing_bathing = st.selectbox("Do you have difficulty dressing or bathing?", ["No", "Yes", "Unknown"])  
    
    st.markdown("<h2>Depressive Disorder</h2>", unsafe_allow_html=True)
    st.selectbox("Have you ever been diagnosed with Depressive Disorder?", ["No", "Yes", "Unknown"])

    st.markdown("<h2>Diabetes</h2>", unsafe_allow_html=True)
    had_diabetes = st.selectbox("Have you ever been diagnosed with Diabetes?", ["No", "Yes", "Pre-diabetes", "Gestational-diabetes", "Unknown"])

    st.markdown("<h2>Kidney Disease</h2>", unsafe_allow_html=True)
    had_kidney_disease = st.selectbox("Have you ever been diagnosed with Kidney Disease?", ["No", "Yes", "Unknown"])

    st.markdown("<h2>Angina</h2>", unsafe_allow_html=True)
    had_angina = st.selectbox("Have you ever been diagnosed with Angina?", ["No", "Yes"])

    st.markdown("<h2>Stroke</h2>", unsafe_allow_html=True)
    had_stroke = st.selectbox("Have you ever had a Stroke?", ["No", "Yes"])

    st.markdown("<h2>Chronic Obstructive Pulmonary Disease (COPD)</h2>", unsafe_allow_html=True)
    had_copd = st.selectbox("Have you ever been diagnosed with COPD?", ["No", "Yes", "Unknown"])

    st.markdown("<h2>Arthritis</h2>", unsafe_allow_html=True)
    had_arthritis = st.selectbox("Have you ever been diagnosed with Arthritis?", ["No", "Yes", "Unknown"])

    
# Prepare input data
input_data = [
    sex,                                # Gender
    race_ethnicity,                     # race_ethnicity_category
    age_category,                       # age_category
    bmi_category.lower(),               # bmi_category
    alcohol_drinker,                    # alcohol_drinkers
    general_health,                     # general_health
    smoker_status,                      # smoker_status
    physical_activities,                # Physical activities
    had_angina,                         # had_angina
    had_stroke,                         # had_stroke
    had_copd,                           # had_copd
    had_diabetes,                       # had_diabetes
    had_kidney_disease,                 # had_kidney_disease
    had_depressive_disorder,            # had_depressive_disorder
    had_arthritis,                      # had_arthritis
    deaf_or_hard_of_hearing,            # deaf_or_hard_of_hearing
    blind_or_vision_difficulty,         # blind_or_vision_difficulty
    difficulty_walking,                 # difficulty_walking
    difficulty_dressing_bathing         # difficulty_dressing_bathing
]

# Create input column name that match the model inpul column name and order
input_columns = [
       'sex', 'race_ethnicity_category', 'age_category', 'bmi_category',
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

# Create an input DataFrame
input_df = pd.DataFrame([input_data], columns=input_columns)

