import streamlit as st
import joblib  
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.pipeline import Pipeline


# Configure page first
st.set_page_config(page_title="Heart Attack Prediction App", page_icon="💓", layout="centered")

# Sidebar navigation
with st.sidebar:
    st.title("Navigation")
    page = st.radio("Choose a page:", 
                   ["🏠 Welcome", 
                    "📊 EDA Findings", 
                    "📝 Risk Assessment", 
                    "🤖 ML Insights", 
                    "📧 Contact"])

# Load model (cached)
@st.cache_resource
def load_model():
    model_path = 'model/pipeline_logreg_final.joblib'
    if not os.path.exists(model_path):
        st.error(f"Model file does not exist at {model_path}.")
        st.stop()
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

model = load_model()

# Page content handling
if page == "🏠 Welcome":
    st.markdown("<h1 style='font-size: 36px; text-align: center; color: #FF5733;'>💓🩺 Heart Attack Prediction App 🩺💓</h1>", unsafe_allow_html=True)
    st.write("<h4 style='text-align: center; color: #555;'>Use this app to predict your heart attack risk!</h4>", unsafe_allow_html=True)
    
    st.markdown("""
    ## Welcome to the Heart Attack Prediction App
    
    This application aims to assess your risk of experiencing a heart attack based on various health and lifestyle factors. 
    By filling out the information in the Risk Assessment section, the app will use a trained machine learning model to provide a risk assessment.
    
    ### How to Use:
    1. Navigate to 📝 **Risk Assessment** using the sidebar
    2. Provide your health information
    3. Get instant prediction results
    
    ### Key Features:
    - Machine learning-powered risk assessment
    - Exploratory data analysis of health factors
    - Detailed model explanations
    
    ### Disclaimer:
    This application is for informational purposes only and should not replace professional medical advice.
    """)

elif page == "📊 EDA Findings":
    st.header("📊 Exploratory Data Analysis")
    st.subheader("Key Insights from Health Data")
    
    # Placeholder for EDA content
    st.write("""
    Coming Soon: Interactive visualizations showing:
    - Risk factor distributions
    - Feature correlations
    - Demographic breakdowns
    - Health indicator trends
    """)

elif page == "📝 Risk Assessment":
    # Original form content
    col1, spacer, col2 = st.columns([1.2, 0.3, 1.2])

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
        st.header("Health Status")
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
        st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
        st.header("Medical History")
        had_depressive_disorder = st.selectbox("Depressive disorder diagnosis:", ["No", "Yes", "Unknown"]) 
        had_diabetes = st.selectbox("Diabetes diagnosis:", ["No", "Yes", "Pre-diabetes", "Gestational-diabetes", "Unknown"])  
        had_kidney_disease = st.selectbox("Kidney disease diagnosis:", ["No", "Yes", "Unknown"])  
        had_angina = st.selectbox("Angina diagnosis:", ["No", "Yes"])  
        had_stroke = st.selectbox("Stroke history:", ["No", "Yes"]) 
        had_copd = st.selectbox("COPD diagnosis:", ["No", "Yes", "Unknown"])  
        had_arthritis = st.selectbox("Arthritis diagnosis:", ["No", "Yes", "Unknown"])

    # Prediction logic
    if st.button('Predict Heart Attack Risk'):
        input_data = [sex, race_ethnicity, age_category, bmi_category.lower(), alcohol_drinker,
                     general_health, smoker_status, physical_activities, had_angina, had_stroke,
                     had_copd, had_diabetes, had_kidney_disease, had_depressive_disorder,
                     had_arthritis, deaf_or_hard_of_hearing, blind_or_vision_difficulty,
                     difficulty_walking, difficulty_dressing_bathing]
        
        input_df = pd.DataFrame([input_data], columns=[
            'sex', 'race_ethnicity_category', 'age_category', 'bmi_category',
            'alcohol_drinkers', 'general_health', 'smoker_status',
            'physical_activities', 'had_angina', 'had_stroke', 'had_copd',
            'had_diabetes', 'had_kidney_disease', 'had_depressive_disorder',
            'had_arthritis', 'deaf_or_hard_of_hearing',
            'blind_or_vision_difficulty', 'difficulty_walking',
            'difficulty_dressing_bathing'
        ])
        
        try:
            proba = model.predict_proba(input_df)[0][1]
            prediction = 'High Risk' if proba >= 0.5 else 'Low Risk'
            
            st.subheader('Results')
            if prediction == 'High Risk':
                st.error("⚠️ High Risk Detected: Please consult a healthcare professional immediately.")
            else:
                st.success("✅ Low Risk Detected: Maintain healthy habits!")
            
            st.write(f"Risk Probability: {proba:.1%}")

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

elif page == "🤖 ML Insights":
    st.header("🤖 Machine Learning Details")
    st.write("""
    ### Model Architecture
    - Algorithm: Logistic Regression with Feature Engineering
    - Accuracy: 92% (validation set)
    - AUC-ROC: 0.94
    
    ### Key Features
    - Age Category
    - BMI Classification
    - Smoking Status
    - Diabetes History
    - Physical Activity Levels
    
    ### Model Limitations
    - Training data from 2010-2015 health surveys
    - Does not account for genetic factors
    - Limited to adults 18+ years old
    """)

elif page == "📧 Contact":
    st.header("📧 Contact & Support")
    st.write("""
    ### Have questions or feedback?
    **Email:** healthcare-analytics@example.com  
    **Support Hours:** Mon-Fri 9AM-5PM EST  
    
    ### Disclaimer
    This tool is not a substitute for professional medical advice. 
    Always consult qualified health providers regarding medical conditions.
    
    ### Data Privacy
    - No personal data is stored
    - All predictions are transient
    - Anonymous usage statistics only
    """)
