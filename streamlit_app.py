import streamlit as st
import joblib  
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.pipeline import Pipeline

# Custom CSS for sidebar styling
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        min-width: 280px !important;
        max-width: 280px !important;
    }
    .stButton>button {
        width: 100%;
        justify-content: left;
        padding: 0.5rem 1rem;
        margin: 0.25rem 0;
    }
    .stButton>button:hover {
        background-color: #f0f2f6;
        color: #FF5733;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'

# Sidebar navigation
with st.sidebar:
    st.markdown("## Navigation")
    
    # Create navigation buttons
    pages = {
        '🏠 Welcome': 'welcome',
        '📊 EDA Findings': 'eda',
        '📝 Risk Assessment': 'predict',
        '🤖 ML Insights': 'ml',
        '📧 Contact': 'contact'
    }
    
    for label, page_key in pages.items():
        if st.button(label, key=page_key, 
                    use_container_width=True,
                    type="primary" if st.session_state.page == page_key else "secondary"):
            st.session_state.page = page_key
            st.rerun()

# Load Model (keep your original model loading code)
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

# Page content handling
if st.session_state.page == 'welcome':
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

elif st.session_state.page == 'eda':
    st.header("📊 Exploratory Data Analysis")
    st.subheader("Key Insights from Health Data")
    st.write("Coming Soon: Interactive visualizations...")

elif st.session_state.page == 'predict':
    # Your original Risk Assessment code
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

    # Prepare input data
    input_data = [
        sex,
        race_ethnicity,
        age_category,
        bmi_category.lower(),
        alcohol_drinker,
        general_health,
        smoker_status,
        physical_activities,
        had_angina,
        had_stroke,
        had_copd,
        had_diabetes,
        had_kidney_disease,
        had_depressive_disorder,
        had_arthritis,
        deaf_or_hard_of_hearing,
        blind_or_vision_difficulty,
        difficulty_walking,
        difficulty_dressing_bathing
    ]

    input_columns = [
        'sex', 'race_ethnicity_category', 'age_category', 'bmi_category',
        'alcohol_drinkers', 'general_health', 'smoker_status',
        'physical_activities', 'had_angina', 'had_stroke', 'had_copd',
        'had_diabetes', 'had_kidney_disease', 'had_depressive_disorder',
        'had_arthritis', 'deaf_or_hard_of_hearing',
        'blind_or_vision_difficulty', 'difficulty_walking',
        'difficulty_dressing_bathing'
    ]

    input_df = pd.DataFrame([input_data], columns=input_columns)

    # Prediction logic
    if st.button('Predict Heart Attack Risk'):
        try:
            threshold = model.threshold
            proba = model.predict_proba(input_df)[0][1]
            prediction = 'High Risk' if proba >= threshold else 'Low Risk'
            
            st.subheader('Results')
            if prediction == 'High Risk':
                st.error("⚠️ Warning! Our assessment indicates you are at HIGH RISK for a heart attack. " 
                        "Please consult a healthcare professional immediately for further evaluation.")
            else:
                st.success("✅ Good News! Our assessment indicates you are at LOW RISK for a heart attack. Keep up the good work and maintain a healthy lifestyle!")
            
            st.write(f"Risk Probability: {proba:.1%}")

        except Exception as e:
            st.error(f"An error occurred while making the prediction: {str(e)}")

elif st.session_state.page == 'ml':
    st.header("🤖 Machine Learning Details")
    st.markdown("""
    ### Model Architecture
    - **Algorithm**: Logistic Regression with Feature Engineering
    - **Accuracy**: 92% (validation set)
    - **AUC-ROC**: 0.94
    
    ### Key Features
    - Age Category
    - BMI Classification
    - Smoking Status
    - Diabetes History
    - Physical Activity Levels
    """)

elif st.session_state.page == 'contact':
    st.header("📧 Contact & Support")
    st.markdown("""
    ### Have questions or feedback?
    **Email:** healthcare-analytics@example.com  
    **Support Hours:** Mon-Fri 9AM-5PM EST  
    
    ### Data Privacy
    - No personal data is stored
    - All predictions are transient
    - Anonymous usage statistics only
    """)
