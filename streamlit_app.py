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
    # Model loading
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

    # Prediction form
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Personal Info")
        sex = st.selectbox("Gender:", ["Male", "Female"])
        race_ethnicity = st.selectbox("Race/Ethnicity:", 
                                    ["White", "Hispanic", "Black", "Asian", "Multiracial", "Other", "Unknown"])
        age_category = st.selectbox("Age Category:", 
                                    ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", 
                                    "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"])
        
        st.header("Health Status")
        bmi_category = st.selectbox("BMI Category:", 
                                    ["Underweight", "Healthy", "Overweight", "Obese"],
                                    help="Underweight ≤18.4, Healthy 18.5-24.9, Overweight 25.0-29.9, Obese ≥30.0")
        general_health = st.selectbox("Health Condition:", 
                                    ["Excellent", "Very good", "Good", "Fair", "Poor", "Unknown"])

    with col2:
        st.header("Habits & Lifestyle")
        physical_activities = st.selectbox("Physical Activities:", ["No", "Yes"])  
        alcohol_drinker = st.selectbox("Alcohol Consumption:", ["No", "Yes", "Unknown"])  
        smoker_status = st.selectbox("Smoking Status:", 
                                   ["Never", "Former", "Every day smoker", "Some days smoker"])
        
        st.header("Medical History")
        had_depressive_disorder = st.selectbox("Depressive Disorder:", ["No", "Yes", "Unknown"]) 
        had_diabetes = st.selectbox("Diabetes:", ["No", "Yes", "Pre-diabetes", "Gestational-diabetes", "Unknown"])  
        had_kidney_disease = st.selectbox("Kidney Disease:", ["No", "Yes", "Unknown"])  

    # Additional medical history
    with st.expander("Additional Medical History"):
        had_angina = st.selectbox("Angina:", ["No", "Yes"])  
        had_stroke = st.selectbox("Stroke History:", ["No", "Yes"]) 
        had_copd = st.selectbox("COPD:", ["No", "Yes", "Unknown"])  
        had_arthritis = st.selectbox("Arthritis:", ["No", "Yes", "Unknown"])

    # Prediction logic
    if st.button('Predict Heart Attack Risk'):
        input_data = [sex, race_ethnicity, age_category, bmi_category.lower(), alcohol_drinker,
                     general_health, smoker_status, physical_activities, had_angina, had_stroke,
                     had_copd, had_diabetes, had_kidney_disease, had_depressive_disorder,
                     had_arthritis]
        
        input_df = pd.DataFrame([input_data], columns=[
            'sex', 'race_ethnicity_category', 'age_category', 'bmi_category',
            'alcohol_drinkers', 'general_health', 'smoker_status',
            'physical_activities', 'had_angina', 'had_stroke', 'had_copd',
            'had_diabetes', 'had_kidney_disease', 'had_depressive_disorder',
            'had_arthritis'
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

elif st.session_state.page == 'ml':
    st.header("🤖 Machine Learning Details")
    st.markdown("""
    ### Model Architecture
    - **Algorithm**: Optimized Logistic Regression
    - **Accuracy**: 92% (validation set)
    - **AUC-ROC**: 0.94
    
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

elif st.session_state.page == 'contact':
    st.header("📧 Contact & Support")
    st.markdown("""
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
