import streamlit as st
import joblib  
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.pipeline import Pipeline

# Custom CSS for sidebar styling
st.markdown("""
<style>
    /* Completely hide sidebar buttons */
    [data-testid="stSidebar"] .stButton>button {
        all: unset !important;
        width: 0 !important;
        height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        opacity: 0 !important;
        position: absolute !important;
        left: -9999px !important;
    }

    /* Main prediction button styling */
    div[data-testid="stVerticalBlock"] > div.stButton > button {
        background-color: #FF4444 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 20px 30px !important;
        font-size: 20px !important;
        font-weight: bold !important;
        border: none !important;
        width: 100% !important;
        margin: 20px auto !important;
        transition: all 0.3s ease !important;
    }

    div[data-testid="stVerticalBlock"] > div.stButton > button:hover {
        background-color: #CC0000 !important;
        transform: scale(1.05) !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'

# Sidebar navigation
with st.sidebar:
    st.markdown("## Navigation")
    
    # Create navigation buttons without a container
    pages = {
        '🏠 Welcome': 'welcome',
        '📝 Risk Assessment': 'predict',
        '🧮 Additional Calculators': 'calculators', 
        '📊 Data Analysis': 'eda',  # Renamed and rearranged
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
        deaf_or_hard_of_hearing = st.selectbox("Hearing Difficulty:", 
                                               ["No", "Yes", "Unknown"],
                                                help="Are you deaf or do you have serious difficulty hearing?")  
        blind_or_vision_difficulty = st.selectbox("Vision Difficulty:", 
                                                  ["No", "Yes", "Unknown"],
                                                  help="Are you blind or do you have serious difficulty seeing, even when wearing glasses?")  
        difficulty_walking = st.selectbox("Walking & Climbing stairs Difficulty:", 
                                          ["No", "Yes", "Unknown"],
                                          help="Do you have serious difficulty walking or climbing stairs?")  
        difficulty_dressing_bathing = st.selectbox("Dressing & Bathing Difficulty:", 
                                                   ["No", "Yes", "Unknown"],
                                                   help="Do you have difficulty dressing or bathing?")

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
        had_kidney_disease = st.selectbox("Kidney disease diagnosis:", 
                                          ["No", "Yes", "Unknown"],
                                         help="Not including kidney stones, bladder infection or incontinence")  
        had_angina = st.selectbox("Angina diagnosis:", 
                                  ["No", "Yes"],
                                 help="Angina is chest pain or discomfort caused by reduced blood flow to the heart muscle, often triggered by physical exertion or stress.")  
        had_stroke = st.selectbox("Stroke history:", ["No", "Yes"]) 
        had_copd = st.selectbox("COPD (Chronic Obstructive Pulmonary Disease) diagnosis:", 
                                ["No", "Yes", "Unknown"],
                               help=" COPD is a progressive lung disease characterized by airflow limitation, making it difficult to breathe.")  
        had_arthritis = st.selectbox("Arthritis diagnosis:", 
                                     ["No", "Yes", "Unknown"],
                                     help="Arthritis is a chronic inflammation of the joints that leads to pain, stiffness, swelling, and reduced mobility, commonly affecting the hands, knees, and hips")

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

    st.markdown("""
    <style>
        /* Custom primary button styling */
        div.stButton > button:first-child {
            background-color: #FF4444 !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 20px 30px !important;
            font-size: 20px !important;
            font-weight: bold !important;
            border: none !important;
            width: 100% !important;
            margin: 20px auto !important;
            display: block !important;
            transition: all 0.3s ease !important;
        }
    
        div.stButton > button:first-child:hover {
            background-color: #CC0000 !important;
            transform: scale(1.05) !important;
        }
    
        div.stButton > button:first-child:active {
            transform: scale(0.95) !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Your existing prediction button code
    if st.button('Predict Heart Attack Risk'):
        try:
            threshold = model.named_steps['logreg'].threshold
            proba = model.predict_proba(input_df)[0][1]
            prediction = 'High Risk' if proba >= threshold else 'Low Risk'
            
            st.subheader('Results')
            if prediction == 'High Risk':
                st.error("""⚠️ **Critical Warning** ⚠️  
                        Our analysis shows **HIGH RISK** of heart attack.  
                        Immediate medical consultation recommended!""")
            else:
                st.success("""✅ **Good News** ✅  
                          Our analysis shows **LOW RISK** of heart attack.  
                          Maintain healthy habits!""")
                
            st.markdown("---")
            st.info("💡 **Recommendation:** Validate results using 🧮 Additional Calculators")
    
        except Exception as e:
            st.error(f"System error: {str(e)}")

    


# Add the new page handler
elif st.session_state.page == 'calculators':
    st.header("🧮 Additional Heart Health Calculators")
    st.markdown("""
    <style>
        .calculator-box {
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 20px;
            margin: 15px 0;
            transition: transform 0.2s;
        }
        .calculator-box:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .calculator-button {
            background-color: #FF5733 !important;
            color: white !important;
            padding: 12px 25px !important;
            border-radius: 25px !important;
            text-decoration: none !important;
            display: inline-block !important;
            margin-top: 10px !important;
            transition: all 0.3s !important;
            border: none !important;
        }
        .calculator-button:hover {
            background-color: #E54A2E !important;
            transform: scale(1.05);
            color: white !important;
        }
    </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="calculator-box">
            <h3 style='color: #FF5733; margin-top: 0;'>AHA PREVENT™ Calculator</h3>
            <p style='color: #666; font-size: 14px;'>By American Heart Association</p>
            <a href="https://professional.heart.org/en/guidelines-and-statements/prevent-calculator" 
               target="_blank" 
               class="calculator-button">
               Access Calculator ➡️
            </a>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="calculator-box">
            <h3 style='color: #FF5733; margin-top: 0;'>ACC ASCVD Risk Estimator Plus</h3>
            <p style='color: #666; font-size: 14px;'>By American College of Cardiology</p>
            <a href="https://tools.acc.org/ascvd-risk-estimator-plus/#!/calculate/estimate/" 
               target="_blank" 
               class="calculator-button">
               Access Calculator ➡️
            </a>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("""
    ---
    <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-top: 20px;'>
        <p style='color: #6c757d; font-size: 14px; margin-bottom: 0;'>
        🔍 These clinical tools are provided by reputable medical organizations and open in new tabs.<br>
        🔄 Use these clinical tools alongside our app's predictions to cross-validate your risk assessment results.<br>
        💡 Always consult with a healthcare professional about your results.
        </p>
    </div>
    """, unsafe_allow_html=True)


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

    ### Data Privacy
    - No personal data is stored
    - All predictions are transient
    - Anonymous usage statistics only
    """)


elif st.session_state.page == 'contact':
    st.header("📧 About Me")
    st.markdown("""
    Hello there! I'm Will Wu. I’m passionate about harnessing the power of data to tackle problems. My journey started as a trader at Morgan Stanley, where I made numerous trade execution decisions based on data—this is where my love for data analytics, machine learning, and automation truly took off! 
    
    To further hone my skills, I enrolled in a data science bootcamp at BrainStation. Now, I’m equipped to blend machine learning with my problem-solving, collaboration, and research abilities to tackle complex challenges and create meaningful data-driven solutions. 
    
    My top skills are Python, SQL, Tableau, and Spark. 
    
    Excited to connect and share insights!
    """)
    
    st.markdown("""
    ### Have questions or feedback?
    **Email:** [willwu2912@gmail.com](willwu2912@gmail.com)  
    **LinkedIn:** [Will Wu](https://www.linkedin.com/in/willwu2912/)  
    """)

    # Optionally, you can provide some markdown description
    st.markdown("""### Access to my CV:""")

    # Create a download button for your CV
    with open('/mount/src/streamlit-heart-attack-prediction/CV-WillWu.pdf', 'rb') as file:
        st.download_button(
            label="Download CV",
            data=file,
            file_name="CV-WillWu.pdf",
            mime="application/pdf"
        )


