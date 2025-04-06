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
        padding: 0.4rem 1rem !important;  /* Reduced vertical padding */
        margin: 0.1rem 0 !important;      /* Reduced vertical margin */
        background-color: #f0f2f6 !important;  /* Match sidebar color */
        border: none !important;
        color: #333 !important;
        border-radius: 4px !important;
        transition: all 0.2s !important;
    }
    .stButton>button:hover {
        background-color: #e6e8ec !important;  /* Slightly darker hover */
        color: #FF5733 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    /* Remove space between buttons */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column"] > [data-testid="stVerticalBlock"] {
        gap: 0rem !important;
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



# Welcome Page
if st.session_state.page == 'welcome':
    # Title Section
    st.markdown("<h1 style='font-size: 48px; text-align: center; color: #FF5733;'>❤️ Heart Attack Risk Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #555;'>Predict Your Risk in Minutes</h3>", unsafe_allow_html=True)
    
    # How It Works - Visual Guide
    st.markdown("## 🚀 How It Works")
    with st.container():
        cols = st.columns(4)
        box_style = """
            padding: 15px; 
            border-radius: 10px; 
            background-color: #f8f9fa; 
            min-height: 200px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        """
        
        step_content = [
            ("1. Assess", "📝 Risk Assessment", "Assess heart attack risk using my ML model"),
            ("2. Validate", "🧮 Calculators", "Cross-check your risk with external clinical tools"),
            ("3. Explore", "📊 Data Analysis", "Learn more about heart attack statistics"),
            ("4. Learn", "🤖 ML Insights", "Discover my machine learning model")
        ]

        for i, (step_num, title, text) in enumerate(step_content):
            with cols[i]:
                st.markdown(f"""
                <div style='{box_style}'>
                    <div>
                        <h3 style='color: #FF5733; margin:0; font-size: 24px;'>{step_num}</h3>
                        <p style='font-size: 32px; margin: 10px 0;'>{title.split()[0]}</p>
                    </div>
                    <p style='font-size: 14px; margin:0; line-height: 1.4;'>{text}</p>
                </div>
                """, unsafe_allow_html=True)

    # Problem & Solution Section - Vertical Layout
    st.markdown("## 📌 Why This Matters")
    
    # Challenge Section
    st.markdown("""
    <div style='
        padding: 20px; 
        background-color: #fff3e0; 
        border-radius: 10px;
        margin-bottom: 20px;
    '>
        <h4 style='color: #d32f2f; margin-top:0;'>🚨 The Challenge</h4>
        <ul style='font-size: 14px; padding-left: 20px;'>
            <li>Leading cause of US deaths (800k+ annually)</li>
            <li>Existing tools exclude 18-29 age group</li>
            <li>Blood test requirements limit access</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Solution Section
    st.markdown("""
    <div style='
        padding: 20px; 
        background-color: #e8f5e9; 
        border-radius: 10px;
        margin-bottom: 20px;
    '>
        <h4 style='color: #2e7d32; margin-top:0;'>✅ Our Solution</h4>
        <ul style='font-size: 14px; padding-left: 20px;'>
            <li>No lab tests required</li>
            <li>Available for adults 18+</li>
            <li>Instant risk assessment</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)



# Risk Assessment page
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
        had_stroke = st.selectbox("Stroke history:", 
                                  ["No", "Yes"],
                                 help="Stroke is a medical emergency that occurs when blood flow to the brain is interrupted, causing brain damage and potentially leading to loss of function such as speech, movement, or memory.") 
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


    # Click the button to predict
       # Click the button to predict
    st.markdown("""
        <div style="background-color: #FFF3E0; padding: 20px; border-radius: 10px; margin: 25px 0; border-left: 5px solid #FF5733;">
            <h4 style="color: #FF5733; margin-bottom: 15px;">🚨 Ready to Check Your Risk?</h4>
            <p style="color: #555; margin-bottom: 0;">
            "Click the 'Predict' to learn your heart attack risk."
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Add centered container for button
    col1, col2, col3 = st.columns([1, 10, 1])
    with col2:
        if st.button('Predict', 
                    use_container_width=True,
                    help="Analyze your risk factors",
                    type="primary"):
            try:
                threshold = model.named_steps['logreg'].threshold
                proba = model.predict_proba(input_df)[0][1]
                prediction = 'High Risk' if proba >= threshold else 'Low Risk'
                
                st.subheader('Results')
                if prediction == 'High Risk':
                    st.error("""⚠️ **Critical Warning** ⚠️  
                            Our analysis shows **HIGH RISK** of heart attack.  
                            Please consult a healthcare professional immediately for further evaluation.""")
                else:
                    st.success("""✅ **Good News** ✅  
                            Our analysis shows **LOW RISK** of heart attack.  
                            Keep up the good work and maintain a healthy lifestyle!""")
                    
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

# EDA Section
elif st.session_state.page == 'eda':
    st.header("📊 Exploratory Data Analysis")
    st.subheader("Key Insights from Health Data")
    st.write("Coming Soon: Interactive visualizations...")


# ML Section
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


