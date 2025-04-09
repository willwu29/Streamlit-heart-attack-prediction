import streamlit as st
import joblib  
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
        '📝 Heart Attack Assessment': 'predict',
        '🧮 Additional Tools': 'calculators', 
        '📊 Data Insights': 'eda',  # Renamed and rearranged
        '🤖 ML Model': 'ml',
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
            ("1. Assess", "📝 Heart Attack Assessment", "Assess heart attack risk using my ML model"),
            ("2. Validate", "🧮 Additional Tools", "Cross-check your risk with external clinical tools"),
            ("3. Explore", "📊 Data Insights", "Learn more about heart attack through data analysis"),
            ("4. Learn", "🤖 ML Model", "Discover my machine learning model for risk assessment")
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

    # The challenge and solution
    st.markdown("""
    <div style='
        padding: 20px; 
        background-color: #fff3e0; 
        border-radius: 10px;
        margin-bottom: 20px;
    '>
        <h4 style='color: #d32f2f; margin-top:0;'>🚨 The Challenge</h4>
        <ul style='font-size: 14px; padding-left: 20px;'>
            <li>Leading cause of death in the US with over 800k incidents annually -> one heart attack every 40 seconds.</li>
            <li>Heart attack healthcare costs exceed $160 million USD each year.</li>
            <li>Current assessment tools (🧮 Additional Tools) overlook individuals aged 18-29 and 80+.</li>
            <li>Current assessment tools also require blood test inputs, limiting accessibility.</li>
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
        <h4 style='color: #2e7d32; margin-top:0;'>✅ My Solution</h4>
        <ul style='font-size: 14px; padding-left: 20px;'>
            <li>Develop an early detection system utilizing Machine Learning for timely risk assessment.</li>
            <li>Accessible to adults aged 18 and older, ensuring broader coverage.</li>
            <li>Provides instant risk evaluations using easily obtainable input data.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)



# Risk Assessment page
elif st.session_state.page == 'predict':
    col1, spacer, col2 = st.columns([1.2, 0.3, 1.2])

    with col1:
        # Personal Information
        st.header("Personal Info")
        sex = st.selectbox("Gender:", ["Male", "Female"], 
                         key='predict_sex')  # Session key added
        race_ethnicity = st.selectbox("Race/Ethnicity:", 
                                    ["White", "Hispanic", "Black", "Asian", "Multiracial", "Other", "Unknown"],
                                    key='predict_race')  # Session key added
        age_category = st.selectbox("Age Category:", 
                                    ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", 
                                    "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"],
                                    key='predict_age')  # Session key added
        
        # Health Condition
        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
        st.header("Health Status")
        bmi_category = st.selectbox("BMI Category:", 
                                    ["Underweight", "Healthy", "Overweight", "Obese", "Unknown"],
                                    help="Underweight ≤18.4, Healthy 18.5-24.9, Overweight 25.0-29.9, Obese ≥30.0",
                                    key='predict_bmi')  # Session key added
        general_health = st.selectbox("How would you rate your Health Condition:", 
                                    ["Excellent", "Very good", "Good", "Fair", "Poor", "Unknown"],
                                    key='predict_health')  # Session key added
        deaf_or_hard_of_hearing = st.selectbox("Hearing Difficulty:", 
                                               ["No", "Yes", "Unknown"],
                                               help="Are you deaf or do you have serious difficulty hearing?",
                                               key='predict_hearing')  # Session key added
        blind_or_vision_difficulty = st.selectbox("Vision Difficulty:", 
                                                  ["No", "Yes", "Unknown"],
                                                  help="Are you blind or do you have serious difficulty seeing, even when wearing glasses?",
                                                  key='predict_vision')  # Session key added
        difficulty_walking = st.selectbox("Walking & Climbing stairs Difficulty:", 
                                          ["No", "Yes", "Unknown"],
                                          help="Do you have serious difficulty walking or climbing stairs?",
                                          key='predict_walking')  # Session key added
        difficulty_dressing_bathing = st.selectbox("Dressing & Bathing Difficulty:", 
                                                   ["No", "Yes", "Unknown"],
                                                   help="Do you have difficulty dressing or bathing?",
                                                   key='predict_dressing')  # Session key added

    with col2:
        # Habits & Lifestyle
        st.header("Habits & Lifestyle")
        physical_activities = st.selectbox("Any Physical activities in past 30 days:", 
                                         ["No", "Yes", "Unknown"],
                                         key='predict_activities')  # Session key added
        alcohol_drinker = st.selectbox("Any Alcohol consumption in past 30 days:", 
                                     ["No", "Yes", "Unknown"],
                                     key='predict_alcohol')  # Session key added
        smoker_status = st.selectbox("Smoking status:", 
                                   ["Never", "Former", "Every day smoker", "Some days smoker"],
                                   key='predict_smoker')  # Session key added
        
        # Medical History
        st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
        st.header("Medical History")
        had_depressive_disorder = st.selectbox("Depressive disorder diagnosis:", 
                                             ["No", "Yes", "Unknown"],
                                             key='predict_depression')  # Session key added
        had_diabetes = st.selectbox("Diabetes diagnosis:", 
                                   ["No", "Yes", "Pre-diabetes", "Gestational-diabetes", "Unknown"],
                                   key='predict_diabetes')  # Session key added
        had_kidney_disease = st.selectbox("Kidney disease diagnosis:", 
                                         ["No", "Yes", "Unknown"],
                                         help="Not including kidney stones, bladder infection or incontinence",
                                         key='predict_kidney')  # Session key added
        had_angina = st.selectbox("Angina diagnosis:", 
                                 ["No", "Yes", "Unknown"],
                                 help="Angina is chest pain or discomfort caused by reduced blood flow to the heart muscle, often triggered by physical exertion or stress.",
                                 key='predict_angina')  # Session key added
        had_stroke = st.selectbox("Stroke history:", 
                                 ["No", "Yes", "Unknown"],
                                 help="Stroke is a medical emergency that occurs when blood flow to the brain is interrupted, causing brain damage and potentially leading to loss of function such as speech, movement, or memory.",
                                 key='predict_stroke')  # Session key added
        had_copd = st.selectbox("COPD (Chronic Obstructive Pulmonary Disease) diagnosis:", 
                               ["No", "Yes", "Unknown"],
                               help=" COPD is a progressive lung disease characterized by airflow limitation, making it difficult to breathe.",
                               key='predict_copd')  # Session key added
        had_arthritis = st.selectbox("Arthritis diagnosis:", 
                                    ["No", "Yes", "Unknown"],
                                    help="Arthritis is a chronic inflammation of the joints that leads to pain, stiffness, swelling, and reduced mobility, commonly affecting the hands, knees, and hips",
                                    key='predict_arthritis')  # Session key added


    # Handle Unknown values
    bmi_category = "Healthy" if bmi_category == "Unknown" else bmi_category
    physical_activities = "No" if physical_activities == "Unknown" else physical_activities
    had_angina = "No" if had_angina == "Unknown" else had_angina
    had_stroke = "No" if had_stroke == "Unknown" else had_stroke

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
                st.info("💡 **Recommendation:** Validate results using 🧮 Additional Tools")
        
            except Exception as e:
                st.error(f"System error: {str(e)}")
                

# Add the new page handler
elif st.session_state.page == 'calculators':
    st.header("🧮 Additional  Tools")
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
        🔄 Use these clinical tools alongside my app's predictions to cross-validate your risk assessment results.<br>
        💡 Always consult with a healthcare professional about your results.
        </p>
    </div>
    """, unsafe_allow_html=True)




# EDA Section
elif st.session_state.page == 'eda':
    st.header("📊 Insights: Heart Attack Risk Factors")
    
    # Page introduction
    st.markdown("""
    <style>
    .intro-text {
        font-size: 17px;
        line-height: 1.7;
        margin-bottom: 30px;
    }
    </style>
    
    <div class="intro-text">
    This analysis examines heart attack risk patterns using a health survey dataset of 800,000 adults from the CDC Behavioral Risk Factor Surveillance System (BRFSS) for the years 2022 and 2023.
    Below you'll find visualizations showing:
    <ul>
        <li>Prevalence of heart attacks in the population</li>
        <li>Age-related risk progression</li>
        <li>Impact of smoking habits</li>
        <li>Impact of BMI</li>
        <li>Relationship with medical histories such as Angina</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    

    # Load data with caching and error handling
    @st.cache_data
    def load_data():
        file_path = 'data/df.csv'  # Ensure this path is correct
        try:
            df = pd.read_csv(file_path)
            return df
        except FileNotFoundError:
            st.error(f"Error: File not found at {file_path}")
            return None
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None
    
    df = load_data()
    

    # Add some space
    st.markdown("<br>", unsafe_allow_html=True)  # Two line breaks

    # First Plot
    st.markdown("#### Heart Attack Occurrence Distribution")
    try:
        st.image("src/Heart_Attack_Occurrence_Distribution.png", 
                 width=600)
    except FileNotFoundError:
        st.error("Heart attack distribution image not found at src/Heart_Attack_Occurrence_Distribution.png")  # Fixed error message

    # First analysis text
    st.markdown("""
    <style>
    .analysis-text {
        font-size: 16px;
        line-height: 1.6;
        margin-top: 15px;
        margin-bottom: 30px;
    }
    </style>
    
    <div class="analysis-text">
    🔍 <strong>Important Note:</strong> The dataset shows significant class imbalance - only 5.3% of respondents reported heart attacks. 
    This context is crucial when interpreting subsequent visualization and ML Model.
    </div>
    """, unsafe_allow_html=True)

        # Add some space
    st.markdown("<br>", unsafe_allow_html=True)  # Two line breaks

    
    # Second image Age group
    st.markdown("#### Heart Attack Likelihood By Age")
    try:
        st.image("src/heart_attack_age_group.png", 
                 width=1000)
    except FileNotFoundError:
        st.error("Age group distribution image not found at src/heart_attack_age_group.png")

    
    # Second analysis text with bullet points
    st.markdown("""
    <style>
    .bullet-points {
        font-size: 16px;
        line-height: 1.8;
        margin-top: 15px;
        margin-bottom: 30px;
    }
    </style>
    
    <div class="bullet-points">
    🔍 <strong>Key Insights:</strong>
    <ul>
        <li>Heart attack risk increases exponentially after age 45</li>
        <li>Existing models (🧮 Additional Tools) exclude individuals aged 80+, leaving this high-risk group inaccessible to existing early detection tools</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)


    # Add some space
    st.markdown("<br>", unsafe_allow_html=True)  # Two line breaks

    
    # Second image Age group
    st.markdown("#### Heart Attack Likelihood By Gender")
    try:
        st.image("src/heart_attack_gender.png", 
                 width=1000)
    except FileNotFoundError:
        st.error("Gender distribution image not found at src/heart_attack_gender.png")

    
    # Second analysis text with bullet points
    st.markdown("""
    <style>
    .bullet-points {
        font-size: 16px;
        line-height: 1.8;
        margin-top: 15px;
        margin-bottom: 30px;
    }
    </style>
    
    <div class="bullet-points">
    🔍 <strong>Key Insights:</strong>
    <ul>
        <li>Male exhibits higher risk in having a heart attack</li>
           </ul>
    </div>
    """, unsafe_allow_html=True)
    

    # Add some space
    st.markdown("<br>", unsafe_allow_html=True)  # Two line breaks

    # Third plot
    st.markdown("#### Heart Attack Likelihood By Smoking Status")
    try:
        st.image("src/heart_attack_smoker_status.png", width=1000)
    except FileNotFoundError:
        st.error("Heart attack by smoker status image not found at src/heart_attack_smoker_status.png")
    
    # Analysis text for smoking status
    st.markdown("""
    <style>
    .analysis-text {
        font-size: 16px;
        line-height: 1.6;
        margin-top: 15px;
        margin-bottom: 30px;
    }
    </style>
    
    <div class="analysis-text">
    🔍 <strong>Analysis:</strong> Smoking frequency is positively correlated with heart attack likelihood, with 'Every Day Smoker' at highest,  'Never' at lowest.
    </div>
    """, unsafe_allow_html=True)

    # Add some space
    st.markdown("<br>", unsafe_allow_html=True)  # Two line breaks

    # Fourth plot
    st.markdown("#### Heart Attack Likelihood By BMI Category")
    try:
        st.image("src/heart_attack_bmi_category.png", width=1000)
    except FileNotFoundError:
        st.error("Heart attack by BMI category image not found at src/heart_attack_bmi_category.png")
    
    # Analysis text for BMI categories
    st.markdown("""
    <style>
    .analysis-text {
        font-size: 16px;
        line-height: 1.6;
        margin-top: 15px;
        margin-bottom: 30px;
    }
    </style>
    
    <div class="analysis-text">
    🔍 <strong>Analysis:</strong> The likelihood of having a heart attack differs across various BMI categories. The Healthy group shows a notably lower likelihood of experiencing a heart attack, while the Obese group exhibits a higher likelihood.
    </div>
    """, unsafe_allow_html=True)

    
    # Add some space
    st.markdown("<br>", unsafe_allow_html=True)  

    # Fifth plot
    st.markdown("#### Heart Attack Likelihood By Angina")
    try:
        st.image("src/heart_attack_had_angina.png", width=1000)
    except FileNotFoundError:
        st.error("Heart attack by angina image not found at src/heart_attack_had_angina.png")
    
    # Analysis text for Angina
    st.markdown("""
    <style>
    .analysis-text {
        font-size: 16px;
        line-height: 1.6;
        margin-top: 15px;
        margin-bottom: 30px;
    }
    </style>
    
    <div class="analysis-text">
    🔍 <strong>Analysis:</strong> Angina:
    <ul>
        <li>Angina is a type of chest pain or discomfort caused by reduced blood flow to the heart muscle.</li>
        <li>Respondents with angina have a 45.4% likelihood of experiencing a heart attack—nearly 9x higher than the baseline population (5.3%).</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)




# ML Section
elif st.session_state.page == 'ml':
    st.header("🤖 Heart Attack Prediction ML Model")
    
    # Intro section
    st.markdown("""
    🌟 **Clinical Early Detection System**  
    This section showcases how machine learning can effectively assess the risk of imminent heart attacks, achieving an identification rate of 80% through a risk assessment questionnaire. It also details the processes involved in training and evaluating the models.

    💻 **Want to learn more about the models?**  
    [![GitHub](https://img.shields.io/badge/GitHub-Repo_Deep_Dive-blue?logo=github)](https://github.com/willwu29/heart-attack-prediction-model)  
    Explore the complete implementation, which includes data collection, preprocessing, exploratory data analysis (EDA), feature engineering, modeling, hyperparameter tuning, model evaluation, and insights into interpretation and limitations.
    """)

    
    # Project Flowchart
    st.markdown("### End-to-End Project Workflow")
    try:
        st.image("src/project_flowchart.png", width=800)
    except FileNotFoundError:
        st.error("Critical workflow diagram missing: Please ensure 'project_flowchart.png' exists in /src directory")
        st.stop()

    # Visual spacing
    st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
    
    st.markdown("""
    ### Success Metrics
    - **🚨 Accuracy Paradox**: Accuracy metric is deceptive for our imbalanced dataset (5.3% positive cases). A classifier predicting all negatives would achieve 94.7% accuracy but wouldn't identify any positive cases.
    - **🎯 Primary Objective**: Maximize Recall (Sensitivity) to identify ≥80% of true high-risk patients. Recall measures the model's ability to identify true positives, crucial for timely intervention.
    - **⚖️ Secondary Control**: Maintain False Positive Rate as low as possible to prevent system from excessive false positive alarms.  High recall can lead to excessive false positives, incorrectly flagging low-risk individuals as high-risk, resulting in unnecessary consultations and resource waste, ultimately undermining the model's
    """)

    # Visual spacing
    st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
    
    st.markdown("""
    ### Modeling 
    - **🔍 Trained Models**: Logistic Regression, Naive Bayes, Decision Tree, Random Forest, XGBoost, Neural Network
    - **✅ Final Model**: Logistic Regression (Superior recall-FPR balance + clinical interpretability)
    """)

    
    # Visual spacing
    st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)

    # Recall Visualization
    st.markdown("### Recall Performance on Test Data")
    try:
        st.image("src/recall_scores.png", width=1000)
    except FileNotFoundError:
        st.error("Critical visualization missing: Please ensure 'recall_scores.png' exists in /src directory")
        st.stop()  # Halt execution if key visual missing
    
    # Enhanced analysis with clinical context
    st.markdown("""
    <style>
    .clinical-insight {
        border-left: 4px solid #ff4b4b;
        padding-left: 1rem;
        margin: 1.5rem 0;
    }
    </style>
    
    <div class='clinical-insight'>
    🔍 <strong>Recall Score Comparison:</strong>
    <ul>
        <li>Adjusted decision thresholds to achieve 80% recall score target on training data</li>
        <li>Following the threshold adjustments, recall scores on the test data were calculated to evaluate the model's ability to identify heart attacks in unseen data.</li>
        <li>Logistic Regression and Naive Bayes emerged as the top performers, with test recall scores nearing 0.8. </li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    
    # Visual spacing
    # FPR Analysis
    st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)

    # Recall Visualization
    st.markdown("### False Positive Rate on Test Data")
    try:
        st.image("src/false_positive_rate.png", width=1000)
    except FileNotFoundError:
        st.error("Critical visualization missing: Please ensure 'false_positive_rate.png' exists in /src directory")
        st.stop()  # Halt execution if key visual missing

    # FPR analysis 
    st.markdown("""
    <style>
    .clinical-insight {
        border-left: 4px solid #ff4b4b;
        padding-left: 1rem;
        margin: 1.5rem 0;
    }
    </style>
    
    <div class='clinical-insight'>
    <h4>Model Performance Analysis</h4>
    <ul>
        <li>Adjusted decision thresholds to achieve 80% recall target on training data</li>
        <li>Evaluated test recall scores post-adjustment to verify real-world detection capability</li>
        <li>Logistic Regression and Naive Bayes demonstrated optimal recall-FPR balance</li>
    </ul>
    """, unsafe_allow_html=True)

    
    # Add some space
    st.markdown("<br>", unsafe_allow_html=True)  

    
    # Clinical benefits and limitations   
    st.markdown("""
    ### 📈 Clinical Benefits
    <ul>
        <li>The model effectively identifies almost 80% of users exposed to High Risk for heart attacks. </li>
        <li>If all heart attack patients in U.S gained access to this risk assessment tool, over 650,000 individuals could be identified and take preventive actions annually.</li>

    </ul>
    
    ### ⚠️ Limitations
    <ul>
        <li>20% of Low Risk users receive a false Risk heart attack alert.</li>
        <li>There is unnecessary expenditure and time spent on checkups and examinations for users who are falsely alarmed</li>
    </ul>
    """, unsafe_allow_html=True)


    
    # Add some space
    st.markdown("<br>", unsafe_allow_html=True)  
    

    # Model coefficients interpretations
    st.markdown("### Model Feature Interpretation")
    try:
        st.image("src/model_coefficients.png", 
                 width=1000)
    except FileNotFoundError:
        st.error("Critical interpretation missing: Please ensure 'model_coefficients.png' exists in /src directory")
        st.stop()

    # Clinical (coefficient)interpretation
    st.markdown("""
        #### ⚠️ High Risk Factors: <small>Increase the odds of high risk, indicated by red bars</small>
        <ul>
            <li><strong>Angina Presence:</strong> (8.94x odds increase)</li>
            <li><strong>Stroke Presence:</strong> (2.94x odds increase)</li>
            <li><strong>Age 80+:</strong> (2.15x odds increase)</li>
            <li><strong>Smoker Status:</strong> Everyday Smoker (1.44x odds increase)</li>
        </ul>
    
        #### ✅ Low Risk Factors: <small>Decrease the odds of high risk, indicated by blue bars.</small>
        <ul>
            <li><strong>Age within 18-24:</strong> (0.23x odds reduction)</li>
            <li><strong>Gender is Female:</strong> (0.55x odds reduction)</li>
            <li><strong>Excellent general health condition:</strong> (0.56x odds reduction)</li>
            <li><strong>Race/Ethnicity is Asian:</strong> (0.74x odds reduction)</li>
        </ul>
        """, unsafe_allow_html=True)
    


elif st.session_state.page == 'contact':
    st.header("📧 About Me")
    st.markdown("""
    Hello there! I'm Will Wu. I’m passionate about harnessing the power of data to tackle problems. My journey started as a trader at Morgan Stanley, where I made countless trade execution decisions based on data—this is where my love for data analytics, machine learning, and automation truly took off! 
    
    To further hone my skills, I enrolled in a data science bootcamp at BrainStation. Now, I’m equipped to blend machine learning with my problem-solving, collaboration, and research abilities to tackle complex challenges and create meaningful data-driven solutions. 
    
    My top skills are Python, SQL, Tableau, and Spark. 
    
    Excited to connect and share insights!
    """)
    
    st.markdown("""
    ### Have questions or feedback?
    **Email:** [willwu2912@gmail.com](willwu2912@gmail.com)  
    **LinkedIn:** [Will Wu](https://www.linkedin.com/in/willwu2912/)  
    **GitHub:** [![GitHub](https://img.shields.io/badge/GitHub-blue?logo=github)](https://github.com/willwu29)
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


