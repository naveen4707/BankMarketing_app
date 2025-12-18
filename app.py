import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Bank Term Deposit Predictor",
    page_icon="🏦",
    layout="wide"
)

# --- CUSTOM CSS FOR BEAUTIFUL UI ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #004a99;
        color: white;
        font-weight: bold;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-top: 20px;
    }
    .success-card { background-color: #28a745; }
    .danger-card { background-color: #dc3545; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ASSETS ---
@st.cache_resource
def load_model_assets():
    with open('bank_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model_columns.pkl', 'rb') as f:
        columns = pickle.load(f)
    return model, columns

try:
    model, model_columns = load_model_assets()
except FileNotFoundError:
    st.error("⚠️ Error: 'bank_model.pkl' or 'model_columns.pkl' not found. Please ensure they are in the same folder.")
    st.stop()

# --- HEADER ---
st.title("🏦 Bank Term Deposit Prediction")
st.markdown("Predict if a customer will subscribe to a term deposit based on campaign demographics and history.")
st.divider()

# --- INPUT FORM ---
with st.container():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Client Profile")
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        job = st.selectbox("Job", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed'])
        marital = st.selectbox("Marital Status", ['married', 'single', 'divorced'])
        education = st.selectbox("Education", ['primary', 'secondary', 'tertiary'])

    with col2:
        st.subheader("💰 Financial Status")
        balance = st.number_input("Average Yearly Balance (EUR)", value=1000)
        housing = st.selectbox("Has Housing Loan?", ['no', 'yes'])
        loan = st.selectbox("Has Personal Loan?", ['no', 'yes'])
        default = st.selectbox("Default on Credit?", ['no', 'yes'])

    with col3:
        st.subheader("📞 Last Campaign")
        duration = st.number_input("Contact Duration (seconds)", value=200)
        campaign = st.number_input("Number of Contacts (Current)", min_value=1, value=1)
        previous = st.number_input("Previous Contacts (History)", value=0)
        poutcome = st.selectbox("Previous Outcome", ['unknown', 'other', 'failure', 'success'])
        pdays = st.number_input("Days since last contact (-1 for never)", value=-1)

# --- PREDICTION LOGIC ---
if st.button("🚀 Analyze Subscription Probability"):
    # 1. Create Dataframe from input
    user_input = pd.DataFrame([{
        'age': age, 'job': job, 'marital': marital, 'education': education,
        'default': default, 'balance': balance, 'housing': housing, 
        'loan': loan, 'duration': duration, 'campaign': campaign, 
        'pdays': pdays, 'previous': previous, 'poutcome': poutcome
    }])

    # 2. One-hot encoding (Matching your notebook logic)
    user_encoded = pd.get_dummies(user_input, drop_first=True)

    # 3. Align with training columns (Critical Step)
    # This ensures columns missing in user input but present in training are added as 0
    final_features = pd.DataFrame(columns=model_columns)
    final_features = pd.concat([final_features, user_encoded], axis=0).fillna(0)
    final_features = final_features[model_columns] # Ensure correct order

    # 4. Predict
    prediction = model.predict(final_features)[0]
    prob = model.predict_proba(final_features)[0]

    # --- RESULT UI ---
    st.divider()
    if prediction == 1:
        st.markdown(f"""
            <div class="prediction-card success-card">
                <h2>✅ Client Likely to Subscribe</h2>
                <p>Confidence: {prob[1]:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="prediction-card danger-card">
                <h2>❌ Client Unlikely to Subscribe</h2>
                <p>Confidence: {prob[0]:.2%}</p>
            </div>
            """, unsafe_allow_html=True)

st.sidebar.info("Model Info: This app uses a Decision Tree Classifier trained on the Bank Marketing dataset to optimize marketing efforts.")
