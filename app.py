
import streamlit as st
import joblib
import pandas as pd
import json
import numpy as np

# Load the trained model and feature columns
model = joblib.load('rf_model.pkl')

with open('feature_columns.json', 'r') as f:
    feature_columns = json.load(f)

# Define the Streamlit app title
st.title('Spotify Churn Prediction')

st.markdown("Enter user details to predict churn probability.")

# Input widgets for features

# Numerical features
age = st.slider('Age', 16, 60, 30)
listening_time = st.number_input('Listening Time (minutes per day)', 10, 300, 150)
songs_played_per_day = st.number_input('Songs Played Per Day', 1, 100, 50)
skip_rate = st.slider('Skip Rate', 0.0, 0.6, 0.3, 0.01)
ads_listened_per_week = st.number_input('Ads Listened Per Week', 0, 50, 5)

# Categorical features
gender = st.selectbox('Gender', ['Female', 'Male', 'Other'])
country = st.selectbox('Country', ['CA', 'DE', 'AU', 'US', 'UK', 'IN', 'FR', 'PK'])
subscription_type = st.selectbox('Subscription Type', ['Free', 'Family', 'Premium', 'Student'])
offline_listening = st.selectbox('Offline Listening', [0, 1])

# Preprocessing function for user input
def preprocess_input(age, listening_time, songs_played_per_day, skip_rate,
                     ads_listened_per_week, offline_listening,
                     gender, country, subscription_type, feature_columns):

    # Create a DataFrame from user inputs
    input_data = {
        'age': age,
        'listening_time': listening_time,
        'songs_played_per_day': songs_played_per_day,
        'skip_rate': skip_rate,
        'ads_listened_per_week': ads_listened_per_week,
        'offline_listening': offline_listening,
        'gender': gender,
        'country': country,
        'subscription_type': subscription_type
    }
    input_df = pd.DataFrame([input_data])

    # One-hot encode categorical features
    categorical_cols = ['gender', 'country', 'subscription_type']
    input_df_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True, dtype=int)

    # Align columns with the training data's feature columns
    final_input = pd.DataFrame(columns=feature_columns)
    for col in feature_columns:
        if col in input_df_encoded.columns:
            final_input[col] = input_df_encoded[col]
        else:
            final_input[col] = 0  # Add missing columns with 0
    
    # Ensure the order of columns matches the training data
    final_input = final_input[feature_columns]

    return final_input

# Prediction button
if st.button('Predict Churn'):
    processed_input = preprocess_input(age, listening_time, songs_played_per_day, skip_rate,
                                       ads_listened_per_week, offline_listening,
                                       gender, country, subscription_type, feature_columns)
    
    # Make prediction
    churn_probability = model.predict_proba(processed_input)[:, 1][0]
    
    st.write(f"Churn Probability: {churn_probability:.2f}")
    if churn_probability > 0.5:
        st.error("This user is likely to churn.")
    else:
        st.success("This user is likely to remain active.")
