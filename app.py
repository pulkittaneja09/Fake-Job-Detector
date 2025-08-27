import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import re
from nltk.stem.porter import PorterStemmer

# --- 1. LOAD THE SAVED ARTIFACTS ---

# Load the trained machine learning model
model = joblib.load("model.pkl")

# Load the label encoders
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Load the scaler
scaler = joblib.load("scaler.pkl")

# Load the GloVe embeddings dictionary
with open('glove_embeddings.pkl', 'rb') as f:
    embeddings_index = pickle.load(f)

glove_words = set(embeddings_index.keys())

# --- 2. DEFINE THE PREPROCESSING FUNCTIONS ---
# These functions must be IDENTICAL to the ones you used for training

def final_preprocess(text):
    text = text.replace('\\r', ' ')
    text = text.replace('\\"', ' ')
    text = text.replace('\\n', ' ')
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
    # Stopwords are not used in this function in your notebook, so I am keeping it consistent
    text = text.lower()
    ps = PorterStemmer()
    text = ps.stem(text)
    return text

def convert_sen_to_vec(sentence):
    vector = np.zeros(300)
    cnt_words = 0
    for word in sentence.split():
        if word in glove_words:
            vector += embeddings_index[word]
            cnt_words += 1
    if cnt_words != 0:
        vector /= cnt_words
    return vector

# --- 3. CREATE THE PREDICTION FUNCTION ---

def make_prediction(features):
    # Create a DataFrame from the user input
    df = pd.DataFrame([features])

    # Preprocess the text data
    df['text'] = df[['title', 'department', 'company_profile', 'description', 'requirements', 'benefits']].apply(lambda x: ' '.join(x), axis=1)
    df['text'] = df['text'].apply(final_preprocess)
    
    # Convert text to GloVe vectors
    text_vector = df['text'].apply(convert_sen_to_vec)
    text_vector_df = pd.DataFrame(text_vector.tolist())

    # Preprocess the categorical features
    for col, le in label_encoders.items():
        # Use lambda to handle unseen values gracefully
        df[col] = df[col].apply(lambda s: le.transform([s])[0] if s in le.classes_ else -1) # -1 for unknown

    # Preprocess the numerical features
    numerical_cols = ['required_education', 'required_experience', 'employment_type']
    
    # Map string inputs to the integer values the model expects before scaling
    exp_map = {'Internship': 0, 'Not Applicable': 1, 'Entry level': 2, 'Associate': 3, 'Mid-Senior level': 4, 'Director': 5, 'Executive': 6}
    edu_map = {'Unspecified':0, 'High School or equivalent':1, 'Some College Coursework Completed':2, 'Vocational':3, 'Associate Degree':4, 'Bachelor\'s':5, 'Master\'s Degree':6, 'Professional':7, 'Doctorate':8, 'Some High School Coursework':9}
    emp_map = {'Other':0, 'Full-time':1, 'Part-time':2, 'Contract':3, 'Temporary':4}

    df['required_experience'] = df['required_experience'].map(exp_map).fillna(-1).astype(int) # Use fillna for safety
    df['required_education'] = df['required_education'].map(edu_map).fillna(-1).astype(int)
    df['employment_type'] = df['employment_type'].map(emp_map).fillna(-1).astype(int)

    df[numerical_cols] = scaler.transform(df[numerical_cols])

    # Combine all features, ensuring correct column order
    # The order must match the training data: 300 GloVe features + other features
    other_features_df = df.drop(columns=['text','title','department','company_profile','description','requirements','benefits'])
    
    # This is a robust way to ensure column order. You might need to adjust based on the exact columns from your notebook.
    # From your notebook: X = data.iloc[:, :-1], Y = data.iloc[:, -1]
    # The order of X.columns is what matters. Let's assume the LabelEncoder columns come first.
    final_features_df = pd.concat([text_vector_df, other_features_df], axis=1)
    final_features_df.columns = final_features_df.columns.astype(str)
    # Make prediction
    prediction = model.predict(final_features_df)
    probability = model.predict_proba(final_features_df)

    return prediction, probability


# --- 4. BUILD THE STREAMLIT UI ---

st.title("Fake Job Posting Detector 🕵️")
st.write("Enter the job posting details below to predict if it's fraudulent.")

# Create input fields for all the features your model needs
title = st.text_input("Job Title")
department = st.text_input("Department", "e.g., Engineering, Sales")
company_profile = st.text_area("Company Profile")
description = st.text_area("Job Description")
requirements = st.text_area("Requirements")
benefits = st.text_area("Benefits")

telecommuting = st.selectbox("Telecommuting?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
has_company_logo = st.selectbox("Has Company Logo?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
has_questions = st.selectbox("Has Questions?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# For other categorical features, you should use the actual string values
employment_type = st.selectbox("Employment Type", ['Full-time', 'Part-time', 'Contract', 'Temporary', 'Other'])
required_experience = st.selectbox("Required Experience", ['Internship', 'Not Applicable', 'Entry level', 'Associate', 'Mid-Senior level', 'Director', 'Executive'])
required_education = st.selectbox("Required Education", ['Unspecified', 'High School or equivalent', 'Some College Coursework Completed', 'Vocational', 'Associate Degree', 'Bachelor\'s', 'Master\'s Degree', 'Professional', 'Doctorate', 'Some High School Coursework'])
industry = st.text_input("Industry", "e.g., Marketing, Computer Software")
function = st.text_input("Function", "e.g., Engineering, Marketing")


if st.button("Predict"):
    # Collect all inputs into a dictionary
    features = {
        'title': title,
        'department': department,
        'company_profile': company_profile,
        'description': description,
        'requirements': requirements,
        'benefits': benefits,
        'telecommuting': telecommuting,
        'has_company_logo': has_company_logo,
        'has_questions': has_questions,
        'employment_type': employment_type,
        'required_experience': required_experience,
        'required_education': required_education,
        'industry': industry,
        'function': function
    }

    prediction, probability = make_prediction(features)

    if prediction[0] == 1:
        st.error("Prediction: This job posting is likely FRAUDULENT.")
        st.write(f"Confidence: {probability[0][1]*100:.2f}%")
    else:
        st.success("Prediction: This job posting is likely REAL.")
        st.write(f"Confidence: {probability[0][0]*100:.2f}%")