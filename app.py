import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pickle

model = tf.keras.models.load_model('model.h5', compile=False)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
    
with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)
    
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)
    
st.title("Customer Churn Prediction")

geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance", min_value=0.0, value=1000.0, step=100.0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0, step=1000.0)
tenure = st.slider("Tenure", min_value=0, max_value=10, value=5)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=2)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# Create input data dict
input_df = pd.DataFrame([{
    'CreditScore': credit_score,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary
}])

input_df['Gender'] = label_encoder_gender.transform(input_df['Gender'])

geo_encoded = onehot_encoder_geo.transform(pd.DataFrame([[geography]], columns=['Geography'])).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

input_df = pd.concat([input_df.reset_index(drop=True), geo_encoded_df.reset_index(drop=True)], axis=1)

expected_cols = scaler.feature_names_in_
input_df = input_df[expected_cols]

st.write("Input DataFrame before scaling:", input_df)

input_data_scaled = scaler.transform(input_df)

prediction = model.predict(input_data_scaled)
prediction_probability = prediction[0][0]

if prediction_probability > 0.5:
    st.write(f"Prediction: Customer will churn (Probability: {prediction_probability:.2f})")
else:
    st.write(f"Prediction: Customer will not churn (Probability: {prediction_probability:.2f})")
