import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('regression_model.h5')

# load the encoder and scaler
with open('label_encoder_gen.pkl', 'rb') as file:
    label_encoder_gen = pickle.load(file)

with open('onehot_encoder_geograph', 'rb') as file:
    onehot_encoder_geograph = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# taking user input 
geography = st.selectbox("Geography", onehot_encoder_geograph.categories_[0])
gender = st.selectbox("Gender", label_encoder_gen.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
exited = st.selectbox('Exited', [0,1])
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

## Prepare the input 

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gen.transform([gender])[0]],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember' : [is_active_member],
    'Exited' : [exited],
})

# One-hot encode the 'Geography' feature

geo_encoded = onehot_encoder_geograph.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geograph.get_feature_names_out(['Geography']))

# Combine One-hot encoded geography features with the input dataframe

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# SCale the input data
input_scaled = scaler.transform(input_data)

prediction = model.predict(input_scaled)
prediction_salary = prediction[0][0]

st.write(f'Predicted Estimated Salary: ${prediction_salary:.2f}')
