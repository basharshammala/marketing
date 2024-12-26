import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load your trained model
with open('marketing_train_model.pkl', 'rb') as m :
    model = pickle.load(m)
with open('encoder.pkl', 'rb') as e :
    encoder = pickle.load(e)
with open('scaler.pkl', 'rb') as s :
    model_tr= pickle.load(s)

# Title of the app
st.title('The Best Ad Campaign for You')

# Add a brief description
st.write("Use this app to get the best recommendations for your ad campaign based on the given parameters.")

# Inputs from the user

# 1. Campaign Name Selector
campaign_name = st.selectbox(
    'Select Campaign Type',
    ['Conversions', 'awareness', 'traffic']
)

# 2. Adset Gender Selector
adsetgender = st.selectbox(
    'Select Adset Gender',
    ['All', 'Male', 'Female']
)

# 3. Adset Type Selector
adsetType = st.selectbox(
    'Select Adset Type',
    ['Retarget', 'Lookalike', 'Public', 'Cust']
)

# 4. Adset Location Selector
adsetlocation = st.selectbox(
    'Select Adset Location',
    ['Saudi Arabia', 'Riyadh', 'Cities']
)

# 5. Ad Type Selector
ad_type = st.selectbox(
    'Select Ad Type',
    ['web view', 'Story ad']
)

# 6. Amount Spent Input
amount_spent = st.slider(
    'Amount Spent',
    min_value=0,
    max_value=700,  # You can adjust the max value based on your data
    value=700,  # Default value
    step=1
)
user_data = pd.DataFrame({
    'campaign name': [campaign_name],
    'adsetgender': [adsetgender],
    'adsetType': [adsetType],
    'adsetlocation': [adsetlocation],
    'ad type': [ad_type],
    'amount spent': [amount_spent]
})
# Encode the categorical data
user_data_encoded = encoder.transform(user_data.iloc[:, :-1]).toarray()

# Append the numerical feature (amount_spent)
user_data_encoded = np.append(user_data_encoded, [[amount_spent]], axis=1)

# Reshape for model input
user_data_encoded = user_data_encoded.reshape(1, -1)

# Transform using model_tr
user_data_encoded = model_tr.transform(user_data_encoded)

# Predict with model2
result = model.predict(user_data_encoded)

st.write(f"The predicted number of purchases is: {result[0]}")
