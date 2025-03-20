import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import streamlit as st
import pandas as pd
from streamlit_navigation_bar import st_navbar
import sys
import os

import appLSTM
import main 

st.sidebar.markdown(
    """
    <style>
        /* Sidebar background and text styling */
        [data-testid="stSidebar"] {
            background-color: #2E4053;
            color: white;
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: white;
        }
        [data-testid="stSidebar"] .css-1v0mbdj {
            color: white;
        }
        /* Sidebar text alignment */
        .css-1aumxhk {
            text-align: center;
        }
        /* Sidebar buttons */
        .stRadio > label {
            font-size: 18px !important;
            color: white !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ✅ Sidebar Branding
st.sidebar.markdown(
    """
    <div style="text-align:center">
        <h2 style="color:white;">ML Dashboard</h2>
        <p style="color:#D4E6F1;">Navigate through different AI tools</p>
    </div>
    <hr style="border-top: 1px solid white;">
    """,
    unsafe_allow_html=True
)

# ✅ Sidebar Navigation Menu
st.sidebar.title("🔍 Select a Tool")
page = st.sidebar.radio("Go to", ["Home", "Customer Churn Prediction", "Sentiment Analysis", "Word Prediction"])

# ✅ Sidebar Footer
st.sidebar.markdown(
    """
    <hr style="border-top: 1px solid white;">
    <p style="text-align:center; color:white;">Made with ❤️ by Vimoh Sharma</p>
    """,
    unsafe_allow_html=True
)

model = tf.keras.models.load_model('model.h5')

with open('scaler.pkl','rb')as file:
    scaler=pickle.load(file)
with open('label_encoder_gender.pkl','rb')as file:
    label_encoder_gender=pickle.load(file)
with open('one_hot_encoder.pkl','rb')as file:
    one_hot_encoder=pickle.load(file)



## Streamlitt
if page == "Home":
    st.title("Welcome to the Machine Learning App")
    st.write("Select a tool from the navigation bar.")
elif page == "Customer Churn Prediction":
    st.title('Customer Churn Prediction')


        #user i/p
    geography = st.selectbox('Geography', one_hot_encoder.categories_[0])
    gender = st.selectbox('Gender', label_encoder_gender.classes_)
    age = st.slider('Age', 18, 92)
    balance = st.number_input('Balance')
    credit_score = st.number_input('Credit Score')
    estimated_salary = st.number_input('Estimated Salary')
    tenure = st.slider('Tenure', 0, 10)
    num_of_products = st.slider('Number of Products', 1, 4)
    has_cr_card = st.selectbox('Has Credit Card', [0, 1])
    is_active_member = st.selectbox('Is Active Member', [0, 1])

    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    geo_encoded=one_hot_encoder.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded,columns=one_hot_encoder.get_feature_names_out(['Geography']))

    input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

    input_scaled=scaler.transform(input_data)

    prediction=model.predict(input_scaled)
    predi_prob=prediction[0][0]


    st.write(predi_prob)
    if(predi_prob>0.5):
        st.write('The customer is likely to churn')
    else:
        st.write('The customer is not likely to churn')

elif page == "Sentiment Analysis": # Import `main.py` when "Sentiment Analysis" is selected
    main.run()
elif page == "Word Prediction":
    appLSTM.run()