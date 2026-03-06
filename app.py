# Streamlit UI
#import necessary libraries

import pickle
import streamlit as st 
import pandas as pd
import numpy as np


#loading cleaned dataset
df = pd.read_csv("cleaned_data.csv")
st.set_page_config(page_title="Customer Churn Prediction using Supervised Machine Learning",
                   page_icon="customerchurn.jpg")


#title for the page
st.title("Customer Churn Prediction using Supervised Machine Learning")


#creating sidebar with image and title
with st.sidebar:
    st.title("Customer Churn Prediction")
    st.image("customerchurn.jpg")


#input fields
# trained seq : gender, senior citizen, partner, dependents, Tenure, Photo Service, Multiple Lins, Internet Service , Online Security, Online Backup , Device Protection, Tech Support , Streaming TV, Streaming Movies , Contract , Paperless Billing , Payment Method , Monthly Charges , Total Charges
# Gender input
gender = st.radio("Gender", ["Female","Male"],horizontal=True)
gender = 0 if gender=="Female" else 1

# Senior Citizen input
SeniorCitizen = st.radio("Senior Citizen", ["No","Yes"],horizontal=True)
SeniorCitizen = 1 if SeniorCitizen=="Yes" else 0

# Partner input
Partner = st.radio("Partner", ["No","Yes"],horizontal=True)
Partner = 1 if Partner=="Yes" else 0

# Dependents input
Dependents = st.radio("Dependents", ["No","Yes"],horizontal=True)
Dependents = 1 if Dependents=="Yes" else 0

# Tenure input
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100)

# Phone Service input
PhoneService = st.radio("Phone Service", ["No","Yes"],horizontal=True)
PhoneService = 1 if PhoneService=="Yes" else 0

# Multiple Lines input
multi_options = ["No phone service","No","Yes"]
MultipleLines = st.selectbox("Multiple Lines", multi_options)
MultipleLines = multi_options.index(MultipleLines)

# Internet Service input
internet_options = ["DSL","Fiber optic","No"]
InternetService = st.selectbox("Internet Service", internet_options)
InternetService = internet_options.index(InternetService)

# Online Security input
OnlineSecurity = st.selectbox("Online Security", ["No","Yes","No internet service"])
OnlineSecurity = ["No","Yes","No internet service"].index(OnlineSecurity)

# Online Backup input
OnlineBackup = st.selectbox("Online Backup", ["No","Yes","No internet service"])
OnlineBackup = ["No","Yes","No internet service"].index(OnlineBackup)

# Device Protection input
DeviceProtection = st.selectbox("Device Protection", ["No","Yes","No internet service"])
DeviceProtection = ["No","Yes","No internet service"].index(DeviceProtection)

# Tech Support input
TechSupport = st.selectbox("Tech Support", ["No","Yes","No internet service"])
TechSupport = ["No","Yes","No internet service"].index(TechSupport)

# Streaming TV input
StreamingTV = st.selectbox("Streaming TV", ["No","Yes","No internet service"])
StreamingTV = ["No","Yes","No internet service"].index(StreamingTV)

# Streaming Movies input
StreamingMovies = st.selectbox("Streaming Movies", ["No","Yes","No internet service"])
StreamingMovies = ["No","Yes","No internet service"].index(StreamingMovies)

# Contract input
contract_options = ["Month-to-month","One year","Two year"]
Contract = st.selectbox("Contract Type", contract_options)
Contract = contract_options.index(Contract)

# Paperless Billing input
PaperlessBilling = st.radio("Paperless Billing", ["No","Yes"],horizontal=True)
PaperlessBilling = 1 if PaperlessBilling=="Yes" else 0

# Payment Method input
payment_options = [
"Electronic check",
"Mailed check",
"Bank transfer",
"Credit card"
]
PaymentMethod = st.selectbox("Payment Method", payment_options)
PaymentMethod = payment_options.index(PaymentMethod)

# Monthly Charges input
MonthlyCharges = st.number_input("Monthly Charges")

# Total Charges input
TotalCharges = st.number_input("Total Charges")


# NEW DATA PREPARATION
new_data = np.array([[gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges]])

#columns for predict button
col1,col2=st.columns([1,2])

#model de-serialization 
with open("Logistic_model.pkl","rb") as file:
    model = pickle.load(file)

#Scalar de-serialization
with open("scaler.pkl","rb") as file2:
    scaler = pickle.load(file2)

#encoder de-serialization
with open("encoders.pkl","rb")as file1:
    le = pickle.load(file1)

#Testing against new data
new_data=scaler.transform(new_data)
if col2.button("Customer Churn Prediction"):
    pred =model.predict(new_data)[0]
    if pred==1:
        st.subheader(" Churn")
    else:
        st.subheader("No  Churn")
