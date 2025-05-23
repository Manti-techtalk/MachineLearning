import streamlit as st
import numpy as np
from joblib import load


model = load('heart.joblib')

st.title("❤️Heart Disease Prediction")

st.write("This is a simple web app to predict heart disease using machine learning.")
st.write("Please enter the following information:")

age = st.number_input("Age", min_value=0, max_value=120, value=25)
ca = st.number_input("CA", min_value=0, max_value=4, value=0)
chol = st.number_input("Chol", min_value=0, max_value=600, value=200)   

if st.button("Predict"):
    input_data = np.array([age, ca, chol])
    input_data = input_data.reshape(1, -1)
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.success("You have heart disease.")
    else:
        st.success("You do not have heart disease.")
st.write("Thank you for using the app!")
st.write("Please enter your details to get a prediction.")
st.write("This app uses a machine learning model to predict heart disease based on your age, CA, and cholesterol levels.")
st.write("The model was trained on a dataset of heart disease patients and uses a Random Forest Classifier to make predictions.")