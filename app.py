import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('rf.pkl')

# Set the title of the web page
st.title(":red[Mall Customers Classes Predictor]")

# Create input widgets for gender, age, annual income, and spending score
gender = st.radio("Select Your Gender", ('Male', 'Female'))
if gender == 'Male':
    gender = 1
if gender == 'Female':
    gender = 0

age = st.slider("Select Your Age", 18, 70, 18)
annual_income = st.slider("Select Annual Income (k$)", 15, 113, 15)
score = st.slider("Spending Score (1-100)", 1, 99, 25)

# Create a button to predict the customer class
if st.button("Predict Customer Class"):
    # Convert the input values into a numpy array
    input_data = np.array([[gender, age, annual_income, score]])
    # Use the loaded model to predict the customer class
    prediction = model.predict(input_data)[0]
    # Display the predicted customer class on the web page
    st.write("Predicted Customer Class:", prediction)
