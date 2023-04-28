# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 23:31:15 2023

@author: KIIT
"""

import numpy as np 
import pickle
import streamlit as st

loaded_model = pickle.load(open('C:/Users/KIIT/Documents/1 My Documents/Projects/ML Deploy Heart Disease Model/trained_model.sav', 'rb'))

#creating a function for prediction

def heart_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diagnosed with Heart Disease'
    else:
      return 'The person is diagnosed with Heart Disease'
  
    
def main():
    
    #giving a title for the web page
    st.title('Heart Disease Prediction Web App')
    
    #getting the input data from the user
    
    age = st.text_input('Age of the Person')
    sex = st.text_input('Gender of the Person (For M-1 & F-0)')
    cp = st.text_input('Chest Pain Type')
    trestbps = st.text_input('Peak Exercise Blood Pressure Score')
    chol = st.text_input('Cholestrol Level')
    fbs = st.text_input('Fasting Blood Sugar Level')
    restecg = st.text_input('Resting Electrocardiography Level')
    thalach = st.text_input('Maximum Achieved Heart Rate')
    exang = st.text_input('Exercise-Induced Angina')
    oldpeak = st.text_input('ST Depression Induced by Exercise to Rest Level')
    slope = st.text_input('Peak Exercise ST Segment')
    ca =  st.text_input('Calcium Score')
    thal = st.text_input('Thalassesmia Level')
    
    #code for Prediction
    diagnosis = ''
    
    #creating a button for Prediction
    
    if st.button('Heart Test Result'):
        diagnosis = heart_prediction([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
        
    
    st.success(diagnosis)
    
    
if __name__ == '__main__':
    main()









