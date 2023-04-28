# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

loaded_model = pickle.load(open('C:/Users/KIIT/Documents/1 My Documents/Projects/ML Deploy Heart Disease Model/trained_model.sav', 'rb'))

input_data = (5,166,72,19,175,25.8,0.587,51,92,88.09,17,121.22,100)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diagnosed with Heart Disease')
else:
  print('The person is diagnosed with Heart Disease')