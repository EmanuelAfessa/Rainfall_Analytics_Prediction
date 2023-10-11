
############################
# Create MDL models 
############################

# import libraries  

import pandas as pd
import pickle
import numpy as np

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.linear_model import LinearRegression

# load data 
data_model = pd.read_csv('data/weather.csv')

# assign X and Y 
Y = data_model['Precipitation']
X = data_model[['TempAvg','WindSpeed']]

print(X)
# trainstest aplit
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2, random_state =5)


# prediction 
model_LR = LinearRegression()
model_LR.fit(X_train, y_train)

y_pred_LR = model_LR.predict(X_test)

with open("models/rain_lin_reg_model.pkl", "wb") as file_handler:
    pickle.dump(model_LR, file_handler)
    
with open("models/rain_lin_reg_model.pkl", "rb") as file_handler:
    loaded_pickle = pickle.load(file_handler)
    
loaded_pickle