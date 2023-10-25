
################################################################
# Script  ML models with divers parameters 
# Date : 25/10/2023
################################################################
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential


# models 

def xgb_model():
    model = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,max_depth = 5, alpha = 10, n_estimators = 10)
    return model

def rf_model():
    model = RandomForestRegressor()
    return model


def nn_mlp1_model():
    #design model
    model = Sequential()
    model.add(Dense(13, input_shape=(2,), kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    #compile model 
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model

def nn_mlp2_model():
    model = Sequential()
    model.add(Dense(200,input_shape=(4,), activation='sigmoid'))
    model.add(Dense(200, activation='sigmoid'))
    model.add(Dense(200, activation='sigmoid'))
    model.add(Dense(1, activation='linear'))
    #compile model 
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    return model

def nn_mlp3_model():
    model = Sequential()
    model.add(Dense(128,input_shape=(4,), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    #compile model 
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    return model
