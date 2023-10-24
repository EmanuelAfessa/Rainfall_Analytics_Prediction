##############################################################
# Script_goal : train models for rainwatch project
# Date : 19/10/2023
##############################################################

###############
# imports
###############

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential

# fetch models from models python file 
from models import *

import os
import warnings
import sys
import logging 
import mlflow, mlflow.sklearn, mlflow.keras

import pickle




# compute metrics 
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2 

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    ###############
    # import data

    ###############
    #path_par=os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    #path_model = os.path.join(path_par, 'backend/model', 'val_lin_reg_model.pkl')
    data_model = pd.read_csv('data/map_cities.csv')
    df = pd.read_csv('data/proc_data/weatherwcity.csv')
    print("data import ok")

    ####################
    # Feature selection
    ####################
    columns = ['TempAvg', 
        'WindSpeed', 
        'Precipitation', 
              ]

    df = df[columns]

    # normalize
    df = (((df-df.min()))/(df.max()-df.min()))


    X = df.iloc[:, 0:2]
    y = df.iloc[:, 2:]


    ####################
    # split data
    ####################

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    ###################################
    # Select and train model 
    ##################################

    #mlflow.set_experiment("rainwatch_classic_ML")
    mlflow.set_experiment("rain_deep learning")
    with mlflow.start_run():
        model = nn_mlp1_model()       


        # for neural network
        mlflow.keras.autolog(registered_model_name="rain_dl_predictor")
        nb_epochs = 50
        model.fit(X_train , y_train,epochs= nb_epochs, batch_size=64, verbose=1)
        
         

        rain_predictions=model.predict(X_test)
        (rmse, mae, r2) = eval_metrics(y_test, rain_predictions)
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)
      
 

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        #model_save_path = r'D:/archives_2023/house_price_prediction/models/model.pkl'
        model_save_path = r'models/deeplearning/dl_rainwatch.pkl'
        print("model save ok")

        pickle.dump(model, open(model_save_path, 'wb'))