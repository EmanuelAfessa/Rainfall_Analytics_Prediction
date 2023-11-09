# RAINWATCH : Rainfall analytics and prediction end to end ML project
## Project goal 
The project showcases data science skills and technologies that can help monitor and predict volatile natural variables. Here we use python to build the back and front ends of a **simple web app for rainfall analytics and prediction** in the United States.
<img src="assets/rainwatch_screenshot.png"/> 
## Machine learning aspects  
The  machine learning aspects conducted with MLflow, sklearn and keras library.

### A. Set Up
We will setup a **conda environment** named rainenv for example with python 3.9 version
```conda create --name rainenv python=3.9``` 
 Then we install via pip libraries such as pandas, numpy , dash, mlflow, tensorflow and sklearn etc . The requirements.txt file in this repository gives the complete list of set up requirements.

### B. Data
**Time series data is data collected on the same subject at different points in time**.Here the independent variable we want to **forecast is the level of precipitation in inches on a given date**. The data has **natural dependent variables** like **wind speed, wind direction, average temperature,etc**.  


### C. Understanding the prediction problem 


Time series forecasting can be univariate or multivariate. The methods used can be : 
* Statitical methods  like ARIMA, SARIMA etc...
* Classic Machine learning methods like XGBoost, Random Forest 
* Deep Learning methods like RNN, LSTM,etc ...
There are also diffrent ways of using these methods :
* Direct use of the generic python machine learning libraries like sklearn and keras
* Using python libaries specific to time series manipulation like Darts

### D. Generic machine learning pipeline with py scripts and MLFlow

Use the training scripts in the root folder. With **classic_ML_train.py** you have options of non deep learning models like linear regression, XGBRegressor  etc... The **deep_ML_train.py** presents different architectures of neural networks. All models are imported to the training scripts. Afterwards, training models are saved in pickle format 
in **models** format
 ![training](assets/archi.png) 

We monitor the training with MLFlow. In MLFLow, A machine learning experiment contains multiple runs. Each run attempts to answer the prediction problem by minimizing a loss metric. To have standardized, comparable runs, the loss metric in one experiment stays the same. Thus, **the importance of choosing the appropriate metric for our specific prediction problem.**

 ![monitoring](assets/mlflow_runs.png) 

# E. Libraries specific to time Series : DARTS 

#### E.1. Choosing a loss metric

#### E.2. Implementing a training tracking system

#### E.3. Analysing the results

#### E.4. Improving the results

##### Improving the data processing 

##### Tuning the existing models 

##### Using new better suited models and libraries for this specific ML problem 
