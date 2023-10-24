# RAINWATCH : Rainfall analytics and prediction end to end ML project
## Project goal 
The project showcases data science skills and technologies that can help monitor and predict volatile natural variables. Here we use python to build the back and front ends of a **simple web app for rainfall analytics and prediction** in the United States.
<img src="assets/rainwatch_screenshot.png"/> 
## Machine learning aspects  
The  machine learning aspects conducted with MLflow library.

### A. Set Up
We will setup a **conda environment** named rainenv for example with python 3.9 version
```conda create --name rainenv python=3.9``` 
 Then we install via pip libraries such as pandas, numpy , dash, mlflow, tensorflow and sklearn etc ML. The requirements.txt file in this repository gives the complete list of set up requirements.

### B. Data

The data has **natural dependent variables** like wind speed, wind direction, average temperature. The independent variable we want to **forecast is the level of precipitation in inches on a given date**. 

### C. Understanding the prediction problem 
**Time series data is data collected on the same subject at different points in time**. Time Series Forecasting real use case examples : 
* The Dow Jones Industrial Average index prices
* The temperature in New York City
* The unemployment rate in the USA
* Website traffic through time and similar

Time series forecasting can be univariate or multivariate. The methods used can be classic statitics like ARIMA and SARIMA or Machine learning methods like XGBoost, Random Forest or RNN, LSTM tec...

### D. Training models
Use the training scripts in the root folder. With **classic_ML_train.py** you options of non deep learning models like linear regression, XGBRegressor  etc... The **deep_ML_train.py** presents different architectures of neural networks. All models are and imported to the training scripts. After training models are saved in pickle format 
in **models.py**
 ![training](assets/archi.png) 


### E. Monitoring training runs with MLflow Tracking

The experiment starts when we define MLflow context with **mlflow.start_run()**. An experiment multiple runs ; each with different parameters. The autolog function logs all parameters so it is not necessary to explicitly log parameters. 
The **experiment runs** are available to be analysed and compared via the following command from the root folder:

``` mlflow ui ``` <br>
Here is an example of a preview with a custom choice of parameters :

![tracking](assets/mlflow_runs.png)

### F. Using libraries specific to time series forecasting 

#### Pycaret 
We use the **setup from pycaret.ts** to create a specific **time series experiment with automated training runs.**  Pycaret trains with a great number of models and identifies the best model according to the specified metric. 
This is launched with 
``` best = compare_models(sort = 'MAE') ``` <br>

#### Darts


#### Kats