# RAINWATCH : Rainfall analytics and prediction end to end ML project
## Project goal 
The project showcases data science skills and technologies that can help monitor and predict volatile natural variables. Here we use python to build the back and front ends of a **simple web app for rainfall analytics and prediction** in the United States.
<img src="assets/rainwatch_screenshot.png"/> 
## Machine learning aspects  
The  machine learning aspects conducted with MLflow library.

### Set Up
We will setup a **conda environment** , the main dependencies being dash, mlflow, tensorflow and sklearn. MLflow can be installed using pip. The requirements.txt file in this repository gives the complete list of set up requirements.

### Data

The data has **natural dependent variables** like wind speed, wind direction, average temperature. The independent variable we want to **forecast is the level of precipitation in inches on a given date**. 

### Understanding the prediction problem 
**Time series data is data collected on the same subject at different points in time**. Time Series Forecasting real use cases include  forecasting of 
* sales for stock management
* server ressource usage 
* epidemic outbreaks and health concerns
* stock prices etc...

Time series forecasting can be univariate or multivariate. The methods used can be classic statitics like ARIMA and SARIMA or Machine learning methods like XGBoost, Random Forest or RNN, LSTM tec...

### Training models
Use the training scripts in the root folder. With **classic_ML_train.py** you options of non deep learning models like linear regression, XGBRegressor  etc... The **deep_ML_train.py** presents different architectures of neural networks. All models are and imported to the training scripts. After training models are saved in pickle format 
in **models.py**
 ![training](assets/archi.png) 


### Monitoring training runs with MLflow Tracking

The experiment starts when we define MLflow context with **mlflow.start_run()**. An experiment multiple runs ; each with different parameters. The autolog function logs all parameters so it is not necessary to explicitly log parameters. 
The **experiment runs** are available to be analysed and compared via the following command from the root folder:

``` mlflow ui ``` <br>
Here is an example of a preview with a custom choice of parameters :

![tracking](assets/mlflow_runs.png)

### Using libraries specific to time series forecasting 
#### Pycaret 
We use the **setup from pycaret.ts** to create a specific **time series experiment with automated training runs.** 
This is launched with 
``` best = compare_models(sort = 'MAE') ``` <br>
