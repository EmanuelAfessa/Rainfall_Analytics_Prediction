# RAINWATCH : Rainfall analytics and prediction data science project
## Project goal 
The project showcases data science skills and technologies that can help monitor and predict volatile natural variables. Here we use python to build the back and front ends of a simple web app for rainfall analytics and prediction in the United States.
<img src="assets/rainwatch_screenshot.png"/> 
## Machine learning aspects  
The  machine learning aspects conducted with MLflow library. 
### Set Up
We will setup a conda environment , the main dependencies being mlflow, tensorflow and sklearn. MLflow can be installed using pip. The requirements.yaml file in this repository gives the complete list of set up requirements.


### Monitoring training runs with MLflow Tracking

The experiment starts when we define MLflow context with mlflow.start_run(). The autolog function logs all parameters so it is not necessary to explicitly log parameters. 
The experiment runs are available to be analysed and compared via the following command from the root folder:

``` mlflow ui ``` <br>
Here is an example of a preview with a custom choice of parameters :

![tracking](assets/mlflow_runs.png)
