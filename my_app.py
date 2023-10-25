################################################################
# Script  dash app importing data and models and output graphs
# Date : 25/10/2023
################################################################



##############################################################################################################################################################
#  0. Import Libraries
#
##############################################################################################################################################################

# data importing & processing
import os
import pandas as pd
from datetime import date

# dash & ploting
import dash
from dash import dcc, html, callback, Output, Input
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go

# machine learning and array transforms 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model, tree, neighbors
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
import pickle

##############################################################################################################################################################
#  1. Get Data
#
##############################################################################################################################################################
#df_map = pd.read_csv('data/weather.csv')
df_map = pd.read_csv('data/proc_data/weatherwcity.csv')
usmap_cities = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/us-cities-top-1k.csv")

##############################################################################################################################################################
#  2. declaring parameters & Helper functions 
#
##############################################################################################################################################################

# pram : plotly logo 
plotly_logo = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"

# param : today 
today = date.today()

# compute error for ml model 
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2 

#################################################################################################
#                               1.GET_MODEL
#################################################################################################
path_par=os.path.abspath(os.path.join(os.getcwd(), os.pardir))
path_model = os.path.join(path_par, 'RAINWATCH/models', 'rain_lin_reg_model.pkl')
model = pickle.load(open(path_model, 'rb'))


##############################################################################################################################################################
#   3. Set Layout
#
##############################################################################################################################################################
app = dash.Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}])

# START LAYOUT
app.layout = html.Div([
#------------------------------------> TITLES<-----------------------------
# start and end a div 
    html.Div([
        html.Div([
            html.Img(src=plotly_logo,# app.get_asset_url('rain_img.jpeg'),
                     id='rain-image',
                     style={
                         "height": "60px",
                         "width": "auto",
                         "margin-bottom": "25px",
                     },
                     )

                     
        ],
            className="one-third column",
        ),
        html.Div([
            html.Div([
                html.H3("RainWatch", style={"margin-bottom": "0px", 'color': 'white'}),
                html.H5("Precipitation Analytics & Prediction", style={"margin-top": "0px", 'color': 'white'}),
            ])
        ], className="one-half column", id="title"),

        html.Div([
            html.H6('Last Updated: ' + str(today) + '  00:01 (UTC)',
                    style={'color': 'orange'}),

        ], className="one-third column", id='title1'),

    ], id="header", className="row flex-display", style={"margin-bottom": "25px"}),

#------------------------------------> INDICATORS<-----------------------------
# all indicators in one div 
    html.Div([



        #-> indicator card 1
        html.Div([
                    dcc.Graph(id='indic_wind',
                            config={'displayModeBar': 'hover'}),
                            ], className="card_container two columns"),
        #-> indicator card 2
        html.Div([
                    dcc.Graph(id='indic_temp',
                            config={'displayModeBar': 'hover'}),
                            ], className="card_container two columns"),
        #-> indicator card 3
        html.Div([
                    dcc.Graph(id='indic_act_precip',
                            config={'displayModeBar': 'hover'}),
                            ], className="card_container three columns"),
        
                            
        #-> indicator card 4
        html.Div([
                    dcc.Graph(id='indic_pred_precip',
                            config={'displayModeBar': 'hover'}),
                            ], className="card_container three columns"),

    ]),
#------------------------------------------------------------------------------
#------------------------------------> FILTER BOX<-----------------------------
#------------------------------------------------------------------------------
    html.Div([
        html.Div([

                    html.P('Select State:', className='fix_label',  style={'color': 'white'}),   
                    dcc.Dropdown(id='states',
                                  multi=False,
                                  clearable=True,
                                  value='Florida',
                                  placeholder='Select State',
                    options=usmap_cities['State'].sort_values().unique(),className='dcc_compon'),
                                  #options=[{'label': c, 'value': c}
                                           #for c in (usmap_cities['State'].unique())], className='dcc_compon'),    
                    
                    html.P('Select city:', className='fix_label',  style={'color': 'white'}),

                     dcc.Dropdown(id='cities',
                                  multi=False,
                                  clearable=True,
                                  value='Miami',
                                  placeholder='Select City',
                                  options=[{'label': c, 'value': c}
                                           for c in (df_map['StationCity'].sort_values().unique())], className='dcc_compon'),
                    
                              
                                           
                    html.P('Select date:', className='fix_label',  style={'color': 'white'}),

                     dcc.Dropdown(id='dates',
                                  multi=False,
                                  clearable=True,
                                  value='2016-09-18',
                                  placeholder='Select date',
                                  options=[{'label': c, 'value': c}
                                           for c in (df_map['Date.Full'].unique())], className='dcc_compon'),                                 
             

        ], className="create_container three columns", id="cross-filter-options"),
        html.Div([
                    dcc.Graph(id='bar_chart',
                            config={'displayModeBar': 'hover'}),
                            ], className="create_container five columns"),

        html.Div([
            dcc.Graph(id="scatter_chart")

        ], className="create_container four columns"),

            

        ], className="row flex-display"),


        
#------------------------------------> MAP GRAPH<-----------------------------
  html.Div([
        html.Div([
            dcc.Graph(id="map")], className="create_container1 twelve columns"),

            ], className="row flex-display"),  

# END LAYOUT
], id="mainContainer",
    style={"display": "flex", "flex-direction": "column"})


#################################################################################################
#                               DEV_GRAPHS
#################################################################################################

#######################################################################
# GRAPH 0 : Filter box 
# chained calbbacks
#######################################################################

@app.callback(
    Output('cities', 'options'),
    Input('states', 'value'),
    prevent_initial_call=False
)
def update_output(value):
    return df_map[df_map.StationState == value].StationCity.unique()
#######################################################################
# GRAPH 1 : Average windspeed
# type:indicator
#######################################################################

@callback(
    Output('indic_wind', 'figure'),
    Input('cities', 'value'),
    Input('dates', 'value'),
   

)

def update_graph(cities,dates): 
    df_mapc = df_map.loc[(df_map['StationCity']==cities) & (df_map['Date.Full']==dates)] 
    fig = go.Figure(go.Indicator(
    mode = "number",
    value = df_mapc.WindSpeed.mean(),
    number= { 'font_size':20,'font_color':'orange','suffix':' mph'},
    
    
    title = {'font_size':20,'text': "Average Windspeed"},
    domain = {'x': [0, 1], 'y': [0, 1]}
    ))
    fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    font_color="white",
    height=100,)
    return fig

#######################################################################
# GRAPH 2 : Average Temperature
# type:indicator
#######################################################################

@callback(
    Output('indic_temp', 'figure'),
    Input('cities', 'value'),
    Input('dates', 'value'),
   

)

def update_graph(cities,dates): 
    df_mapc = df_map.loc[(df_map['StationCity']==cities) & (df_map['Date.Full']==dates)] 
    fig = go.Figure(go.Indicator(
    mode = "number",
    value = df_mapc.TempAvg.mean(),
    number= {'font_size':20,'font_color':'orange','suffix':' °F'},
    
    title = {'font_size':20,'text': "Average Temperature"},
    #domain = {'x': [0, 1], 'y': [0, 1]}
    ))
    fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    font_color="white",
    height=100,)
    return fig


#######################################################################
# GRAPH 3 : Actual Precipitation
# type:indicator
#######################################################################

@callback(
    Output('indic_act_precip', 'figure'),
    Input('cities', 'value'),
    Input('dates', 'value'),
   

)

def update_graph(cities,dates): 
    df_mapc = df_map.loc[(df_map['StationCity']==cities) & (df_map['Date.Full']==dates)] 
    fig = go.Figure(go.Indicator(
    mode = "number",
    value = df_mapc.Precipitation.mean(),
    number= {'font_size':20, 'font_color':'orange','suffix':' inches'},
    
    title = {'font_size':20,'text': "Actual Precipitation"},
    #domain = {'x': [0, 1], 'y': [0, 1]}
    ))
    fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    font_color="white",
    height=100,)
    return fig

#######################################################################
# GRAPH 4 : Predicted Precipitation
# type:indicator
#######################################################################

@callback(
    Output('indic_pred_precip', 'figure'),
    Input('cities', 'value')
   

)


def update_graph(cities):
            data_model = df_map.loc[(df_map['StationCity']==cities)] 
            x_values = data_model[['TempAvg','WindSpeed']]  
            #x_values = np.asarray(x_values).reshape(1,-1)       
            pred_precip = model.predict(x_values)[0]
            fig = go.Figure(go.Indicator(
            mode = "number",
            value = pred_precip,
            number= {'font_size':20, 'font_color':'red','suffix':' inches '},
            title = {'font_size':20,'text': "Predicted Precipitation"},
            domain = {'x': [0, 1], 'y': [0, 1]}
))
            fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            height=100,)
            return fig



#######################################################################
# GRAPH 5 : Rainfall trend
# type : bar
#######################################################################

@callback(
    Output('bar_chart', 'figure'),
    Input('cities', 'value')
   
)


def update_graph(cities):
    df_maps = df_map.loc[(df_map['StationCity']==cities)]  
    fig = px.line(df_maps, x= "Date.Full", y="Precipitation",   title='Precipitation Trend')
    fig.update_layout(font_color="white", bargap=0.2,yaxis_title="Amount of precipitation (inches)",xaxis_title="Days" , paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)" )
    return fig

#######################################################################
# GRAPH 6 : Precipitation - temperature correlation
# type : bar
#######################################################################

@callback(
    Output('scatter_chart', 'figure'),
    Input('cities', 'value')
   
)
def update_graph(cities):
    df_maps = df_map.loc[(df_map['StationCity']==cities)]  
    fig = px.scatter(df_maps, y= "WindSpeed", x="TempAvg", size="Precipitation", color="Precipitation",  title='Precipitation Windspeed Temperature correlation')
    fig.update_layout(font_color="white",yaxis_title="Wind speed (mph)",xaxis_title="Temperature(Fa)" , paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)" )
    return fig



#######################################################################
# GRAPH 7 : map
# type : bar
#######################################################################


@callback(
    Output('map', 'figure'),
    Input('states', 'value'),
    Input('dates', 'value')

   
)
def update_graph(states,dates): 
    df_mapc = df_map.loc[(df_map['StationState']==states) & (df_map['Date.Full']==dates)]   
    fig = px.scatter_mapbox(df_mapc, lat="lat", lon="lon", size="Precipitation", color="Precipitation", 
                        color_discrete_sequence=px.colors.sequential.Viridis, zoom=5, height=300)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)


 

