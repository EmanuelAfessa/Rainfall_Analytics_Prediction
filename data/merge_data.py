############################
# Create MDL models 
############################

# import libraries  

import pandas as pd


# load data 
data_weather = pd.read_csv('weather.csv')
data_cities = pd.read_csv('map_cities.csv')

#merge
#df_merged = data_weather.merge(data_cities, how='right')
df_merged = pd.merge(data_weather,data_cities,how="inner",on=['Station.City','Station.State'])

#save
df_merged.to_csv('proc_data/weatherwcity.csv', index = False)

print(df_merged.info())