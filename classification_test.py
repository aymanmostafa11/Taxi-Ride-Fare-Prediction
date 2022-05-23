from pickle import load
import pandas as pd
from sklearn import metrics

loaded_preprocessing = load(open('saved_preprocessing', 'rb'))
loaded_models = load(open('saved_models', 'rb'))

taxiRides = pd.read_csv('/content/taxi-test-samples.csv')
weather = pd.read_csv('/content/classification/weather.csv')

# Data Cleaning
weather['rain'].fillna(0,inplace=True)

# Encoding Time
weatherDate = pd.to_datetime(weather['time_stamp'], unit='s').apply(lambda x: x.strftime(('%Y-%m-%d')))
taxiRidesDate = pd.to_datetime(taxiRides['time_stamp'], unit='ms').apply(lambda x: x.strftime(('%Y-%m-%d')))
weather['date'] = weatherDate
taxiRides['date'] = taxiRidesDate

# Joining Dataframes based on date

taxiRides.drop(['time_stamp'],axis = 1, inplace = True)
weather.drop(['time_stamp'],axis = 1, inplace = True)
mergedData = pd.merge(taxiRides,weather.drop_duplicates(subset=['date', 'location']), how = 'left',
                      left_on=['date', 'source'], right_on=['date', 'location'])

# Preprocessing
columnsToDrop = ['id', 'date', 'product_id', 'location']
mergedData.drop(columnsToDrop,axis = 1,inplace=True)
loaded_preprocessing.encode_name(mergedData['name'])
nonIntegerColumns = [col for col in mergedData.columns if mergedData[col].dtypes == object]
loaded_preprocessing.encode_cached(mergedData,nonIntegerColumns)

# Weather Features Engineering
mergedData['rainType'] = 0
mergedData['rainType'][(mergedData['rain'] > 0) & (mergedData['rain'] < 0.1)] = 1
mergedData['rainType'][(mergedData['rain'] > 0.1) & (mergedData['rain'] < 0.3)] = 2
mergedData['sunnyDay'] = 0
mergedData['sunnyDay'][mergedData['clouds'] <= 0.1] = 1

# PCA
subsetOfData = mergedData[['temp','sunnyDay','rainType','wind','pressure','humidity']]
mergedData.drop(['temp','clouds','sunnyDay','rainType','rain','wind','pressure','humidity'],axis=1,inplace=True)
lowerDimensionWeatherData = loaded_preprocessing.reduceDimentionsOf(subsetOfData)
mergedData['weatherState'] = lowerDimensionWeatherData

dataFeatures = mergedData.drop(['RideCategory'],axis=1)
dataLabel = mergedData['RideCategory']

for model in loaded_models:
  print(model)
  print(metrics.f1_score(dataLabel, loaded_models[model].predict(dataFeatures), average='micro'))