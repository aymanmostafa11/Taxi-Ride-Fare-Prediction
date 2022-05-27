from pickle import load
from pyexpat import model
import pandas as pd
from sklearn import metrics
from helpers import PreProcessing , Model


preprocessing = PreProcessing()
loaded_encoders = load(open('saved_regressionEncoders', 'rb'))
loaded_models = load(open('saved_regressionModels', 'rb'))
preprocessing.encoders = loaded_encoders


taxiRides = pd.read_csv('sampleTests/taxi-test-samples-Regression.csv')
weather = pd.read_csv('taxi/weather.csv')

# Data Cleaning
weather['rain'].fillna(0,inplace=True)

# Encoding Timestamps to date

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

# Name Feature
preprocessing.encode_name(mergedData['name'])

# Encoding with previously fit encoders
nonIntegerColumns = [col for col in mergedData.columns if mergedData[col].dtypes == object]
preprocessing.encode_cached(mergedData, nonIntegerColumns)

# Weather Features Engineering

mergedData['rainType'] = 0
mergedData['rainType'][(mergedData['rain'] > 0) & (mergedData['rain'] < 0.1)] = 1
mergedData['rainType'][(mergedData['rain'] > 0.1) & (mergedData['rain'] < 0.3)] = 2
mergedData['sunnyDay'] = 0
mergedData['sunnyDay'][mergedData['clouds'] <= 0.1] = 1

# PCA

subsetOfData = mergedData[['temp','sunnyDay','rainType','wind','pressure','humidity']]
mergedData.drop(['temp','clouds','sunnyDay','rainType','rain','wind','pressure','humidity'],axis=1,inplace=True)
lowerDimensionWeatherData = preprocessing.reduceDimentionsOf(subsetOfData)
mergedData['weatherState'] = lowerDimensionWeatherData

# split 

dataFeatures = mergedData.drop(['price'],axis=1)
dataLabel = mergedData['price']

# Models
model = Model()
linearModel = loaded_models['LinearRegression']
polyModel = loaded_models['PolyLinearRegression']

print("LinearRegression")
print(metrics.r2_score(dataLabel, linearModel.predict(dataFeatures)))

print("PolynomialRegression")
dataFeatures = model.changeDegreeOf(dataFeatures,degree=4)
print(metrics.r2_score(dataLabel, polyModel.predict(dataFeatures)))