"""ML Project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/aymanmostafa11/Taxi-Ride-Fare-Prediction/blob/main/ML%20Project.ipynb
"""

# import gdown
# gdown.download_folder('https://drive.google.com/drive/folders/1r9BARaPl-5odlOwCPE8LZJ1cFWRZGsYO?usp=sharing')

import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from helpers import Model , preProcessing

taxiRides = pd.read_csv('‪taxi‬‏/taxi-rides.csv')
weather = pd.read_csv('‪taxi‬‏/weather.csv')

print(f"Taxi Rides has {taxiRides.shape[0]} Rows and {taxiRides.shape[1]} Columns")
print(f"Weather has {weather.shape[0]} Rows and {weather.shape[1]} Columns")

# Data Cleaning

# Nulls
print(f"Null Values in columns: \n{taxiRides.isnull().sum()}")

# product_id and name
print(f"Value counts of 'product_id' feature: \n{taxiRides['product_id'].value_counts()}\n")
print(f"Value counts of 'name' feature: \n{taxiRides['name'].value_counts()}\n")

"""
    product_id and name represent the same feature so we can drop one of them
"""

#  Encoding Timestamps to date
weatherDate = pd.to_datetime(weather['time_stamp'], unit='s').apply(lambda x: x.strftime(('%Y-%m-%d')))
taxiRidesDate = pd.to_datetime(taxiRides['time_stamp'], unit='ms').apply(lambda x: x.strftime(('%Y-%m-%d')))
weather['date'] = weatherDate
taxiRides['date'] = taxiRidesDate
taxiRides.drop(['time_stamp'],axis = 1, inplace = True)
weather.drop(['time_stamp'],axis = 1, inplace = True)

# Joining Dataframes based on date
mergedData = pd.merge(taxiRides,weather.drop_duplicates(subset=['date', 'location']), how = 'left', left_on=['date', 'source'], right_on=['date', 'location'])

# Rain Feature
print(f"percentage of Nulls in Rain Column: {(weather['rain'].isnull().sum() / weather['rain'].shape[0])*100}")

"""
    Do null values of rain revolve around certain values?
"""

print(f"Rows with null rain value statistics: \n{weather[weather['rain'].isnull()].describe()}")
print(f"Rows with non-null rain value statistics: \n{weather[weather['rain'].notna()].describe()}")
print(f"Values of 0 in the rain feature:\n{(weather['rain'] == 0).sum()}")

"""
    Rain feature nulls could indicate no rain
"""
print(f"Null values in Merged Data: \n{mergedData.isnull().sum()}")


"""
    Apparently all *price* values of *Taxi* are missing, 
    could all the missing values from *price* be from the *taxi* cab type? 
    we need to verify this
"""

taxiNullValues = mergedData[mergedData['price'].isnull()]['name'].value_counts()['Taxi']
totalNullValues = mergedData.isnull().sum()['price']
print(f"There are {taxiNullValues} price null values with Taxi as subtype from a total of {totalNullValues} \
: {taxiNullValues / totalNullValues * 100}%")

# Preprocessing

# Encoding
preProcessing = preProcessing()
preProcessing.encode_name(mergedData['name'])
preProcessing.drop_adjust(mergedData)

# Other Features
nonIntegerColumns = [col for col in mergedData.columns if mergedData[col].dtypes == object]
print(f"Non Integer Columns : {nonIntegerColumns}")
preProcessing.encode(mergedData,nonIntegerColumns)
mergedData.dropna(axis=0, subset=['price'], inplace=True)

print(f"Nulls after dropping null values in price: \n{mergedData.isnull().sum()}")

#   Rain feature

mergedData['rain'].fillna(0,inplace=True)

print(f"Rain Feature:\n {mergedData['rain'].describe()}")

"""
    Referring to google:
    Light rainfall is considered <b>less than 0.10 inches</b> of rain per hour. 
    Moderate rainfall measures <b>0.10 to 0.30 inches</b> of rain per hour. 
    Heavy rainfall is more than <b>0.30 inches</b>
    of rain per hour.</blockquote>
    0 : no rain <br>
    1 : light rain <br>
    2 : mid rain <br>
    3 : heavy rain (doesn't exist in the data)
"""

mergedData['rainType'] = 0
mergedData['rainType'][(mergedData['rain'] > 0) & (mergedData['rain'] < 0.1)] = 1
mergedData['rainType'][(mergedData['rain'] > 0.1) & (mergedData['rain'] < 0.3)] = 2

print(f"rainType value counts: \n{mergedData['rainType'].value_counts()}")

""" 
    Clouds engineering
    making the assumption that clouds are on normalized [Okta Scale](https://polarpedia.eu/en/okta-scale/),
    that means values less than 0.1 are sunny days
"""

mergedData['sunnyDay'] = 0
mergedData['sunnyDay'][mergedData['clouds'] <= 0.1] = 1
print(f"sunnyDay value counts: \n{mergedData['sunnyDay'].value_counts()}")

# Outliers

standardPrice = (mergedData['price'] - mergedData['price'].mean()) / mergedData['price'].std()
priceOutliers = mergedData[((standardPrice > 3.5) | (standardPrice < -3.5))]
print(f"Length of Outliers: {len(priceOutliers)}")
print(f"Value counts of surge_multipliers for outlier prices:\n {priceOutliers['surge_multiplier'].value_counts()}")
print(f"Value counts of cab_class for outlier prices with normal surge_multipliers\n {priceOutliers[priceOutliers['surge_multiplier'] == 1.0]['ride_class'].value_counts()}")

# Dimentionality Reduction

subsetOfData = mergedData[['temp','sunnyDay','rainType','wind','pressure','humidity']]
mergedData.drop(['temp','clouds','sunnyDay','rain','rainType','wind','pressure','humidity'],axis=1,inplace=True)
lowerDimensionWeatherData = preProcessing.reduceDimentionsOf(subsetOfData)
mergedData['weatherState'] = lowerDimensionWeatherData
print(f"Merged Data Correlation: \n{mergedData.corr()}")

# Model

dataFeatures = mergedData.drop(['price'],axis=1)
dataLabel = mergedData['price']
print(f"DataFeatures Head\n {dataFeatures.head()}")

model = Model()

# First Model

splitData = model.splitData(dataFeatures,dataLabel)

model.fitLinearModel(splitData["trainFeatures"],
                     splitData["trainLabel"],
                     metrics.r2_score,
                     splitData["testFeatures"],
                     splitData["testLabel"])

# Second Model

polyDegree = 4
model.fitPolyModel(splitData["trainFeatures"],
                    splitData["trainLabel"],
                    polyDegree,
                    metrics.r2_score,
                    splitData["testFeatures"],
                    splitData["testLabel"])

model.crossValidateOn(linear_model.LinearRegression(),
                      dataFeatures,
                      dataLabel,
                      polyDegree,
                      metric = metrics.r2_score, k = 3)

