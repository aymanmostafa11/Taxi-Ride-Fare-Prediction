from pickle import load
import pandas as pd
from sklearn import metrics
from helpers import PreProcessing
from helpers import Model
import warnings
warnings.filterwarnings("ignore")

def reg_test(loaded_models, dataFeatures, dataLabel):

    model = Model()
    linearModel = loaded_models['linearModel']
    polyModel = loaded_models['polyModel']

    print("LinearRegression")
    pred = linearModel.predict(dataFeatures)
    print(f"r2 score: {metrics.r2_score(dataLabel, pred)}\n, MSE: {metrics.mean_squared_error(dataLabel, pred)}")

    print("PolynomialRegression")
    dataFeatures = model.changeDegreeOf(dataFeatures, degree=4)
    pred = polyModel.predict(dataFeatures)
    print(f"r2 score: {metrics.r2_score(dataLabel, pred)}\n, MSE: {metrics.mean_squared_error(dataLabel, pred)}")

def classification_test(loaded_models, dataFeatures, dataLabel):

    for model in loaded_models:
        print(model)
        if model == 'PolynomialLogistic':
            print(metrics.f1_score(dataLabel, loaded_models[model].predict(Model().changeDegreeOf(dataFeatures, 3)), average='micro'))
        else:
            print(metrics.f1_score(dataLabel, loaded_models[model].predict(dataFeatures), average='micro'))


def run_test(path):

    preprocessing = PreProcessing()

    taxiRides = pd.read_csv(path)
    weather = pd.read_csv('taxi/weather.csv')
    # weather = pd.read_csv('/content/classification/weather.csv')
    regression = True if 'price' in taxiRides.columns else False
    
    if regression:
        cache_path = 'regression_cache'
        target = 'price'
    else:
        cache_path = 'classification_cache'
        target = 'RideCategory'

    # Loading Cache

    cache = load(open(cache_path, 'rb'))
    preprocessing.encoders = cache['encoders']
    loaded_models = cache['models']
    loaded_imputers = cache['imputers']
    preprocessing.featuresUniqueValues = cache['categoricalFeaturesValues']

    # imputing null values
    for feature in taxiRides.columns:
        if feature in loaded_imputers:
            taxiRides[feature].fillna(loaded_imputers[feature],inplace=True)

    # Data Cleaning
    weather['rain'].fillna(0, inplace=True)
  
    # Encoding Timestamps to date

    weatherDate = pd.to_datetime(weather['time_stamp'], unit='s').apply(lambda x: x.strftime(('%Y-%m-%d')))
    taxiRidesDate = pd.to_datetime(taxiRides['time_stamp'], unit='ms').apply(lambda x: x.strftime(('%Y-%m-%d')))
    weather['date'] = weatherDate
    taxiRides['date'] = taxiRidesDate

    # Joining Dataframes based on date

    taxiRides.drop(['time_stamp'], axis=1, inplace=True)
    weather.drop(['time_stamp'], axis=1, inplace=True)
    mergedData = pd.merge(taxiRides, weather.drop_duplicates(subset=['date', 'location']), how='left',
                          left_on=['date', 'source'], right_on=['date', 'location'])

    # Preprocessing
    columnsToDrop = ['id', 'date', 'product_id', 'location']
    mergedData.drop(columnsToDrop, axis=1, inplace=True)

    # Name Feature
    preprocessing.encodeManually(mergedData['name'], PreProcessing.nameFeatureMap)
    if not regression:
        preprocessing.encodeManually(mergedData['RideCategory'], PreProcessing.labelFeatureMap)
        mergedData['RideCategory'] = mergedData['RideCategory'].astype(int)
    # Encoding with previously fit encoders
    nonIntegerColumns = [col for col in mergedData.columns if mergedData[col].dtypes == object ]
    preprocessing.encode_cached(mergedData, nonIntegerColumns)

    # Weather Features Engineering

    mergedData['rainType'] = 0
    mergedData['rainType'][(mergedData['rain'] > 0) & (mergedData['rain'] < 0.1)] = 1
    mergedData['rainType'][(mergedData['rain'] > 0.1) & (mergedData['rain'] < 0.3)] = 2
    mergedData['sunnyDay'] = 0
    mergedData['sunnyDay'][mergedData['clouds'] <= 0.1] = 1

    # handling nulls in weather features for unknown sources

    for feature in mergedData:
       if mergedData[feature].isnull().sum()!=0 and weather.__contains__(feature):
           mergedData[feature].fillna(weather[feature].mean(),inplace=True)

    # PCA

    subsetOfData = mergedData[['temp', 'sunnyDay', 'rainType', 'wind', 'pressure', 'humidity']]
    mergedData.drop(['temp', 'clouds', 'sunnyDay', 'rainType', 'rain', 'wind', 'pressure', 'humidity'], axis=1,
                    inplace=True)
    lowerDimensionWeatherData = preprocessing.reduceDimentionsOf(subsetOfData)
    mergedData['weatherState'] = lowerDimensionWeatherData

    # split & scale

    dataFeatures = mergedData.drop([target], axis=1)
    if 'scalers' in cache:
        preprocessing.scalingCache = cache['scalers']
        preprocessing.scaleCached(dataFeatures)
    dataLabel = mergedData[target]

    # Models
    if regression:
        reg_test(loaded_models, dataFeatures, dataLabel)
    else:
        classification_test(loaded_models, dataFeatures, dataLabel)

print("Classification:")
run_test('sampleTests/taxi-classification-test-samples.csv')
print()
print('#' * 20)
print()
run_test('sampleTests/taxi-reg-test-samples.csv')