from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures , LabelEncoder , MinMaxScaler , StandardScaler
import pandas as pd
class Model:

  # Cache
  lastLinearModel = None

  def fitLinearModel(self, features, label, evaluationFunction = None,
                     validationFeatures = None, validationLabel = None):
    '''
    Parameters :
      features : data
      label : target
      -optional- evaluationMetric :
        an Sk-Learn evaluation function to display the final
        performance of the model after fitting
    Returns :
      sklearn LinearRegression Model fit on data
    '''
    linearRegressionModel = linear_model.LinearRegression()
    linearRegressionModel.fit(features,label)

    # there's a probably cleaner way to do this
    if evaluationFunction is not None:
      prediction = linearRegressionModel.predict(features)
      print(f"Final model {evaluationFunction.__name__} on train: {evaluationFunction(label, prediction)}")

      if validationFeatures is not None:
        prediction = linearRegressionModel.predict(validationFeatures)
        print(f"Final model {evaluationFunction.__name__} on validation: {evaluationFunction(validationLabel, prediction)}")
      
    self.lastLinearModel = linearRegressionModel
    return self.lastLinearModel
  
  def fitPolyModel(self, features, label, polynomialDegree = 1, 
                   evaluationFunction = None,
                   validationFeatures = None, validationLabel = None):
    '''
    Transforms data to a polynomial degree then fits a linear model on it
    Parameters :
      features : data
      label : target
      -optional- evaluationMetric :
        an Sk-Learn evaluation function to display the final
        performance of the model after fitting
    Returns :
      sklearn LinearRegression Model fit on data after polynomial transformation
    '''
    features = self.changeDegreeOf(features, polynomialDegree)
    if validationFeatures is not None:
      validationFeatures = self.changeDegreeOf(validationFeatures, polynomialDegree)
    
    return self.fitLinearModel(features, label, evaluationFunction,
                                             validationFeatures,
                                             validationLabel)
    

  def changeDegreeOf(self, features, degree = 1):
    polynomialFunction = PolynomialFeatures(degree=degree)
    polynomialDataFeatures = polynomialFunction.fit_transform(features)
    return polynomialDataFeatures
  
  def crossValidateOn(self, model, features, label, polynomialDegree = 1
                      , k = 5, metric = metrics.mean_squared_error):
    '''
    Performs K-Fold cross Validation on data using given model
    Parameters : 
      model : an Sk-Learn model or any models that implements .fit() function
      features : data
      label : target
      polynomialDegree : change features to polynomial, default is 1 (no change)
      Metric :
        an Sk-Learn evaluation function to asses the model on (default is MSE)
    '''
    features = self.changeDegreeOf(features, polynomialDegree)
    scores = cross_val_score(model, features, label, cv = k, scoring = metrics.make_scorer(metric))
    print(f"Average Score : {sum(scores) / k}")


  def splitData(self, dataFeatures, dataLabel, test_size = 0.2):
    '''
    Splits data into train test set
    return :
      dictionary containing split data
      Example:
        data = splitData(features, label)
        data.keys() -> view names used to access data
        data["trainFeatures"] -> training data
        data["testLabel"] -> test labels
    '''
    data = tuple(train_test_split(dataFeatures, dataLabel, shuffle=True, random_state=10, test_size= test_size))
    return {"trainFeatures":data[0], "testFeatures": data[1], "trainLabel": data[2], "testLabel": data[3]}

class PreProcessing:

  encoder = None

  def scale(self,dataFeatures,type = "minMax"):
    '''
    Scale data in specific range
    dataFeatures : data
    type: method of scaling to use
    return: scaled features
    '''
    scaler = None
    if type == "minMax":
      scaler = MinMaxScaler()
    elif type == "standardization":
      scaler = StandardScaler()  
    scaledDataFeatures = scaler.fit_transform(dataFeatures)    
    return scaledDataFeatures

  def encode(self,data,features, method = "label" ):
    ''' 
    change data features values from strings to numeric values
    data: dataframe 
    features: non integer features to encode
    method: encoding technique to change feature values
    return: encoded features
    '''
    if self.encoder == None:
      if method == "label":
        self.encoder = LabelEncoder()
      elif method == "oneHot":
        self.encoder = OneHotEncoder()
      transform_ = self.encoder.fit_transform
    else:
      transform_ = self.encoder.transform

    for feature in features:
      data[feature] = transform_(data[feature])

  
  def reduceDimentionsOf(self,dataFeatures,reduceTo = 1):
    '''
    reduce features of data from dimention n to dimention k using PCA algorithm
    dataFeatures: features to reduce
    reduceTo: number of components to keep
    return reduced data in lower dimentions
    '''
    pca = PCA(n_components=1)
    reducedDataFeatures = pca.fit_transform(dataFeatures)
    return reducedDataFeatures
  
  
  def encode_name(self,names):
    """
    encoding the name col to numbers that represent the price for each class
    :param names: the column to be encoded
    :return: encoded column
    """
    services = ['Shared', 'UberPool', 'Lyft', 'Taxi', 'WAV', 'UberX', 'UberXL', 'Lyft XL', 'Lux', 'Black',
                'Lux Black', 'Black SUV', 'Lux Black XL']
    labels = [0, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4]
    names.replace(services, labels, inplace=True)
    

  def drop_adjust(self,data):
    """
    dropping unnecessary columns
    and giving a col a meaningful name
    :param data: whole dataframe
    :return: adujest dataframe
    """
    data.drop(['date', 'id', 'product_id', 'location'], axis=1, inplace=True)
    data.rename(columns={'name': 'ride_class'}, inplace=True)
