from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures

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
    features = self.__changeDegreeOf(features, polynomialDegree)
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