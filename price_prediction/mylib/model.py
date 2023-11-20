
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

def train_model(X_train, y_train):
    '''Train an ElasticNet model on the training data.'''
    model = ElasticNet()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    '''Evaluate the model using mean squared error.'''
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse
