
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.ensemble import RandomForestClassifier

def train_test_split_data(data):
    '''Split the data into train and test sets.'''
    X_train, X_test, y_train, y_test = train_test_split(data.drop('position', axis=1), data['position'], test_size=0.2)
    return X_train, X_test, y_train, y_test

def train_model_elastic(X_train, y_train):
    '''Train an ElasticNet model on the training data.'''
    model = ElasticNet()
    model.fit(X_train, y_train)
    return model

def train_model_rf(X_train, y_train):
    '''Train a Random Forest model on the training data.'''
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def evaluate_model_mse(model, X_test, y_test):
    '''Evaluate the model using mean squared error.'''
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse

def evaluate_model_f1(model, X_test, y_test):
    '''Evaluate the model using mean squared error.'''
    predictions = model.predict(X_test)
    f1 = f1_score(y_test, predictions)
    return f1
