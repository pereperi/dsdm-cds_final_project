
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier

def train_test_split_data(data):
    '''Split the data into train and test sets.'''
    y = data['position']
    data.drop('position',axis=1,inplace=True)
    X = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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

def evaluate_model_mse(y_test, y_pred):
    '''Evaluate the model using mean squared error.'''
    mse = mean_squared_error(y_test, y_pred)
    return mse

def evaluate_model_f1(y_test,y_pred):
    '''Evaluate the model using  F1 score.'''
    f1 = f1_score(y_test, y_pred, average='weighted')
    return f1

def evaluate_model_accuracy(y_test,y_pred):
    '''Evaluate the model using  accuracy.'''
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy