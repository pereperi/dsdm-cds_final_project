
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def train_test_split_data(data):
    '''Split the data into train and test sets.'''
    y = data['position']
    data.drop('position',axis=1,inplace=True)
    X = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model_rf(X_train, y_train, use_grid_search=False):
    '''Train a Random Forest model on the training data.'''
    if use_grid_search:
        print("Hyperparameter tuning using Grid Search")
        hyperparameters = hyperparameters_GridSearch(X_train, y_train, RandomForestClassifier())
        model = RandomForestClassifier(**hyperparameters)
    else:
        model = RandomForestClassifier(bootstrap=False, min_samples_split=5)
    
    model.fit(X_train, y_train)
    return model

def train_model_logistic(X_train, y_train, use_grid_search=False):
    '''Train a Logistic Regression model on the training data.'''
    if use_grid_search:
        print("Hyperparameter tuning using Grid Search")
        hyperparameters = hyperparameters_GridSearch(X_train, y_train, LogisticRegression())
        model = LogisticRegression(**hyperparameters)
    else:
        model = LogisticRegression(C=1, max_iter=200, penalty='l1', solver='liblinear')
    
    model.fit(X_train, y_train)
    return model

def train_model_svc(X_train, y_train, use_grid_search=False):
    '''Train a SVC model on the training data.'''
    if use_grid_search:
        print("Hyperparameter tuning using Grid Search")
        hyperparameters = hyperparameters_GridSearch(X_train, y_train, SVC())
        model = SVC(**hyperparameters)
    else:
        model = SVC()
    
    model.fit(X_train, y_train)
    return model

def evaluate_model_f1(y_test,y_pred):
    '''Evaluate the model using  F1 score.'''
    f1 = f1_score(y_test, y_pred, average='weighted')
    return f1

def evaluate_model_accuracy(y_test,y_pred):
    '''Evaluate the model using  accuracy.'''
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def evaluate_model_auc(y_test,y_pred):
    '''Evaluate the model using  accuracy.'''
    auc = roc_auc_score(y_test, y_pred,multi_class='ovr')
    return auc

def hyperparameters_GridSearch(X_train, y_train, model):
    '''Perform hyperparameter tuning using Grid Search.'''
    random_forest_param_grid = {'n_estimators': [50, 100],'max_depth': [None, 10],'min_samples_split': [2, 5, 10],'min_samples_leaf': [1, 3],'bootstrap': [True, False]}
    svc_param_grid = {'C': [0.1, 1],'kernel': ['linear', 'poly'],'gamma': ['scale', 'auto'],'degree': [1, 2]}
    logistic_regression_param_grid = {'penalty': ['l1', 'l2'],'C': [0.1, 1],'solver': ['liblinear', 'saga'],'max_iter': [200, 300]}

    if model.__class__.__name__ == 'RandomForestClassifier':
        param_grid = random_forest_param_grid
    elif model.__class__.__name__ == 'SVC':
        param_grid = svc_param_grid
    elif model.__class__.__name__ == 'LogisticRegression':
        param_grid = logistic_regression_param_grid
    else:
        # Return error
        param_grid = None
        print("Model not supported. Please use one of the following models: RandomForestClassifier, SVC, LogisticRegression")

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,cv=5,scoring='f1_weighted',verbose=1,n_jobs=-1)
    grid_search.fit(X_train, y_train)
    grid_search.best_params_
    print("Best Hyperparameters:", grid_search.best_params_)
    return grid_search.best_params_

