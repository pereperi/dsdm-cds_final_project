
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    '''Load data from a CSV file.'''
    return pd.read_csv(file_path)

def drop_string_columns(data):
    '''Drop string columns from the dataset.'''
    return data.select_dtypes(exclude=['object'])

def handle_missing_values(data):
    '''Handle missing values in the dataset.'''
    # Example: Fill missing values with the median or mean
    return data.fillna(data.median())
