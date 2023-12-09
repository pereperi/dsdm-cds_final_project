
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    '''Load data from a CSV file.'''
    return pd.read_csv(file_path)

def drop_string_columns(data):
    '''Drop string columns from the dataset except for the 'position' column.'''
    non_string_data = data.select_dtypes(exclude=['object'])
    position_column = data[['position']]
    return pd.concat([non_string_data, position_column], axis=1)

def handle_missing_values(data):
    '''Handle missing values in the dataset, excluding string columns.'''

    # Separate the data into string and non-string columns
    string_data = data.select_dtypes(include=['object'])
    non_string_data = data.select_dtypes(exclude=['object'])

    # Handle missing values in non-string columns only
    non_string_data_filled = non_string_data.fillna(non_string_data.median())

    # Concatenate the string and filled non-string data
    data_filled = pd.concat([non_string_data_filled, string_data], axis=1)

    return data_filled
