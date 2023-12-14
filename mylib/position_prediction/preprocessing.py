
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer

def load_data(file_path):
    '''Load data from a CSV file.'''
    return pd.read_csv(file_path)

def drop_string_columns(data):
    '''Drop string columns from the dataset except for the 'position' column.'''
    non_string_data = data.select_dtypes(exclude=['object'])
    return non_string_data

def handle_missing_values(data):
    '''Handle missing values in the dataset, excluding string columns.'''

    # Separate the data into string and non-string columns
    string_data = data.select_dtypes(include=['object'])
    non_string_data = data.select_dtypes(exclude=['object'])
    
    # Fill missing values in the non-string columns with KNN
    non_string_data_filled = column_fill_KNN(non_string_data, non_string_data.columns)
    
    data_filled = pd.concat([non_string_data_filled, string_data], axis=1)
    
    return data_filled

def numerical_encoder(data):
    '''Numerically encode the specified column in the dataset.
    Return both the encoded column, the encoder and the decoder.'''
    encoder = LabelEncoder()
    encoded_column = encoder.fit_transform(data)
    
    # Create a mapping dictionary from encoded labels to original labels
    encoder_mapping = dict(zip(encoder.classes_, range(len(encoder.classes_))))
    decoder_mapping = dict(zip(range(len(encoder.classes_)), encoder.classes_))
    
    return encoded_column, encoder_mapping, decoder_mapping

# normalization the column given
def scaling_normalization(df, column_name:str):
    df[column_name] = (df[column_name] - df[column_name].min()) / (df[column_name].max() - df[column_name].min())
    return df

# standardization the column given
def scaling_standardization(df, column_name:str):
    df[column_name] = (df[column_name] - df[column_name].mean()) / df[column_name].std()
    return df

# fill missing values with KNN
def column_fill_KNN(df, column_name:str, neighbors=3):
    imputer = KNNImputer()
    imputed_data = imputer.fit_transform(df)
    df_temp = pd.DataFrame(imputed_data)
    df_temp.columns = df.columns
    df[column_name] = df_temp[column_name]
    return df

# Encodes numerical values to categorical numbers (keeps the NaN values as NaN)
def column_string_to_num_encoding(df, name_column: str):
    unique_values = df[name_column].dropna().unique()
    mapping_dict = {value: idx for idx, value in enumerate(unique_values)}
    df[name_column] = df[name_column].map(mapping_dict)
    return df