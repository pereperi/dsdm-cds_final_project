import pytest
import pandas as pd
from mylib.preprocessing import *

# Test that the functions from preprocessing.py work as expected
def test_drop_string_columns():
    # Define the test input
    test_input = pd.DataFrame({'string_column': ['a', 'b', 'c'], 'non_string_column': [1, 2, 3]})
    
    # Define the expected output
    expected_output = pd.DataFrame({'non_string_column': [1, 2, 3]})
    
    # Run the function
    test_output = drop_string_columns(test_input)
    
    # Compare the actual and expected outputs
    assert test_output.equals(expected_output)

def test_column_string_to_num_encoding():
    # Define the test input
    test_input = pd.DataFrame({'string_column': ['a', 'b', 'c']})
    
    # Define the expected output
    expected_output = pd.DataFrame({'string_column': [0, 1, 2]})
    
    # Run the function
    test_output = column_string_to_num_encoding(test_input, 'string_column')
    
    # Compare the actual and expected outputs
    assert test_output.equals(expected_output)

def test_scaling_normalization():
    # Define the test input
    test_input = pd.DataFrame({'column_to_normalize': [1, 2, 3]})
    
    # Define the expected output
    expected_output = pd.DataFrame({'column_to_normalize': [0, 0.5, 1]})
    
    # Run the function
    test_output = scaling_normalization(test_input, 'column_to_normalize')
    
    # Compare the actual and expected outputs
    assert test_output.equals(expected_output)