import pytest
import pandas as pd
from mylib.position_prediction.features import *

# Test that the functions from features.py work as expected
def test_process_work_rate():
    # Define the test input
    test_input = pd.DataFrame({'work_rate': ['High/Medium', 'Medium/Medium', 'Low/High']})
    
    # Define the expected output
    expected_output = pd.DataFrame({'attacking_wr': [3, 2, 1], 'defensive_wr': [2, 2, 3]})
    
    # Run the function
    test_output = process_work_rate(test_input)
    
    # Compare the actual and expected outputs
    assert test_output.equals(expected_output)

def test_handle_foot():
    # Define the test input
    test_input = pd.DataFrame({'preferred_foot': ['Right', 'Left', 'Right']})
    
    # Define the expected output
    expected_output = pd.DataFrame({'preferred_foot': [1, 0, 1]})
    
    # Run the function
    test_output = handle_foot(test_input)
    
    # Compare the actual and expected outputs
    assert test_output.equals(expected_output)

def test_calculate_age():
    # Define the test input
    test_input = pd.DataFrame({'birthday_date': ['1990-01-01', '1995-01-01', '2000-01-01']})
    
    # Define the expected output
    expected_output = pd.DataFrame({'age': [33, 28, 23]})
    
    # Run the function
    test_output = calculate_age(test_input)
    
    # Compare the actual and expected outputs
    assert test_output.equals(expected_output)