import pandas as pd
from datetime import datetime

# Define the function that cleans the player traits variable
def clean_player_traits(data):
    def clean_traits(x):
        if isinstance(x, str):
            return x.split(',')
        else:
            return x

    data['player_traits'] = data['player_traits'].apply(clean_traits)
    return data

# Define the function to clean body type
def clean_body_type(data):
    def extract_bodytype(bodytype_string):
        return bodytype_string.split()[0]

    data['body_type'] = data['body_type'].apply(extract_bodytype)
    return data


# Define the function to process work rate and have it as a categorical variable (separating attacking and defensive wr)
def process_work_rate(data):
    def wr_converter(wr):
        if wr == 'High':
            return 3
        if wr == 'Medium':
            return 2
        if wr == 'Low':
            return 1

    data['attacking_wr'] = data['work_rate'].apply(lambda x: wr_converter(x.split('/')[0]))
    data['defensive_wr'] = data['work_rate'].apply(lambda x: wr_converter(x.split('/')[1]))
    data.drop('work_rate', axis = 1, inplace = True)
    
    return data


# Define the function to calculate years until contract expires
def calculate_years_until_expiry(data):
    data['years_until_contract_expires'] = data['club_contract_valid_until'] - datetime.today().year
    data.drop('club_contract_valid_until', axis = 1, inplace = True)
    
    return data


# Define the function to calculate age
def calculate_age(data):
    reference_date = datetime.now()
    data['age'] = (reference_date - pd.to_datetime(data['birthday_date'])).dt.days // 365
    data.drop('birthday_date', axis = 1, inplace = True)
    
    return data


# Define the function to calculate years in the club
def calculate_years_in_club(data):
    reference_date = datetime.now()
    data['yearinclub'] = (reference_date - pd.to_datetime(data['club_joined'])).dt.days // 365
    data.drop('club_joined', axis = 1, inplace = True)
    
    return data