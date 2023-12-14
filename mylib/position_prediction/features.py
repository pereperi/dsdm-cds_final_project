from .preprocessing import *
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer

# Player_traits column can contain multiple strings, only the key traits are kept (see player_traits_simplified)
# Then this is function uses TfidfVectorizer (with a limit of max_features) to create a new column for each trait
player_traits_simplified = ["Crosser", "Speed", "Dribbler", "Finesse", "Shot", "Solid", "Player", "Cautious", "Leadership", "Passes", "Taker", "Throw-in", "Injury", "Playmaker", "Outside", "Technical", "Team", "Tackles", "Power", "Header", "Flair", "Chip", "Free-Kick"]
def handle_player_traits(df,max_features=None):
    def transform_player_trait(trait):
        if pd.notna(trait):  # Check for NaN values
            relevant_words = [word for word in trait.split() if word in player_traits_simplified]
            return ', '.join(relevant_words) if len(relevant_words) > 0 else np.nan
        return trait

    # Apply the transformation to each element in the 'player_traits' column
    df['player_traits'] = df['player_traits'].apply(transform_player_trait)
    df['player_traits'].fillna('', inplace=True)
    df['player_traits'] = df['player_traits'].apply(lambda x: x.split(','))
    df['combined_trait'] = df['player_traits'].apply(lambda x: ', '.join(x))
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_trait'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out()+'_trait')
    df = df.reset_index(drop=True)
    df = pd.concat([df, tfidf_df], axis=1)
    df.drop(['player_traits', 'combined_trait'], axis=1, inplace=True)
    return df

# Define the function to clean body type
def handle_body_type(df):
    # body_type -> Mapping and replace 'Unique' by KNN['height','weight']
    body_type_mapping = {'Normal': 1,'Lean': 2,'Stocky': 3,}
    df['body_type'] = df['body_type'].str.extract(r'([a-zA-Z]+)')
    df['body_type'] = df['body_type'].replace('Unique', pd.NA)
    df['body_type'] = df['body_type'].map(body_type_mapping)
    df_temp = df[['height_cm','weight_kg','body_type']].copy()
    df_temp = column_fill_KNN(df_temp, 'body_type')
    df['body_type'] = df_temp['body_type'].round(0).astype(int)
    return df

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

#Turning variable from categorical to numerical
def handle_foot(df):
    df['preferred_foot'] = df['preferred_foot'].apply(lambda x: 1 if x == 'Right' else 0)
    return df

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
