
from .preprocessing import load_data, handle_missing_values, drop_string_columns, numerical_encoder
from .features import clean_player_traits, clean_body_type, process_work_rate, calculate_years_until_expiry, calculate_age, calculate_years_in_club
from .model import train_model, evaluate_model, train_test_split_data

def run_pipeline(data_path):
    '''Run the end-to-end pipeline.'''
    # Load data
    data = load_data(data_path)
    
    target = 'position'
    
    data[target], encoder_dict, decoder_dict = numerical_encoder(data[target])

    # Feature engineering
    data = clean_player_traits(data)
    data = clean_body_type(data)
    data = process_work_rate(data)
    data = calculate_years_until_expiry(data)
    data = calculate_age(data)
    data = calculate_years_in_club(data)
    
        # Preprocess data
    data = handle_missing_values(data)
    data = drop_string_columns(data)

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split_data(data)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    mse = evaluate_model(model, X_test, y_test)
    
    print('Mean squared error of the model: ', mse)
    return mse
