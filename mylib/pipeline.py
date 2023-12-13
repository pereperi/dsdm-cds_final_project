
from .preprocessing import *
from .features import *
from .model import *

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
    #model = train_model_elastic(X_train, y_train)
    model = train_model_rf(X_train, y_train)
    print("Model used: ", model)

    # Make predictions (on test data)
    y_pred = model.predict(X_test)

    # Evaluate model
    mse = evaluate_model_mse(y_test,y_pred)
    f1_score = evaluate_model_f1(y_test,y_pred)
    accuracy = evaluate_model_accuracy(y_test,y_pred)
    
    print('Mean squared error of the model: ', mse)
    print('F1 score of the model: ', f1_score)
    print('Accuracy of the model: ', accuracy)
    return f1_score,mse,accuracy
