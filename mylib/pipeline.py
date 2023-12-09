
from preprocessing import load_data, handle_missing_values
from features import feature_engineering_step1, feature_engineering_step2
from model import train_model, evaluate_model

def run_pipeline(data_path):
    '''Run the end-to-end pipeline.'''
    # Load data
    data = load_data(data_path)

    # Preprocess data
    data = handle_missing_values(data)

    # Feature engineering
    data = feature_engineering_step1(data)
    data = feature_engineering_step2(data)

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split_data(data)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    mse = evaluate_model(model, X_test, y_test)
    
    print('Mean squared error of the model: ', mse)
    return mse
