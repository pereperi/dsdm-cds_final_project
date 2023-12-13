
from imblearn.over_sampling import SMOTE
from .preprocessing import *
from .features import *
from .model import *

def run_pipeline(data_path):
    '''Run the end-to-end pipeline.'''
    # Load data
    data = load_data(data_path)
    
    target = 'position'
    
    data[target], encoder_dict, decoder_dict = numerical_encoder(data[target])

    # SMOTE
    use_SMOTE = False

    # Grid search
    use_grid_search = True

    # Feature engineering
    data = handle_player_traits(data)
    data = handle_body_type(data)
    data = process_work_rate(data)
    data = calculate_years_until_expiry(data)
    data = calculate_age(data)
    data = calculate_years_in_club(data)
    data = handle_foot(data)
    
    # Preprocess data
    data = handle_missing_values(data)
    data = drop_string_columns(data)

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split_data(data)

    # Feature scaling
    if use_SMOTE:
            print("SMOTE")
            smote = SMOTE(sampling_strategy='auto', random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)

    # TODO: Feature selection

    print("X_train shape: ", X_train.shape)
    print("X_train cols.: ", X_train.columns)
    # Train model
    #model = train_model_elastic(X_train, y_train)
    model = train_model_rf(X_train, y_train)
    print("Model used: ", model)

    # Make predictions (on test data)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # Evaluate model
    f1_score = evaluate_model_f1(y_test,y_pred)
    accuracy = evaluate_model_accuracy(y_test,y_pred)
    roc_auc = evaluate_model_auc(y_test,y_prob)
    
    print('F1 score of the model: ', f1_score)
    print('Accuracy of the model: ', accuracy)
    print('ROC AUC of the model: ', roc_auc)
    return f1_score,accuracy,roc_auc
