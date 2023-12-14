
from imblearn.over_sampling import SMOTE
from .preprocessing import *
from .features import *
from .model import *

def run_pipeline(data_path, models = ['rf', 'logistic', 'svc'],use_grid_search = False, use_feature_selection = False, use_SMOTE = True):
    '''Run the end-to-end pipeline.'''
    # Load data
    data = load_data(data_path)
    
    target = 'position'
    
    data[target], encoder_dict, decoder_dict = numerical_encoder(data[target])

    # Feature engineering
    data = handle_player_traits(data)
    data = handle_body_type(data)
    data = process_work_rate(data)
    data = calculate_years_until_expiry(data)
    data = calculate_age(data)
    data = calculate_years_in_club(data)
    data = handle_foot(data)
    
    # Set the nan values from columns ending with _tag and _trait to 0
    columns_to_fill = [col for col in data.columns if col.endswith('_trait')]
    data[columns_to_fill] = data[columns_to_fill].fillna(0)

    # Preprocess data
    data = handle_missing_values(data)
    data = drop_string_columns(data)

    # Normalize the columns execpt 'id' and 'position'
    data = scaling_normalization(data, data.loc[:, ~data.columns.isin(['id', 'position'])].columns)

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split_data(data)

    # Feature scaling
    if use_SMOTE:
            print("SMOTE")
            smote = SMOTE(sampling_strategy='auto', random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)

    # TODO: Feature selection

    print("X_train shape: ", X_train.shape)

    def evaluate_model(model, X_test, y_test):
        '''Evaluate the model using  F1 score.'''
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        accuracy = accuracy_score(y_test, y_pred)
        print("Model: ", model)
        print('F1 score of the model: ', f1)
        print('Accuracy of the model: ', accuracy)
        #return f1, accuracy, auc

    for m_name in models:
        if m_name == 'rf':
            model = train_model_rf(X_train, y_train, use_grid_search)
            evaluate_model(model, X_test, y_test)
        elif m_name == 'logistic':
            model = train_model_logistic(X_train, y_train, use_grid_search)
            evaluate_model(model, X_test, y_test)
        elif m_name == 'svc':
            model = train_model_svc(X_train, y_train, use_grid_search)
            evaluate_model(model, X_test, y_test)
        else:
            print("Model not supported. Please use one of the following models: rf, elastic, logistic, svc")
