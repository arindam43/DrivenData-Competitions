from importlib import reload
import pandas as pd
import numpy as np
import re, os, sys, itertools, time, datetime

try:
    reload(sys.modules['FeatureEngineering'])
    reload(sys.modules['Modeling'])
    reload(sys.modules['Predict'])
    reload(sys.modules['Ingest'])
except KeyError:
    pass

from FeatureEngineering import engineer_features, remove_outliers, create_model_datasets
from Modeling import build_models, calculate_validation_metrics
from Predict import predict_test_values
from Ingest import ingest_data, preprocess_data


# Read in all data sets
path = os.getcwd() + '\\Data\\'
if 'raw_data' not in locals():
    raw_data, labels, metadata, test_data, start_times = ingest_data(path)

    # Pre-process data
    raw_data = preprocess_data(raw_data, start_times)
    test_data = preprocess_data(test_data, start_times)
else:
    print('Raw data already found, skipping data read and initial pre-processing.')

#
# if 'train_eda' not in locals():
#     train_eda = raw_data.describe(percentiles=[0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])
#
# if 'test_eda' not in locals():
#     test_eda = test_data.describe(percentiles=[0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])


# Create walk forward train/validation splits
validation_results = pd.DataFrame(columns=['Model_Type', 'Train_Ratio',
                                           'Num_Leaves', 'Min_Data_In_Leaf', 'Feature_Fraction',
                                           'Best_MAPE', 'Best_Num_Iters'])
response = 'final_rinse_total_turbidity_liter'
train_val_ratios = list(range(40, 53, 4))  # training set sizes of 40, 44, 48, and 52 days
max_train_ratio = max(train_val_ratios)
start_time = time.time()

for train_ratio in train_val_ratios:
    print('')
    print('Training with first ' + str(int(train_ratio)) + ' days of training data...')

    # Identify which processes will be train and which will be validation
    train_processes = pd.DataFrame(start_times.process_id[start_times.day_number <= train_ratio])
    val_processes = pd.DataFrame(start_times.process_id[start_times.day_number > train_ratio])

    raw_train_data = raw_data[raw_data.process_id.isin(train_processes.process_id)]
    raw_val_data = raw_data[raw_data.process_id.isin(val_processes.process_id)]

    processed_train_data, \
    processed_val_data = create_model_datasets(raw_train_data, raw_val_data, start_times, labels, metadata,
                                               path, type='validation')

    non_phase_cols = ['object_id', 'recipe_caus', 'recipe_int', 'recipe_ac']
    non_phase_cols_acid = ['object_id']
    cols_to_include = {'pre_rinse': list(filter(lambda x: re.search(r'(?=.*pre_rinse)', x), list(processed_train_data.columns))) + non_phase_cols,
                       'caustic': list(filter(lambda x: re.search(r'(?=.*pre_rinse|.*caustic)', x), list(processed_train_data.columns))) + non_phase_cols,
                       'int_rinse': list(filter(lambda x: re.search(r'(?=.*pre_rinse|.*caustic|.*int_)', x), list(processed_train_data.columns))) + non_phase_cols,
                       'acid': list(filter(lambda x: re.search(r'(?=.*pre_rinse|.*caustic|.*int_|.*acid)', x), list(processed_train_data.columns))) + non_phase_cols_acid
                       #'acid': list(set(processed_train_data.columns) - set(['object_id', 'process_id', 'pipeline', 'day_number', 'start_time', response])) + non_phase_cols
                       }

    modeling_approach = 'single_model'

    if modeling_approach == 'parameter_tuning':
        leaves_tuning = [31, 40, 50, 63, 70, 80]
        min_data_in_leaf_tuning = [25]
        feature_fraction_tuning = [0.6, 0.7, 0.8, 0.9, 1]
        tuning_grid = list(itertools.product(leaves_tuning, min_data_in_leaf_tuning, feature_fraction_tuning))
        counter = 1

        for tuning_params in tuning_grid:
            print('')
            print('Hyperparameter tuning, model ' + str(counter) + ' of ' + str(len(tuning_grid)) + '...')
            print('num_leaves: ' + str(tuning_params[0]))
            print('min_data_in_leaf: ' + str(tuning_params[1]))
            print('feature_fraction: ' + str(tuning_params[2]))

            # specify your configurations as a dict
            params = {'pre_rinse': {'boosting_type': 'gbdt', 'objective': 'mape', 'num_leaves': tuning_params[0],
                                    'learning_rate': 0.02, 'verbose': -1, 'min_data_in_leaf': tuning_params[1],
                                    'feature_fraction': tuning_params[2]},
                      'caustic': {'boosting_type': 'gbdt', 'objective': 'mape', 'num_leaves': tuning_params[0],
                                  'learning_rate': 0.02, 'verbose': -1, 'min_data_in_leaf': tuning_params[1],
                                  'feature_fraction': tuning_params[2]},
                      'int_rinse': {'boosting_type': 'gbdt', 'objective': 'mape', 'num_leaves': tuning_params[0],
                                    'learning_rate': 0.02, 'verbose': -1, 'min_data_in_leaf': tuning_params[1],
                                    'feature_fraction': tuning_params[2]},
                      'acid': {'boosting_type': 'gbdt', 'objective': 'mape', 'num_leaves': tuning_params[0],
                               'learning_rate': 0.02, 'verbose': -1, 'min_data_in_leaf': tuning_params[1],
                               'feature_fraction': tuning_params[2]},
            }

            for model_type in cols_to_include.keys():
                validation_results = build_models(model_type, processed_train_data, processed_val_data, params[model_type],
                                                  response, cols_to_include[model_type], train_ratio, max_train_ratio,
                                                  tuning_params, validation_results)

            counter = counter + 1
    else:
        tuning_params = ('NA', 'NA', 'NA')
        # specify your configurations as a dict
        params = {'pre_rinse': {'boosting_type': 'gbdt', 'objective': 'mape', 'num_leaves': 31,
                                'learning_rate': 0.02, 'verbose': -1, 'min_data_in_leaf': 25,
                                'feature_fraction': 1},
                  'caustic': {'boosting_type': 'gbdt', 'objective': 'mape', 'num_leaves': 48,
                              'learning_rate': 0.02, 'verbose': -1, 'min_data_in_leaf': 25,
                              'feature_fraction': 0.9},
                  'int_rinse': {'boosting_type': 'gbdt', 'objective': 'mape', 'num_leaves': 63,
                                'learning_rate': 0.02, 'verbose': -1, 'min_data_in_leaf': 25,
                                'feature_fraction': 0.8},
                  'acid': {'boosting_type': 'gbdt', 'objective': 'mape', 'num_leaves': 70,
                           'learning_rate': 0.02, 'verbose': -1, 'min_data_in_leaf': 25,
                           'feature_fraction': 0.7}}

        for model_type in cols_to_include.keys():
            validation_results = build_models(model_type, processed_train_data, processed_val_data, params[model_type],
                                              response, cols_to_include[model_type], train_ratio, max_train_ratio,
                                              tuning_params, validation_results)


validation_results.Best_Num_Iters = validation_results.Best_Num_Iters.astype(int)

validation_summary = validation_results.groupby(['Model_Type', 'Num_Leaves', 'Min_Data_In_Leaf', 'Feature_Fraction']).\
    agg({'Best_MAPE': np.mean, 'Best_Num_Iters': np.median}).reset_index()

validation_summary.Best_Num_Iters = validation_summary.Best_Num_Iters.astype(int)

end_time = time.time()
print('Total time taken for hyperparameter tuning: ' + str(datetime.timedelta(seconds=end_time - start_time)))

test_iterations = calculate_validation_metrics(validation_summary)


# Train on full data and make predictions
print('')
print('Training full model and making test set predictions...')
predict_test_values(raw_data, test_data, start_times, metadata, path,
                    params, response, test_iterations, cols_to_include, labels)
