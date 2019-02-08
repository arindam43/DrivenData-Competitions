from importlib import reload
import pandas as pd
import numpy as np
import re, os, sys, itertools, time, datetime

try:
    for module in ['FeatureEngineering', 'Modeling', 'Predict', 'Ingest']:
        reload(sys.modules[module])
except KeyError:
    pass

from FeatureEngineering import create_model_datasets
from Modeling import build_models, calculate_validation_metrics
from Predict import predict_test_values
from Ingest import ingest_data, preprocess_data


# Read in and pre-process phase-timestamp-level data
path = os.getcwd() + '\\Data\\'
if 'raw_data' not in locals():
    raw_data, labels, metadata, test_data, start_times = ingest_data(path)

    # Pre-process data
    raw_data = preprocess_data(raw_data, test_data, start_times)
    test_data = preprocess_data(test_data, test_data, start_times)

    # Pre-process metadata by converting the one-hot encoded recipe types to a single column
    # Only 3 recipe types in total: pre_rinse + caustic, all phases, and acid only
    metadata['recipe_type'] = np.where(metadata.caustic == 0, 'acid_only',
                                       np.where(metadata.intermediate_rinse == 1, 'full_clean', 'short_clean'))
    metadata = metadata[['process_id', 'recipe_type']]
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
                                           'Num_Leaves', 'Min_Data_In_Leaf', 'Feature_Fraction', 'Min_Hessian',
                                           'Best_MAPE', 'Best_Num_Iters'])
response = 'final_rinse_total_turbidity_liter'
train_val_ratios = list(range(40, 53, 4))  # training set sizes of 40, 44, 48, and 52 days
max_train_ratio = max(train_val_ratios)
start_time = time.time()

for train_ratio in train_val_ratios:
    print('')
    print('Training with first ' + str(int(train_ratio)) + ' days of training data...')

    # Identify which processes will be used for training and which will be used for validation
    train_processes = pd.DataFrame(start_times.process_id[start_times.day_number <= train_ratio])
    val_processes = pd.DataFrame(start_times.process_id[start_times.day_number > train_ratio])

    # Split data into training and validation sets
    raw_train_data = raw_data[raw_data.process_id.isin(train_processes.process_id)]
    raw_val_data = raw_data[raw_data.process_id.isin(val_processes.process_id)]

    # Process the raw data to create model-ready datasets
    # Features engineering, aggregation to process_id level, outlier removal
    processed_train_data, \
        processed_val_data = create_model_datasets(raw_train_data, raw_val_data, start_times, labels, metadata,
                                                   path, val_or_test='validation')

    # For each of the four models, identify which columns should be kept from overall set
    # Simulates data censoring in test data
    non_phase_cols_short = ['object_id', 'recipe_type']
    non_phase_cols_full = ['object_id']
    flow_cols = set(filter(lambda x: re.search(r'(?=.*flow)', x), list(processed_train_data.columns)))
    turb_cols = set(filter(lambda x: re.search(r'(?=.*turb)', x), list(processed_train_data.columns)))

    none_cols = set(filter(lambda x: re.search(r'(?=.*none|row_count.*)', x), list(processed_train_data.columns)))

    cols_to_include = {'pre_rinse': list(set(filter(lambda x: re.search(r'(?=.*pre_rinse)', x),
                                             list(processed_train_data.columns))) - none_cols - flow_cols) + non_phase_cols_short,
                       'caustic': list(set(filter(lambda x: re.search(r'(?=.*pre_rinse|.*caustic)', x),
                                           list(processed_train_data.columns))) - none_cols - flow_cols) + non_phase_cols_short,
                       'int_rinse': list(set(filter(lambda x: re.search(r'(?=.*pre_rinse|.*caustic|.*int_rinse)', x),
                                             list(processed_train_data.columns))) - none_cols - flow_cols) + non_phase_cols_full,
                       'acid': list(set(filter(lambda x: re.search(r'(?=.*pre_rinse|.*caustic|.*int_rinse|.*acid|.*other)', x),
                                        list(processed_train_data.columns))) - none_cols - turb_cols) + non_phase_cols_full
                       #'acid': list(set(processed_train_data.columns) - set(['object_id', 'process_id', 'pipeline', 'day_number', 'start_time', response])) + non_phase_cols
                       }

    modeling_approach = 'single_model'

    # Hyperparameter tuning - simple grid search
    if modeling_approach == 'parameter_tuning':
        leaves_tuning = [31, 45, 63, 75, 90]
        min_data_in_leaf_tuning = [10, 20, 30]
        feature_fraction_tuning = [1]
        min_sum_hessian_in_leaf_tuning = [10, 25, 40, 55, 65]
        tuning_grid = list(itertools.product(leaves_tuning, min_data_in_leaf_tuning, feature_fraction_tuning,
                                             min_sum_hessian_in_leaf_tuning))
        counter = 1

        for tuning_params in tuning_grid:
            print('')
            print('Hyperparameter tuning, model ' + str(counter) + ' of ' + str(len(tuning_grid)) + ', train ratio = '
                  + str(train_ratio) + '...')
            print('num_leaves: ' + str(tuning_params[0]))
            print('min_data_in_leaf: ' + str(tuning_params[1]))
            print('feature_fraction: ' + str(tuning_params[2]))
            print('min_hessian: ' + str(tuning_params[3]))

            # specify your configurations as a dict
            params = {'pre_rinse': {'boosting_type': 'gbdt', 'objective': 'mape', 'num_leaves': tuning_params[0],
                                    'learning_rate': 0.02, 'verbose': -1, 'min_data': tuning_params[1],
                                    'feature_fraction': tuning_params[2], 'min_hessian': tuning_params[3]},
                      'caustic': {'boosting_type': 'gbdt', 'objective': 'mape', 'num_leaves': tuning_params[0],
                                  'learning_rate': 0.02, 'verbose': -1, 'min_data': tuning_params[1],
                                  'feature_fraction': tuning_params[2], 'min_hessian': tuning_params[3]},
                      'int_rinse': {'boosting_type': 'gbdt', 'objective': 'mape', 'num_leaves': tuning_params[0],
                                    'learning_rate': 0.02, 'verbose': -1, 'min_data': tuning_params[1],
                                    'feature_fraction': tuning_params[2], 'min_hessian': tuning_params[3]},
                      'acid': {'boosting_type': 'gbdt', 'objective': 'mape', 'num_leaves': tuning_params[0],
                               'learning_rate': 0.02, 'verbose': -1, 'min_data': tuning_params[1],
                               'feature_fraction': tuning_params[2], 'min_hessian': tuning_params[3]},
            }

            for model_type in cols_to_include.keys():
                validation_results = build_models(model_type, processed_train_data, processed_val_data, params[model_type],
                                                  response, cols_to_include[model_type], train_ratio, max_train_ratio,
                                                  tuning_params, validation_results)

            counter = counter + 1

    elif modeling_approach == 'single_model':
        tuning_params = ('NA', 'NA', 'NA', 'NA')
        # specify your configurations as a dict
        params = {'pre_rinse': {'boosting_type': 'gbdt', 'objective': 'mape', 'num_leaves': 63,
                                'learning_rate': 0.02, 'verbose': -1, 'min_data_in_leaf': 10,
                                'feature_fraction': 1, 'min_hessian': 25},
                  'caustic': {'boosting_type': 'gbdt', 'objective': 'mape', 'num_leaves': 90,
                              'learning_rate': 0.02, 'verbose': -1, 'min_data_in_leaf': 10,
                              'feature_fraction': 1, 'min_hessian': 25},
                  'int_rinse': {'boosting_type': 'gbdt', 'objective': 'mape', 'num_leaves': 63,
                                'learning_rate': 0.02, 'verbose': -1, 'min_data_in_leaf': 10,
                                'feature_fraction': 1, 'min_hessian': 25},
                  'acid': {'boosting_type': 'gbdt', 'objective': 'mape', 'num_leaves': 45,
                           'learning_rate': 0.02, 'verbose': -1, 'min_data_in_leaf': 10,
                           'feature_fraction': 1, 'min_hessian': 55}}

        for model_type in cols_to_include.keys():
            validation_results = build_models(model_type, processed_train_data, processed_val_data, params[model_type],
                                              response, cols_to_include[model_type], train_ratio, max_train_ratio,
                                              tuning_params, validation_results)

    else:
        print('Invalid value for modeling approach, must be parameter_tuning or single_model.')

# Summarize validation results
validation_results.Best_Num_Iters = validation_results.Best_Num_Iters.astype(int)
validation_summary = validation_results.groupby(['Model_Type', 'Num_Leaves', 'Min_Data_In_Leaf', 'Feature_Fraction', 'Min_Hessian']).\
    agg({'Best_MAPE': np.mean, 'Best_Num_Iters': np.median}).reset_index()
validation_summary.Best_Num_Iters = validation_summary.Best_Num_Iters.astype(int)
validation_summary = validation_summary.loc[validation_summary.groupby('Model_Type')['Best_MAPE'].idxmin()]

end_time = time.time()
print('Total time taken for hyperparameter tuning: ' + str(datetime.timedelta(seconds=end_time - start_time)))

# Determine the appropriate hyperparameters for final model tuning
test_iterations = calculate_validation_metrics(validation_summary)

#
# if modeling_approach == 'single_model':
#     # Train on full data and make predictions
#     print('')
#     print('Training full model and making test set predictions...')
#     predict_test_values(raw_data, test_data, start_times, metadata, path,
#                         params, response, test_iterations, cols_to_include, labels)
