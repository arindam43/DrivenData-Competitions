from importlib import reload
import pandas as pd
import numpy as np
import os, sys, itertools, time, datetime, csv
from matplotlib import pyplot as plt

try:
    for module in ['FeatureEngineering', 'Modeling', 'Predict', 'Ingest']:
        reload(sys.modules[module])
except KeyError:
    pass

from FeatureEngineering import create_model_datasets
from Modeling import build_models, calculate_validation_metrics, select_model_columns
from Predict import predict_test_values
from Ingest import ingest_data, preprocess_data


# Read in and pre-process phase-timestamp-level data
path = os.getcwd() + '\\Data\\'
if 'raw_data' not in locals():
    raw_data, labels, metadata, test_data, start_times = ingest_data(path)

    # Pre-process data
    raw_data, return_phases, supply_phases = preprocess_data(raw_data, test_data, start_times)
    test_data = preprocess_data(test_data, test_data, start_times, return_phases, supply_phases)

    # Pre-process metadata by converting the one-hot encoded recipe types to a single column
    # Only 3 recipe types in total: pre_rinse + caustic, all phases, and acid only
    metadata['recipe_type'] = np.where(metadata.caustic == 0, 'acid_only',
                                       np.where(metadata.intermediate_rinse == 1, 'full_clean', 'short_clean'))
    metadata = metadata[['process_id', 'recipe_type']]
    #
    # # EDA
    # train_eda = raw_data.describe(
    #     percentiles=[0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])
    #
    # test_eda = test_data.describe(percentiles=[0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])

else:
    print('Raw data already found, skipping data read and initial pre-processing.')

# Create walk forward train/validation splits
validation_results = pd.DataFrame(columns=['Model_Type', 'Train_Ratio', 'Excluded_Cols',
                                           'Num_Leaves', 'Min_Data_In_Leaf', 'Min_Gain',
                                           'Best_MAPE', 'Best_Num_Iters'])
response = 'final_rinse_total_turbidity_liter'
column_selection_mode = 'none'
modeling_approach = 'single_model'
train_val_ratios = list(range(44, 57, 4))  # training set sizes of 40, 44, 48, and 52 days
max_train_ratio = max(train_val_ratios)
start_time = time.time()


for train_ratio in train_val_ratios:
    print('')
    print('Training with first ' + str(int(train_ratio)) + ' days of training data...')

    # Identify which processes will be used for training and which will be used for validation
    train_processes = pd.Series(start_times.process_id[start_times.day_number <= train_ratio])
    val_processes = pd.DataFrame(start_times.process_id[start_times.day_number > train_ratio])

    # Split data into training and validation sets
    raw_train_data = raw_data[raw_data.process_id.isin(train_processes)]
    raw_val_data = raw_data[raw_data.process_id.isin(val_processes.process_id)]

    # Process the raw data to create model-ready datasets
    # Features engineering, aggregation to process_id level, outlier removal
    processed_train_data, \
        processed_val_data = create_model_datasets(raw_train_data, raw_val_data, start_times, labels, response,
                                                   metadata, path, val_or_test='validation')

    if column_selection_mode == 'grid':
        grid_1 = ['residue_acid', 'turb_acid', 'residue_acid|.*turb_acid']
        grid_2 = ['residue_caustic', 'turb_caustic', 'residue_caustic|.*turb_caustic']
        grid_3 = ['residue_pre_rinse', 'turb_pre_rinse', 'residue_pre_rinse|.*turb_pre_rinse']
        grid_4 = ['residue_int_rinse', 'turb_int_rinse', 'residue_int_rinse|.*turb_int_rinse']
        cols_subset = list(itertools.product(grid_1, grid_2, grid_3, grid_4))
    else:
        cols_subset = [None]

    for cols in cols_subset:
        if cols is not None:
            print('Column subset evaluated: ' + '|.*'.join(cols))
            cols = '|.*'.join(cols)

        # Create dictionary of columns to be included in each of the four models
        cols_to_include = select_model_columns(processed_train_data, cols)

        learning_rate = 0.01

        # Hyperparameter tuning - simple grid search
        if modeling_approach == 'parameter_tuning':
            leaves_tuning = [63]
            min_data_tuning = [20, 30, 40, 50]
            min_gain_tuning = [0, 2.5e-12, 5e-12, 7.5e-12]

            tuning_grid = list(itertools.product(leaves_tuning, min_data_tuning, min_gain_tuning))
            counter = 1

            for tuning_params in tuning_grid:
                print('')
                print('Hyperparameter tuning, model ' + str(counter) + ' of ' + str(len(tuning_grid)) +
                      ', train ratio = ' + str(train_ratio) + '...')
                print('num_leaves: ' + str(tuning_params[0]))
                print('min_data: ' + str(tuning_params[1]))
                print('min_gain: ' + str(tuning_params[2]))

                param_config = {'boosting_type': 'gbdt', 'objective': 'mape', 'num_leaves': tuning_params[0],
                                'learning_rate': learning_rate, 'verbose': -1, 'min_data': tuning_params[1],
                                'min_split_gain': tuning_params[2]}
                phases = ['pre_rinse', 'caustic', 'int_rinse', 'acid']
                params = {}

                for phase in phases:
                    params[phase] = param_config

                for model_type in cols_to_include.keys():
                    validation_results = build_models(model_type, processed_train_data, processed_val_data,
                                                      params[model_type], response, cols_to_include[model_type],
                                                      train_ratio, max_train_ratio, tuning_params, validation_results,
                                                      cols, False)
                counter = counter + 1

        elif modeling_approach == 'single_model':
            tuning_params = ('NA', 'NA', 'NA', 'NA')
            # specify your configurations as a dict
            params = {'pre_rinse': {'boosting_type': 'gbdt', 'objective': 'mape', 'num_leaves': 63,
                                    'learning_rate': learning_rate, 'verbose': -1, 'min_data': 50,
                                    'min_split_gain': 0},
                      'caustic':   {'boosting_type': 'gbdt', 'objective': 'mape', 'num_leaves': 63,
                                    'learning_rate': learning_rate, 'verbose': -1, 'min_data': 40,
                                    'min_split_gain': 2.5e-12},
                      'int_rinse': {'boosting_type': 'gbdt', 'objective': 'mape', 'num_leaves': 63,
                                    'learning_rate': learning_rate, 'verbose': -1, 'min_data': 40,
                                    'min_split_gain': 5e-12},
                      'acid':      {'boosting_type': 'gbdt', 'objective': 'mape', 'num_leaves': 63,
                                    'learning_rate': learning_rate, 'verbose': -1, 'min_data': 40,
                                    'min_split_gain': 5e-12}}

            for model_type in cols_to_include.keys():
                validation_results = build_models(model_type, processed_train_data, processed_val_data,
                                                  params[model_type], response, cols_to_include[model_type],
                                                  train_ratio, max_train_ratio, tuning_params, validation_results,
                                                  cols, True)

        else:
            print('Invalid value for modeling approach, must be parameter_tuning or single_model.')

end_time = time.time()
print('Total time taken for hyperparameter tuning: ' + str(datetime.timedelta(seconds=end_time - start_time)))

# Summarize validation results
validation_results.Best_Num_Iters = validation_results.Best_Num_Iters.astype(int)

validation_groupby = ['Model_Type', 'Num_Leaves', 'Excluded_Cols', 'Min_Data_In_Leaf', 'Min_Gain']
validation_summary = validation_results.groupby(validation_groupby).\
    agg({'Best_MAPE': np.mean, 'Best_Num_Iters': np.median}).reset_index()
validation_summary.Best_Num_Iters = validation_summary.Best_Num_Iters.astype(int)

# Determine best hyperparameters for final model tuning
validation_best = validation_summary.loc[validation_summary.groupby('Model_Type')['Best_MAPE'].idxmin()]
test_iterations = calculate_validation_metrics(validation_best)


# Train on full data and make predictions
print('')
print('Training full model and making test set predictions...')
predict_test_values(raw_data, test_data, start_times, metadata, path,
                    params, response, test_iterations, labels, cols_to_include)

