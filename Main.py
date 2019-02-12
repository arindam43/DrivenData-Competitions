from importlib import reload
import pandas as pd
import numpy as np
import os, sys, itertools, time, datetime

try:
    for module in ['FeatureEngineering', 'Modeling', 'Predict', 'Ingest']:
        reload(sys.modules[module])
except KeyError:
    pass

from FeatureEngineering import create_model_datasets
from Modeling import build_models, calculate_validation_metrics, subset_modeling_columns
from Predict import predict_test_values
from Ingest import ingest_data, preprocess_data


# Read in and pre-process phase-timestamp-level data
path = os.getcwd() + '\\Data\\'
if 'raw_data' not in locals():
    raw_data, labels, metadata, test_data, start_times = ingest_data(path)

    # Pre-process data
    raw_data, return_phases, supply_phases = preprocess_data(raw_data, start_times)
    test_data = preprocess_data(test_data, start_times, return_phases, supply_phases)

    # Pre-process metadata by converting the one-hot encoded recipe types to a single column
    # Only 3 recipe types in total: pre_rinse + caustic, all phases, and acid only
    metadata['recipe_type'] = np.where(metadata.caustic == 0, 'acid_only',
                                       np.where(metadata.intermediate_rinse == 1, 'full_clean', 'short_clean'))
    metadata = metadata[['process_id', 'recipe_type']]
else:
    print('Raw data already found, skipping data read and initial pre-processing.')


if 'train_eda' not in locals():
    train_eda = raw_data.groupby('object_id').describe(percentiles=[0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])

if 'test_eda' not in locals():
    test_eda = test_data.describe(percentiles=[0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])


# Create walk forward train/validation splits
validation_results = pd.DataFrame(columns=['Model_Type', 'Train_Ratio', 'Num_Leaves', 'Min_Data_In_Leaf', 'Min_Hessian',
                                           'Best_MAPE', 'Best_Num_Iters'])
response = 'final_rinse_total_turbidity_liter'
train_val_ratios = list(range(40, 53, 4))  # training set sizes of 40, 44, 48, and 52 days
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
        processed_val_data = create_model_datasets(raw_train_data, raw_val_data, start_times, labels, metadata,
                                                   path, val_or_test='validation')

    # Create dictionary of columns to be included in each of the four models
    cols_to_include = subset_modeling_columns(processed_train_data)

    modeling_approach = 'single_model'
    learning_rate = 0.02

    # Hyperparameter tuning - simple grid search
    if modeling_approach == 'parameter_tuning':
        leaves_tuning = [31, 45, 63, 75]
        min_data_tuning = [10, 20]
        min_hessian_tuning = [10, 25, 40, 55, 70]

        tuning_grid = list(itertools.product(leaves_tuning, min_data_tuning, min_hessian_tuning))
        counter = 1

        for tuning_params in tuning_grid:
            print('')
            print('Hyperparameter tuning, model ' + str(counter) + ' of ' + str(len(tuning_grid)) + ', train ratio = '
                  + str(train_ratio) + '...')
            print('num_leaves: ' + str(tuning_params[0]))
            print('min_data: ' + str(tuning_params[1]))
            print('min_hessian: ' + str(tuning_params[2]))

            # specify your configurations as a dict
            params = {'pre_rinse': {'boosting_type': 'gbdt', 'objective': 'mape', 'num_leaves': tuning_params[0],
                                    'learning_rate': learning_rate, 'verbose': -1, 'min_data': tuning_params[1],
                                    'min_hessian': tuning_params[2]},
                      'caustic': {'boosting_type': 'gbdt', 'objective': 'mape', 'num_leaves': tuning_params[0],
                                  'learning_rate': learning_rate, 'verbose': -1, 'min_data': tuning_params[1],
                                  'min_hessian': tuning_params[2]},
                      'int_rinse': {'boosting_type': 'gbdt', 'objective': 'mape', 'num_leaves': tuning_params[0],
                                    'learning_rate': learning_rate, 'verbose': -1, 'min_data': tuning_params[1],
                                    'min_hessian': tuning_params[2]},
                      'acid': {'boosting_type': 'gbdt', 'objective': 'mape', 'num_leaves': tuning_params[0],
                               'learning_rate': learning_rate, 'verbose': -1, 'min_data': tuning_params[1],
                               'min_hessian': tuning_params[2]},
                      }

            for model_type in cols_to_include.keys():
                validation_results = build_models(model_type, processed_train_data, processed_val_data,
                                                  params[model_type], response, cols_to_include[model_type],
                                                  train_ratio, max_train_ratio, tuning_params, validation_results, False)
            counter = counter + 1

    elif modeling_approach == 'single_model':
        tuning_params = ('NA', 'NA', 'NA', 'NA')
        # specify your configurations as a dict
        params = {'pre_rinse': {'boosting_type': 'gbdt', 'objective': 'mape', 'num_leaves': 63,
                                'learning_rate': learning_rate, 'verbose': -1, 'min_data': 10,'min_hessian': 40},
                  'caustic': {'boosting_type': 'gbdt', 'objective': 'mape', 'num_leaves': 63,
                              'learning_rate': learning_rate, 'verbose': -1, 'min_data': 10,'min_hessian': 40},
                  'int_rinse': {'boosting_type': 'gbdt', 'objective': 'mape', 'num_leaves': 63,
                                'learning_rate': learning_rate, 'verbose': -1, 'min_data': 10, 'min_hessian': 25},
                  'acid': {'boosting_type': 'gbdt', 'objective': 'mape', 'num_leaves': 63,
                           'learning_rate': learning_rate, 'verbose': -1, 'min_data': 10, 'min_hessian': 25}}

        for model_type in cols_to_include.keys():
            validation_results = build_models(model_type, processed_train_data, processed_val_data, params[model_type],
                                              response, cols_to_include[model_type], train_ratio, max_train_ratio,
                                              tuning_params, validation_results, True)

    else:
        print('Invalid value for modeling approach, must be parameter_tuning or single_model.')

# Summarize validation results
validation_results.Best_Num_Iters = validation_results.Best_Num_Iters.astype(int)
validation_summary = validation_results.groupby(['Model_Type', 'Num_Leaves', 'Min_Data_In_Leaf', 'Min_Hessian']).\
    agg({'Best_MAPE': np.mean, 'Best_Num_Iters': np.median}).reset_index()
validation_summary.Best_Num_Iters = validation_summary.Best_Num_Iters.astype(int)
validation_summary = validation_summary.loc[validation_summary.groupby('Model_Type')['Best_MAPE'].idxmin()]

end_time = time.time()
print('Total time taken for hyperparameter tuning: ' + str(datetime.timedelta(seconds=end_time - start_time)))

# Determine the appropriate hyperparameters for final model tuning
test_iterations = calculate_validation_metrics(validation_summary)


# Train on full data and make predictions
print('')
print('Training full model and making test set predictions...')
predict_test_values(raw_data, test_data, start_times, metadata, path,
                    params, response, test_iterations, labels, cols_to_include)
