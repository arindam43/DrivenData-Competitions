import pandas as pd
import numpy as np
import logging as logger
import os
import sys
import itertools
import time
import datetime

logger.basicConfig(
    level=logger.INFO,
    format='%(levelname)s %(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S',
    handlers=[
        logger.StreamHandler(sys.stdout)
    ])

from engineer_features import create_model_datasets
from build_models import build_models, calculate_validation_metrics, select_model_columns
from make_predictions import predict_test_values
from ingest_data import ingest_data, preprocess_data


def main(path, make_predictions=True, feature_selection=False, hyperparameter_tuning=False):
    logger.info('*****Data Ingestion*****')

    # Read in all relevant source data files
    raw_data, labels, metadata, test_data, start_times = ingest_data(path)

    # Pre-process data
    raw_data, return_phases, supply_phases = preprocess_data(raw_data, test_data, start_times)
    test_data = preprocess_data(test_data, test_data, start_times, return_phases, supply_phases)

    # Pre-process metadata by converting the one-hot encoded recipe types to a single column
    # Only 3 recipe types in total: pre_rinse + caustic, all phases, and acid only
    metadata['recipe_type'] = np.where(metadata.caustic == 0, 'acid_only',
                                       np.where(metadata.intermediate_rinse == 1, 'full_clean', 'short_clean'))
    metadata = metadata[['process_id', 'recipe_type']]

    logger.info('*****Data Ingestion Complete*****\n')

    # Initialize output data frames, modeling parameters
    response = 'final_rinse_total_turbidity_liter'
    validation_results = pd.DataFrame(columns=['Model_Type', 'Train_Ratio', 'Excluded_Cols',
                                               'Num_Leaves', 'Min_Data_In_Leaf', 'Min_Gain',
                                               'Best_MAPE', 'Best_Num_Iters'])
    validation_predictions = pd.DataFrame(columns=['process_id', response, 'train_ratio', 'model_type',
                                                   'predicted_response'])

    # Specify how walk-forward/rolling origin training and validation splits will be made
    train_val_ratios = list(range(44, 57, 4))  # training set sizes of 44, 48, 52, and 56 days
    # train_val_ratios = [56]
    max_train_ratio = max(train_val_ratios)
    start_time = time.time()

    # For each train/validation split, build a model on the training data and make predictions on validation set
    logger.info('*****Train/Validation Set Modeling*****')
    for train_ratio in train_val_ratios:
        logger.info('Training with first ' + str(int(train_ratio)) + ' days of training data...')

        # Identify which processes will be used for training and which will be used for validation
        train_processes = pd.Series(start_times.process_id[start_times.day_number <= train_ratio])
        val_processes = pd.DataFrame(start_times.process_id[start_times.day_number > train_ratio])

        # Split data into training and validation sets
        raw_train_data = raw_data[raw_data.process_id.isin(train_processes)]
        raw_val_data = raw_data[raw_data.process_id.isin(val_processes.process_id)]

        # Create full feature set and model-ready datasets
        # Includes feature engineering, aggregation to process_id level, outlier removal
        processed_train_data, \
            processed_val_data = create_model_datasets(raw_train_data, raw_val_data, start_times, labels, response,
                                                       metadata, path, val_or_test='validation')

        # Feature selection using grid search
        # All columns in subset are excluded using regex
        if feature_selection is True:
            grid_1 = ['residue_acid', 'turb_acid', 'residue_acid|.*turb_acid']
            grid_2 = ['residue_caustic', 'turb_caustic', 'residue_caustic|.*turb_caustic']
            grid_3 = ['residue_pre_rinse', 'turb_pre_rinse', 'residue_pre_rinse|.*turb_pre_rinse']
            grid_4 = ['residue_int_rinse', 'turb_int_rinse', 'residue_int_rinse|.*turb_int_rinse']
            cols_subset = list(itertools.product(grid_1, grid_2, grid_3, grid_4))
        elif feature_selection is False:
            cols_subset = [None]
        else:
            logger.error('Invalid value for feature_selection parameter; must be a boolean (True or False).')

        for cols in cols_subset:
            if cols is not None:
                logger.info('Column subset evaluated: ' + '|.*'.join(cols))
                cols = '|.*'.join(cols)

            # Create dict of columns to be included in each of the four models (pre rinse, caustic, int rinse, acid)
            cols_to_include = select_model_columns(processed_train_data, cols)

            learning_rate = 0.01  # should be low (not tuned; overfitting prevented with early stopping)

            # Hyperparameter tuning using grid search
            if hyperparameter_tuning is True:
                leaves_tuning = [63]
                min_data_tuning = list(range(20, 24, 3))
                min_gain_tuning = list(np.linspace(0, 1e-11, 2))

                tuning_grid = list(itertools.product(leaves_tuning, min_data_tuning, min_gain_tuning))
                counter = 1

                for tuning_params in tuning_grid:
                    logger.info('')
                    logger.info('Hyperparameter tuning, model ' + str(counter) + ' of ' + str(len(tuning_grid)) +
                                ', train ratio = ' + str(train_ratio) + '...')
                    logger.info('num_leaves: ' + str(tuning_params[0]))
                    logger.info('min_data: ' + str(tuning_params[1]))
                    logger.info('min_gain: ' + str(tuning_params[2]))

                    param_config = {'boosting_type': 'gbdt', 'objective': 'mape', 'num_leaves': tuning_params[0],
                                    'learning_rate': learning_rate, 'verbose': -1, 'min_data': tuning_params[1],
                                    'min_split_gain': tuning_params[2]}
                    phases = ['pre_rinse', 'caustic', 'int_rinse', 'acid']
                    params = {}

                    for phase in phases:
                        params[phase] = param_config

                    for model_type in cols_to_include.keys():
                        validation_predictions,\
                            validation_results = build_models(model_type, processed_train_data, processed_val_data,
                                                              params[model_type], response, cols_to_include[model_type],
                                                              train_ratio, max_train_ratio, tuning_params,
                                                              validation_results, cols)
                    counter = counter + 1
            elif hyperparameter_tuning is False:
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
                                        'min_split_gain': 2.5e-12},
                          'acid':      {'boosting_type': 'gbdt', 'objective': 'mape', 'num_leaves': 63,
                                        'learning_rate': learning_rate, 'verbose': -1, 'min_data': 40,
                                        'min_split_gain': 5e-12}}

                for model_type in cols_to_include.keys():
                    validation_predictions, \
                        validation_results = build_models(model_type, processed_train_data, processed_val_data,
                                                          params[model_type], response, cols_to_include[model_type],
                                                          train_ratio, max_train_ratio, tuning_params,
                                                          validation_results, cols, validation_predictions)

            else:
                logger.error('Invalid value for hyperparameter_tuning parameter; must be a boolean (True or False).')

        logger.info('Training with first ' + str(int(train_ratio)) + ' days of training data complete.\n')

    end_time = time.time()
    logger.info('Total time taken for walk-forward validation set training: '
                + str(datetime.timedelta(seconds=end_time - start_time)))

    # Summarize validation results
    test_iterations = calculate_validation_metrics(validation_results)

    logger.info('*****Train/Validation Set Modeling Complete*****\n')

    # Build model on full training data and make predictions on test set
    if make_predictions:
        logger.info('*****Test Set Predictions*****')
        predict_test_values(raw_data, test_data, start_times, metadata, path,
                            params, response, test_iterations, labels, cols_to_include)
        logger.info('*****Test Set Predictions Complete*****\n')


if __name__ == "__main__":
    path_to_data = os.getcwd() + '\\Data\\'
    main(path_to_data, False)
