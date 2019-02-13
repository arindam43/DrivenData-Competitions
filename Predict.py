import pandas as pd
import numpy as np
import sys
import os
from importlib import reload
from datetime import datetime
import re

try:
    reload(sys.modules['FeatureEngineering'])
except KeyError:
    pass
from FeatureEngineering import create_model_datasets

try:
    reload(sys.modules['Modeling'])
except KeyError:
    pass
from Modeling import build_test_models


def predict_test_values(raw_data, test_data, start_times, metadata, path,
                        params, response, test_iterations, labels, cols_to_include):

    # Initialize list of predictions; will contain four dataframes corresponding to predictions from the four models
    test_predictions = []

    # Create model-ready datasets from the full training data and test data
    processed_full_train_data, \
    processed_test_data = create_model_datasets(raw_data, test_data, start_times, labels, metadata,
                                                path, val_or_test='test')

    # Build the four test models and make the predictions on the set
    for model_type in cols_to_include.keys():
        test_predictions = build_test_models(model_type, processed_full_train_data, processed_test_data,
                                             response, params[model_type], test_iterations,
                                             cols_to_include[model_type], test_predictions)

    # Combine predictions from four models into one dataframegit s
    test_predictions = pd.concat(test_predictions).sort_values(by='process_id')

    # Handle negative values by setting them equal to the lowest predicted value
    test_predictions.loc[test_predictions[response] < 0, response] = \
        test_predictions.loc[test_predictions[response] > 0, response].min()

    write_predictions_to_csv(test_predictions, processed_test_data, response)


def write_predictions_to_csv(predictions, test_data, response):
    current_time = str(datetime.now().replace(microsecond=0)).replace(':', '.')
    current_directory = os.getcwd()

    # Predictions sorted by process_id - these are submitted to leaderboard
    output_path = current_directory + '\\Predictions\\Test Predictions ' + current_time + '.csv'
    predictions.to_csv(output_path, index=False)

    # Predictions sorted by response value
    # Sanity check that the predictions are reasonable
    predictions = predictions.sort_values(by=response)
    output_path = current_directory + '\\Predictions\\Test Predictions ' + current_time + ' (Sorted).csv'
    predictions.to_csv(output_path, index=False)

    # Predictions joined to test set features
    # Useful for looking at full details of specific predictions
    test_pred_full = predictions.merge(test_data, on='process_id').sort_values(by='process_id')
    output_path = current_directory + '\\Predictions\\Test Predictions ' + current_time + ' (Full).csv'
    test_pred_full.to_csv(output_path, index=False)

    print('Test set predictions made at ' + str(current_time) +' saved to csv file.')
