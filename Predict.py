import pandas as pd
import numpy as np
import sys
import os
from importlib import reload
from datetime import datetime

try:
    reload(sys.modules['FeatureEngineering'])
except KeyError:
    pass
from FeatureEngineering import engineer_features, remove_outliers, create_model_datasets

try:
    reload(sys.modules['Modeling'])
except KeyError:
    pass
from Modeling import build_test_models


def predict_test_values(raw_data, test_data, start_times, metadata, path,
                        params, response, test_iterations, cols_to_include, labels):

    y_test_pred = []

    processed_full_train_data, \
    processed_test_data = create_model_datasets(raw_data, test_data, start_times, labels, metadata,
                                                path, type='test')

    for model_type in cols_to_include.keys():
        y_test_pred = build_test_models(model_type, processed_full_train_data, processed_test_data,
                                        response, params[model_type], test_iterations, cols_to_include[model_type], y_test_pred)

    y_test_pred_final = pd.concat(y_test_pred).sort_values(by='process_id')


    # Handle negative values by setting them equal to the lowest predicted value
    current_time = str(datetime.now().replace(microsecond=0)).replace(':', '.')
    current_directory = os.getcwd()

    y_test_pred_final.loc[y_test_pred_final[response] < 0, response] = y_test_pred_final.loc[y_test_pred_final[response] > 0, response].min()
    outputfilename = current_directory + '\\Predictions\\Test Predictions ' + current_time + '.csv'
    y_test_pred_final.to_csv(outputfilename, index=False)

    y_test_pred_final = y_test_pred_final.sort_values(by=response)
    outputfilename = current_directory + '\\Predictions\\Test Predictions ' + current_time + ' (Sorted).csv'
    y_test_pred_final.to_csv(outputfilename, index=False)

    test_pred_full = y_test_pred_final.merge(processed_test_data, on='process_id').sort_values(by='process_id')
    outputfilename = current_directory + '\\Predictions\\Test Predictions ' + current_time + ' (Full).csv'
    test_pred_full.to_csv(outputfilename, index=False)

    print('Test set predictions made at ' + str(current_time) +' saved to csv file.')
