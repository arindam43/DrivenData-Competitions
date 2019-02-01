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
from FeatureEngineering import engineer_features, remove_outliers

try:
    reload(sys.modules['Modeling'])
except KeyError:
    pass
from Modeling import build_test_models


def predict_test_values(raw_data, train_start_times, test_data, test_start_times,
                        params, response, test_iterations, cols_to_include, labels):

    y_test_pred = []

    # Build data set on full training data
    processed_full_train_data = engineer_features(raw_data, train_start_times)
    processed_full_train_data = processed_full_train_data.merge(labels, on='process_id').\
                                                          sort_values(by='start_time')

    # Remove training data outliers
    processed_full_train_data = remove_outliers(processed_full_train_data)

    # Build data set on test data
    processed_test_data = engineer_features(test_data, test_start_times)
    processed_test_data = processed_test_data.sort_values(by='process_id')

    # Clipping experiments
    quantiles = (processed_full_train_data.groupby('object_id')[response].quantile(0.9) * 1.5).reset_index()
    quantiles.columns = ['object_id', 'response']
    processed_full_train_data = processed_full_train_data.merge(quantiles, on='object_id')
    processed_full_train_data[response] = np.where(processed_full_train_data.response < processed_full_train_data[response],
                                                   processed_full_train_data.response,
                                                   processed_full_train_data[response])

    # Align categories across full training data and test set
    processed_full_train_data.object_id = processed_full_train_data.object_id.astype('category')
    processed_test_data.object_id = processed_test_data.object_id.astype('category', categories=processed_full_train_data.object_id.cat.categories)
    processed_full_train_data.to_csv('full_modeling_data.csv')
    processed_test_data.to_csv('processed_test_data.csv')

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
