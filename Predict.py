import os
import lightgbm as lgb
from datetime import datetime
import pandas as pd
from FeatureEngineering import engineer_features
from Modeling import build_lgbm_test_datasets
import re


def predict_test_values(raw_data, train_process_start_times, test_data, test_process_start_times,
                        params, response, test_iterations, labels):

    y_test_pred = []

    # Build data set on full training data
    processed_full_train_data = engineer_features(raw_data, train_process_start_times)
    processed_full_train_data = processed_full_train_data.merge(labels, on='process_id').\
                                                          sort_values(by='timestamp')

    # Build data set on test data
    processed_test_data = engineer_features(test_data, test_process_start_times)
    processed_test_data = processed_test_data.merge(test_process_start_times, on='process_id').\
                                              sort_values(by='process_id')

    # Align categories across full training data and test set
    processed_full_train_data.object_id = processed_full_train_data.object_id.astype('category')
    processed_test_data.object_id = processed_test_data.object_id.astype('category', categories=processed_full_train_data.object_id.cat.categories)



    # Acid model
    cols_to_include = list(filter(lambda x: re.search(r'(?=.*pre_rinse|.*caustic|.*intermediate|.*acid)', x),
                                  list(processed_full_train_data.columns))) + \
                      ['object_id', 'process_id']

    # Build lgbm data sets on full train and test data
    prediction_data = build_lgbm_test_datasets(processed_full_train_data, processed_test_data, response, cols_to_include=cols_to_include)

    # Build model on full training data to make predictions for test set
    print('Building model on full training data...')

    gbm_full = lgb.train(params,
                         prediction_data['full_train'],
                         num_boost_round=test_iterations['acid'])

    # Make predictions on test set and save to .csv
    print('Making test set predictions...')

    y_test_pred.append(pd.DataFrame({'process_id': prediction_data['test_acid'].process_id,
                                response: gbm_full.predict(prediction_data['test_acid'])}
                               ))



    # Intermediate rinse model
    cols_to_include = list(filter(lambda x: re.search(r'(?=.*pre_rinse|.*caustic|.*intermediate)', x),
                                  list(processed_full_train_data.columns))) + \
                      ['object_id', 'process_id']

    # Build lgbm data sets on full train and test data
    prediction_data = build_lgbm_test_datasets(processed_full_train_data, processed_test_data, response, cols_to_include=cols_to_include)

    # Build model on full training data to make predictions for test set
    print('Building model on full training data...')

    gbm_full = lgb.train(params,
                         prediction_data['full_train'],
                         num_boost_round=test_iterations['int_rinse'])

    # Make predictions on test set and save to .csv
    print('Making test set predictions...')

    y_test_pred.append(pd.DataFrame({'process_id': prediction_data['test_int_rinse'].process_id,
                                     response: gbm_full.predict(prediction_data['test_int_rinse'])}
                                    ))


    # Caustic model
    cols_to_include = list(filter(lambda x: re.search(r'(?=.*pre_rinse|.*caustic)', x),
                                  list(processed_full_train_data.columns))) + \
                      ['object_id', 'process_id']

    # Build lgbm data sets on full train and test data
    prediction_data = build_lgbm_test_datasets(processed_full_train_data, processed_test_data, response, cols_to_include=cols_to_include)

    # Build model on full training data to make predictions for test set
    print('Building model on full training data...')

    gbm_full = lgb.train(params,
                         prediction_data['full_train'],
                         num_boost_round=test_iterations['caustic'])

    # Make predictions on test set and save to .csv
    print('Making test set predictions...')

    y_test_pred.append(pd.DataFrame({'process_id': prediction_data['test_caustic'].process_id,
                                     response: gbm_full.predict(prediction_data['test_caustic'])}
                                    ))



    # Pre-rinse model
    cols_to_include = list(filter(lambda x: re.search(r'(?=.*pre_rinse)', x),
                                  list(processed_full_train_data.columns))) + \
                      ['object_id', 'process_id']

    # Build lgbm data sets on full train and test data
    prediction_data = build_lgbm_test_datasets(processed_full_train_data, processed_test_data, response, cols_to_include=cols_to_include)

    # Build model on full training data to make predictions for test set
    print('Building model on full training data...')

    gbm_full = lgb.train(params,
                         prediction_data['full_train'],
                         num_boost_round=test_iterations['pre_rinse'])

    # Make predictions on test set and save to .csv
    print('Making test set predictions...')

    y_test_pred.append(pd.DataFrame({'process_id': prediction_data['test_pre_rinse'].process_id,
                                     response: gbm_full.predict(prediction_data['test_pre_rinse'])}
                                    ))

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
