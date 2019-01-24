import pandas as pd
import numpy as np
import lightgbm as lgb
import re
from FeatureEngineering import engineer_features, calculate_start_times, remove_phases
from Modeling import build_lgbm_validation_datasets, build_lgbm_test_datasets
from Predict import predict_test_values

# Read in all data sets
# Additionally, determine start times for each process
if 'raw_data' not in locals():
    print('Reading in data sets...')
    raw_data = pd.read_csv("train_values.csv")
    labels = pd.read_csv('train_labels.csv')
    test_data = pd.read_csv("test_values.csv")
    print('Successfully read in data sets.')

    # Determine start times for train and test data
    # Necessary to properly do walk forward validation
    print('Determining start times...')
    train_process_start_times = calculate_start_times(raw_data).sort_values(by='timestamp')
    test_process_start_times = calculate_start_times(test_data)
    print('Start times successfully determined.')
else:
    print('Data already read in, skipping data read and start time calculations.')

# Create walk forward train/validation splits
num_total_rows = train_process_start_times.shape[0]
validation_results = pd.DataFrame(columns=['Model_Type', 'Train_Ratio', 'Best_MAPE', 'Best_Num_Iters'])
response = 'final_rinse_total_turbidity_liter'
train_val_ratios = list(np.linspace(0.6, 0.9, 7))

for train_ratio in train_val_ratios:
    print('')
    print('Training with ' + str(round(train_ratio * 100, 1)) + '% of training data...')

    # Identify which processes will be train and which will be validation
    num_train_rows = int(round(train_ratio * num_total_rows))
    train_processes = pd.DataFrame(train_process_start_times.process_id[0:num_train_rows])
    val_processes = pd.DataFrame(train_process_start_times.process_id[num_train_rows:num_total_rows])

    # Delete phases in validation data to match distribution of test data
    # 10% stop at pre-rinse, 30% at caustic, 30% at intermediate rinse, 30% full data
    num_val_processes = val_processes.shape[0]
    data_truncation_type = ['pre_rinse'] * int(round(0.1 * num_val_processes)) +\
                           ['caustic'] * int(round(0.3 * num_val_processes)) +\
                           ['int_rinse'] * int(round(0.3 * num_val_processes)) +\
                           ['acid'] * (num_val_processes - int(round(0.1 * num_val_processes)) -
                                       2 * int(round(0.3 * num_val_processes)))
    val_processes['type'] = data_truncation_type

    raw_train_data = raw_data[raw_data.process_id.isin(train_processes.process_id)]
    raw_val_data = raw_data[raw_data.process_id.isin(val_processes.process_id)]
    raw_val_data = raw_val_data.merge(val_processes, on='process_id')

    raw_val_data = remove_phases(raw_val_data)

    # Engineer phase-level features on train, validation, and test sets
    print('Engineering features on train, validation sets...')
    processed_train_data = engineer_features(raw_train_data, train_process_start_times)
    processed_val_data = engineer_features(raw_val_data, train_process_start_times)
    print('Successfully engineered features.')

    # Bring in labels and start times for train and validation data
    processed_train_data = processed_train_data.merge(labels, on='process_id').\
                                                sort_values(by='timestamp')

    processed_val_data = processed_val_data.merge(labels, on='process_id').\
                                            sort_values(by='timestamp')

    # Convert object id to category
    # Ensure that categories are consistent across training, validation, and test sets
    processed_train_data.object_id = processed_train_data.object_id.astype('category')
    processed_val_data.object_id = processed_val_data.object_id.astype('category', categories=processed_train_data.object_id.cat.categories)

    # Bring type back in to processed data
    # Needed to create multiple eval sets
    processed_val_data = processed_val_data.merge(val_processes, on='process_id')
    #processed_test_data.to_csv('test_processed_data.csv')

    cols_to_drop = ['process_id', 'timestamp', 'pipeline', 'day_of_week', 'previous_run_start_time', 'previous_object']
    cols_to_include = list(filter(lambda x: re.search(r'(?=.*pre_rinse)', x), list(processed_train_data.columns))) +\
                      ['object_id']

    # Modeling
    modeling_data = build_lgbm_validation_datasets(processed_train_data, processed_val_data, response, cols_to_include=cols_to_include)

    # specify your configurations as a dict
    params = {
        'boosting_type': 'gbdt',
        'objective': 'mape',
        'num_leaves': 63,
        'learning_rate': 0.01,
        'verbose': -1
    }

    print('Starting training...')
    # train
    gbm_prerinse = lgb.train(params,
                             modeling_data['train'],
                             num_boost_round=2000,
                             valid_sets=modeling_data['eval_pre_rinse'],
                             verbose_eval=2000,
                             early_stopping_rounds=50)

    if train_ratio == 0.9:
        lgb.plot_importance(gbm_prerinse, importance_type='gain')

    validation_results = validation_results.append(pd.DataFrame([['Pre-rinse',
                                                                  train_ratio,
                                                                  round(gbm_prerinse.best_score['valid_0']['mape'], 5),
                                                                  gbm_prerinse.best_iteration]],
                                                   columns=validation_results.columns))

    # Caustic-specific model
    cols_to_include = list(filter(lambda x: re.search(r'(?=.*pre_rinse|.*caustic)', x),
                                  list(processed_train_data.columns))) + \
                      ['object_id']

    modeling_data = build_lgbm_validation_datasets(processed_train_data, processed_val_data, response,
                                                   cols_to_include=cols_to_include)

    gbm_caustic = lgb.train(params,
                            modeling_data['train'],
                            num_boost_round=2000,
                            valid_sets=modeling_data['eval_caustic'],
                            verbose_eval=2000,
                            early_stopping_rounds=50)

    validation_results = validation_results.append(pd.DataFrame([['Caustic',
                                                                  train_ratio,
                                                                  round(gbm_caustic.best_score['valid_0']['mape'], 5),
                                                                  gbm_caustic.best_iteration]],
                                                                columns=validation_results.columns))

    # Intermediate rinse-specific model
    cols_to_include = list(filter(lambda x: re.search(r'(?=.*pre_rinse|.*caustic|.*intermediate)', x),
                                  list(processed_train_data.columns))) + \
                      ['object_id']

    modeling_data = build_lgbm_validation_datasets(processed_train_data, processed_val_data, response,
                                                   cols_to_include=cols_to_include)

    gbm_int_rinse = lgb.train(params,
                         modeling_data['train'],
                         num_boost_round=2000,
                         valid_sets=modeling_data['eval_int_rinse'],
                         verbose_eval=2000,
                         early_stopping_rounds=50)

    validation_results = validation_results.append(pd.DataFrame([['Intermediate Rinse',
                                                                  train_ratio,
                                                                  round(gbm_int_rinse.best_score['valid_0']['mape'], 5),
                                                                  gbm_int_rinse.best_iteration]],
                                                                columns=validation_results.columns))

    # Acid-specific model
    cols_to_include = list(filter(lambda x: re.search(r'(?=.*pre_rinse|.*caustic|.*intermediate|.*acid)', x),
                                  list(processed_train_data.columns))) + \
                           ['object_id']

    modeling_data = build_lgbm_validation_datasets(processed_train_data, processed_val_data, response,
                                                   cols_to_include=cols_to_include)

    gbm_acid = lgb.train(params,
                         modeling_data['train'],
                         num_boost_round=2000,
                         valid_sets=modeling_data['eval_acid'],
                         verbose_eval=2000,
                         early_stopping_rounds=50)

    validation_results = validation_results.append(pd.DataFrame([['Acid',
                                                                  train_ratio,
                                                                  round(gbm_acid.best_score['valid_0']['mape'], 5),
                                                                  gbm_acid.best_iteration]],
                                                   columns=validation_results.columns))

# Show validation results
test_iterations_pre_rinse = int(round(validation_results[validation_results.Model_Type == 'Pre-rinse'].Best_Num_Iters.mean()))
test_iterations_acid = int(round(validation_results[validation_results.Model_Type == 'Acid'].Best_Num_Iters.mean()))
test_iterations_caustic = int(round(validation_results[validation_results.Model_Type == 'Caustic'].Best_Num_Iters.mean()))
test_iterations_int_rinse = int(round(validation_results[validation_results.Model_Type == 'Intermediate Rinse'].Best_Num_Iters.mean()))
est_error_pre_rinse = round(validation_results[validation_results.Model_Type == 'Pre-rinse'].Best_MAPE.mean(), 4)
est_error_acid = round(validation_results[validation_results.Model_Type == 'Acid'].Best_MAPE.mean(), 4)
est_error_caustic = round(validation_results[validation_results.Model_Type == 'Pre-rinse'].Best_MAPE.mean(), 4)
est_error_int_rinse = round(validation_results[validation_results.Model_Type == 'Intermediate Rinse'].Best_MAPE.mean(), 4)

print(validation_results.sort_values(by=['Model_Type', 'Train_Ratio']))
print('Best Iterations, pre-rinse model: ' + str(test_iterations_pre_rinse))
print('Best Iterations, caustic model: ' + str(test_iterations_caustic))
print('Best Iterations, intermediate-rinse model: ' + str(test_iterations_int_rinse))
print('Best Iterations, acid model: ' + str(test_iterations_acid))
print('')
print('Estimated error for pre-rinse predictions: ' + str(est_error_pre_rinse))
print('Estimated error for caustic predictions: ' + str(est_error_caustic))
print('Estimated error for intermediate-rinse predictions: ' + str(est_error_int_rinse))
print('Estimated error for acid predictions: ' + str(est_error_acid))
print('')
print('Estimated total error for all predictions: ' + str(round(0.1 * est_error_pre_rinse +\
                                                                0.3 * est_error_caustic +\
                                                                0.3 * est_error_int_rinse +\
                                                                0.3 * est_error_acid, 4)))


#
# # Train on full data and make predictions
# print('')
# print('Training full model and making test set predictions...')
# predict_test_values(raw_data, train_process_start_times, test_data, test_process_start_times,
#                     params, response, test_iterations, labels, cols_to_include)
#
#




#
# # Process test data
# print('Processing test data...')
# test_process_start_times['process_id'] = test_process_start_times.process_id.astype(int)
#
# df_test_output = engineer_features(test_data)
# df_test_output = df_test_output.replace(0, np.nan)
#
# df_test_output = df_test_output.merge(test_process_start_times, on='process_id').sort_values(by='timestamp')
# df_test_output['day_of_week'] = df_test_output.timestamp.dt.date
# df_test_output['cumulative_runs_day'] = df_test_output.groupby(['pipeline', 'day_of_week']).cumcount()
# df_test_output['previous_object'] = df_test_output.groupby(['pipeline', 'day_of_week'])['object_id'].shift(1)
# df_test_output['previous_run_start_time'] = df_test_output.groupby(['pipeline', 'day_of_week']).timestamp.shift(1)
# df_test_output['previous_run_delta'] = (df_test_output.timestamp - df_test_output.previous_run_start_time).astype('timedelta64[s]')
# test_data.object_id = test_data.object_id.astype(str)
#
#
# #processed_train_data = processed_train_data[processed_train_data.object_id.isin(df_test_output.object_id)].replace(0, np.nan)
# processed_train_data = processed_train_data.replace(0, np.nan).sort_values(by=['timestamp'])
# processed_train_data['day_of_week'] = processed_train_data.timestamp.dt.date
# processed_train_data['cumulative_runs_day'] = processed_train_data.groupby(['pipeline', 'day_of_week']).cumcount()
# processed_train_data['previous_object'] = processed_train_data.groupby(['pipeline', 'day_of_week'])['object_id'].shift(1)
# processed_train_data['previous_run_start_time'] = processed_train_data.groupby(['pipeline', 'day_of_week']).timestamp.shift(1)
# processed_train_data['previous_run_delta'] = (processed_train_data.timestamp - processed_train_data.previous_run_start_time).astype('timedelta64[s]')
# processed_train_data.to_csv('modeling_data.csv')