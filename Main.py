from importlib import reload
import pandas as pd
import numpy as np
import re
import sys

try:
    reload(sys.modules['FeatureEngineering'])
except KeyError:
    pass
from FeatureEngineering import engineer_features, calculate_start_times, remove_outliers

try:
    reload(sys.modules['Modeling'])
except KeyError:
    pass
from Modeling import build_models, calculate_validation_metrics

try:
    reload(sys.modules['Predict'])
except KeyError:
    pass
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
    train_start_times = calculate_start_times(raw_data).sort_values(by='timestamp')
    test_start_times = calculate_start_times(test_data)
    print('Start times successfully determined.')
else:
    print('Data already read in, skipping data read and start time calculations.')


# Create walk forward train/validation splits
validation_results = pd.DataFrame(columns=['Model_Type', 'Train_Ratio', 'Best_MAPE', 'Best_Num_Iters'])
response = 'final_rinse_total_turbidity_liter'
train_val_ratios = list(range(40, 54, 4))
max_train_ratio = max(train_val_ratios)

for train_ratio in train_val_ratios:
    print('')
    print('Training with first ' + str(int(train_ratio)) + ' days of training data...')

    # Identify which processes will be train and which will be validation
    train_processes = pd.DataFrame(train_start_times.process_id[train_start_times.day_number <= train_ratio])
    val_processes = pd.DataFrame(train_start_times.process_id[train_start_times.day_number > train_ratio])

    raw_train_data = raw_data[raw_data.process_id.isin(train_processes.process_id)]
    raw_val_data = raw_data[raw_data.process_id.isin(val_processes.process_id)]

    # Engineer phase-level features on train, validation, and test sets
    print('Engineering features on train, validation sets...')
    processed_train_data = engineer_features(raw_train_data, train_start_times)
    processed_val_data = engineer_features(raw_val_data, train_start_times)
    print('Successfully engineered features.')

    # Remove outliers from training data
    processed_train_data = remove_outliers(processed_train_data)

    # Bring in labels for train and validation data
    processed_train_data = processed_train_data.merge(labels, on='process_id')
    processed_val_data = processed_val_data.merge(labels, on='process_id')

    # Convert object id to category
    # Ensure that categories are consistent across training, validation, and test sets
    for col in ['object_id']:
        processed_train_data[col] = processed_train_data[col].astype('category')
        processed_val_data[col] = processed_val_data[col].astype('category', categories=processed_train_data['object_id'].cat.categories)

    # processed_test_data.to_csv('test_processed_data.csv')

    non_phase_cols = ['object_id']
    cols_to_include = {'pre_rinse': list(filter(lambda x: re.search(r'(?=.*pre_rinse)', x), list(processed_train_data.columns))) + non_phase_cols,
                       'caustic': list(filter(lambda x: re.search(r'(?=.*pre_rinse|.*caustic)', x), list(processed_train_data.columns))) + non_phase_cols,
                       'int_rinse': list(filter(lambda x: re.search(r'(?=.*pre_rinse|.*caustic|.*intermediate)', x), list(processed_train_data.columns))) + non_phase_cols,
                       'acid': list(filter(lambda x: re.search(r'(?=.*pre_rinse|.*caustic|.*intermediate|.*acid)', x), list(processed_train_data.columns))) + non_phase_cols
                       #'acid': list(set(processed_train_data.columns) - set(['object_id', 'process_id', 'pipeline', 'day_number', 'timestamp', response])) + non_phase_cols
                       }

    # specify your configurations as a dict
    params = {
        'boosting_type': 'gbdt',
        'objective': 'mape',
        'num_leaves': 63,
        'learning_rate': 0.01,
        'verbose': -1,
    }

    for model_type in cols_to_include.keys():
        validation_results = build_models(model_type, processed_train_data, processed_val_data, params,
                                          response, cols_to_include[model_type], train_ratio, max_train_ratio, validation_results)

test_iterations = calculate_validation_metrics(validation_results)

#processed_train_data = processed_train_data.sort_values(by=['object_id', 'timestamp'])

# Train on full data and make predictions
print('')
print('Training full model and making test set predictions...')
predict_test_values(raw_data, train_start_times, test_data, test_start_times,
                    params, response, test_iterations, cols_to_include, labels)






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
