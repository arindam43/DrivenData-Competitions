import pandas as pd
import numpy as np
import lightgbm as lgb
from FeatureEngineering import engineer_features, calculate_start_times
from Modeling import build_lgbm_datasets

# Read in all data sets
print('Reading in data sets...')
raw_data = pd.read_csv("train_values.csv")
labels = pd.read_csv('train_labels.csv')
test_data = pd.read_csv("test_values.csv")
print('Successfully read in data sets.')

# Process training data
print('Processing training data...')
df_process_start_times = calculate_start_times(raw_data)
df_final_output = engineer_features(raw_data)

df_final_output['object_id'] = df_final_output['object_id'].astype(str)
for col in ['process_id']:
    labels[col] = labels[col].astype(str)
    df_final_output[col] = df_final_output[col].astype(str)

joined_data = df_final_output.merge(labels, on='process_id')
joined_data.process_id = joined_data.process_id.astype(int)
joined_data = joined_data.merge(df_process_start_times, on='process_id').sort_values(by='timestamp')


print('Training data processed.')

# Process test data
print('Processing test data...')
df_test_process_start_times = calculate_start_times(test_data)
df_test_process_start_times['process_id'] = df_test_process_start_times.process_id.astype(int)

df_test_output = engineer_features(test_data)
df_test_output['process_id'] = df_test_output.process_id.astype(int)
df_test_output = df_test_output.replace(0, np.nan)

df_test_output = df_test_output.merge(df_test_process_start_times, on='process_id').sort_values(by='timestamp')
df_test_output['day_of_week'] = df_test_output.timestamp.dt.date
df_test_output['cumulative_runs_day'] = df_test_output.groupby(['pipeline', 'day_of_week']).cumcount()
df_test_output['previous_object'] = df_test_output.groupby(['pipeline', 'day_of_week'])['object_id'].shift(1)
df_test_output['previous_run_start_time'] = df_test_output.groupby(['pipeline', 'day_of_week']).timestamp.shift(1)
df_test_output['previous_run_delta'] = (df_test_output.timestamp - df_test_output.previous_run_start_time).astype('timedelta64[s]')
test_data.object_id = test_data.object_id.astype(str)


#joined_data = joined_data[joined_data.object_id.isin(df_test_output.object_id)].replace(0, np.nan)
joined_data = joined_data.replace(0, np.nan).sort_values(by=['timestamp'])
joined_data['day_of_week'] = joined_data.timestamp.dt.date
joined_data['cumulative_runs_day'] = joined_data.groupby(['pipeline', 'day_of_week']).cumcount()
joined_data['previous_object'] = joined_data.groupby(['pipeline', 'day_of_week'])['object_id'].shift(1)
joined_data['previous_run_start_time'] = joined_data.groupby(['pipeline', 'day_of_week']).timestamp.shift(1)
joined_data['previous_run_delta'] = (joined_data.timestamp - joined_data.previous_run_start_time).astype('timedelta64[s]')
joined_data.to_csv('modeling_data.csv')

joined_data.object_id = joined_data.object_id.astype(int)
#joined_data.previous_object = joined_data.previous_object.astype(int)
df_test_output.object_id = df_test_output.object_id.astype(int)
#df_test_output.previous_object = df_test_output.previous_object.astype(int)
df_test_output.to_csv('test_processed_data.csv')

print('Test data processed.')


# Modeling
cols_to_drop = ['process_id', 'timestamp', 'pipeline', 'day_of_week', 'previous_run_start_time', 'previous_object']
cols_to_include = ['object_id']
response = 'final_rinse_total_turbidity_liter'
modeling_data = build_lgbm_datasets(joined_data, df_test_output, 0.8, response, cols_to_include=cols_to_include)

# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'mape',
    'num_leaves': 63,
    'learning_rate': 0.01,
    'verbose': 0
}

print('Starting training...')
# train
gbm = lgb.train(params,
                modeling_data['train'],
                num_boost_round=1000,
                valid_sets=modeling_data['eval'],
                verbose_eval=50,
                early_stopping_rounds=10)

lgb.plot_importance(gbm, importance_type='gain')


# Build model on full training data to make predictions for test set
print('Building model on full training data...')

gbm_full = lgb.train(params,
                     modeling_data['train_full'],
                     num_boost_round=gbm.best_iteration)


# Make predictions on test set and save to .csv
print('Making test set predictions...')

y_test_pred = pd.DataFrame({'process_id': df_test_output.process_id,
                            response: gbm_full.predict(modeling_data['test'])}
                           )

# # Handle negative values
# y_test_pred.loc[y_test_pred[response] < 0, response] = y_test_pred.loc[y_test_pred[response] > 0, response].min()
# y_test_pred = y_test_pred.sort_values(by='process_id')
# y_test_pred.to_csv('test_predictions.csv', index=False)
#
# y_test_pred = y_test_pred.sort_values(by=response)
# y_test_pred.to_csv('test_predictions_sorted.csv', index=False)
#
# test_pred_full = y_test_pred.merge(df_test_output, on = 'process_id').sort_values(by='process_id')
# test_pred_full.to_csv('test_predictions_full.csv', index=False)



#  sed -i '/,932,\|process_id/!d' ./train_values_subset.csv

# subset = joined_data[joined_data.object_id == 932].sort_values(by = ['final_rinse_total_turbidity_liter'])
# subset = joined_data.sort_values(by = ['duration'])
#
# temp = gbm_full.predict(X_test)
# temp = joined_data.sort_values(by = ['final_rinse_total_turbidity_liter'])
