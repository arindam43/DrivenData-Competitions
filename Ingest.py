import pandas as pd
import os
import numpy as np
pd.options.mode.chained_assignment = None


def calculate_start_times(df):
    output = pd.DataFrame(df.groupby(['process_id']).timestamp.min()).reset_index()
    output.timestamp = output.timestamp.astype('datetime64[ns]')
    output['day_number'] = output.timestamp.dt.dayofyear - 51
    output.columns = ['process_id', 'start_time', 'day_number']

    return output


def ingest_data(path):
    print('Reading in source data sets...')
    raw_data = pd.read_pickle(path + 'train_values.pkl')
    labels = pd.read_csv(path + 'train_labels.csv')
    metadata = pd.read_csv(path + 'recipe_metadata.csv')
    test_data = pd.read_pickle(path + 'test_values.pkl')
    print('Successfully read in source data sets.')
    print('')

    # Determine start times for train and test data
    # Necessary to properly do walk forward validation
    print('Determining process start times...')
    train_start_times = calculate_start_times(raw_data)
    test_start_times = calculate_start_times(test_data)
    start_times = pd.concat([train_start_times, test_start_times]).sort_values(by='start_time')
    print('Process start times successfully determined.')
    print('')

    # Change column names for metadata to work with rest of script
    metadata.columns = ['process_id', 'recipe_caus', 'recipe_int', 'recipe_ac']
    return raw_data, labels, metadata, test_data, start_times


def preprocess_data(df, start_times):
    # Pre-processing - convert "intermediate rinse" to 'int_rinse'
    df.phase[df.phase == 'intermediate_rinse'] = 'int_rinse'

    # Preprocessing - remove processes with objects that aren't in test set
    # raw_data = raw_data[raw_data.object_id.isin(test_data.object_id)]

    print('Calculating process-timestamp-level features...')
    df.timestamp = df.timestamp.astype('datetime64[s]')
    df = df.merge(start_times, on='process_id')
    # Row level features
    df['return_flow'] = np.maximum(0, df.return_flow)
    df['total_turbidity'] = df.return_flow * df.return_turbidity
    df['phase_elapse_end'] = (
            df.groupby(['process_id', 'phase']).timestamp.transform('max') - df.timestamp).dt.seconds

    for phase in['caustic', 'acid', 'recovery_water', 'drain']:
        df[phase + '_flow'] = df['return_' + phase] * df.return_flow * df.return_turbidity
        df[phase + '_temp'] = df['return_' + phase] * df.return_temperature
        df[phase + '_flow_end'] = df['return_' + phase] * df.return_flow * df.return_turbidity * (
            df.phase_elapse_end <= 40)

    print('Successfully calculated process-timestamp-level features.')
    print('')
    return df
