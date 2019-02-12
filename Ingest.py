import pandas as pd
import numpy as np
import re
pd.options.mode.chained_assignment = None


def ingest_data(path):
    # Read in data from source files
    print('Reading in source data sets...')
    raw_data = pd.read_pickle(path + 'train_values.pkl')
    labels = pd.read_csv(path + 'train_labels.csv')
    metadata = pd.read_csv(path + 'recipe_metadata.csv')
    test_data = pd.read_pickle(path + 'test_values.pkl')
    print('Successfully read in source data sets.')
    print('')

    # Determine when each process started for both train and test data
    # Necessary to properly do walk forward validation and for feature engineering
    print('Determining process start times...')
    train_start_times = calculate_start_times(raw_data)
    test_start_times = calculate_start_times(test_data)
    start_times = pd.concat([train_start_times, test_start_times]).sort_values(by='start_time')
    print('Process start times successfully determined.')
    print('')

    return raw_data, labels, metadata, test_data, start_times


def preprocess_data(df, start_times, return_phase_defs=None, supply_phase_defs=None):
    # Pre-processing - convert "intermediate rinse" to 'int_rinse'
    df.phase[df.phase == 'intermediate_rinse'] = 'int_rinse'

    # # Optional pre-processing - remove processes with objects that aren't in test set
    # df = df[df.object_id.isin(test_data.object_id)]

    print('Calculating process-timestamp-level features...')
    df.timestamp = df.timestamp.astype('datetime64[s]')
    df = df.merge(start_times, on='process_id')

    # Return phase definition
    df['return_phase'] = df.phase + '_' + np.where(df.return_drain == True, 'drain',
                                          np.where(df.return_caustic == True, 'caus',
                                          np.where(df.return_acid == True, 'ac',
                                          np.where(df.return_recovery_water == True, 'rec_water', 'none'))))

    if return_phase_defs is None:
        return_phases = list(df.return_phase.value_counts()[df.return_phase.value_counts() > 300000].reset_index()['index'])
    else:
        return_phases = return_phase_defs
    df['return_phase'] = np.where(df.return_phase.isin(return_phases), df.return_phase, 'other')

    # Supply phase definition
    df['supply_phase'] = df.phase + '_' + np.where(df.supply_pre_rinse == True, 'pre_rin',
                                                   np.where(df.supply_caustic == True, 'caus',
                                                            np.where(df.supply_acid == True, 'ac',
                                                                     np.where(df.supply_clean_water == True,
                                                                              'clean_water', 'none'))))

    if supply_phase_defs is None:
        supply_phases = list(
            df.supply_phase.value_counts()[df.supply_phase.value_counts() > 100000].reset_index()['index'])
    else:
        supply_phases = supply_phase_defs
    df['supply_phase'] = np.where(df.supply_phase.isin(supply_phases), df.supply_phase, 'other')


    # Other process-timestamp-level features
    df['return_flow'] = np.maximum(0, df.return_flow)
    df['supply_flow'] = np.maximum(0, df.supply_flow)

    df['total_flow'] = df.return_flow * df.return_turbidity

    df['phase_elapse_end'] = (
            df.groupby(['process_id', 'phase']).timestamp.transform('max') - df.timestamp).dt.seconds
    df['end_turb'] = df.return_turbidity * (df.phase_elapse_end <= 40)
    df['end_flow'] = df.total_flow * (df.phase_elapse_end <= 40)

    print('Successfully calculated process-timestamp-level features.')
    print('')

    if return_phase_defs is None:
        return df, return_phases, supply_phases
    else:
        return df


def calculate_start_times(df):
    output = pd.DataFrame(df.groupby(['process_id']).timestamp.min()).reset_index()
    output.timestamp = output.timestamp.astype('datetime64[ns]')
    output['day_number'] = output.timestamp.dt.dayofyear - 51
    output.columns = ['process_id', 'start_time', 'day_number']

    return output

