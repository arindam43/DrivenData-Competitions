import pandas as pd
import numpy as np
import re


def create_model_datasets(df_train, df_test, start_times, labels, metadata, path, val_or_test='validation'):

    # Create normalization lookup tables
    train_lookup = df_train.groupby(['object_id', 'return_phase']).\
        agg({'return_flow': 'median', 'return_conductivity': 'median'}).reset_index()
    train_lookup.columns = ['object_id', 'return_phase', 'median_return_flow', 'median_conductivity']
    df_train = df_train.merge(train_lookup, on=['object_id', 'return_phase'])
    df_test = df_test.merge(train_lookup, on=['object_id', 'return_phase'])

    train_lookup = df_train.groupby(['object_id', 'supply_phase']).\
        agg({'supply_flow': 'median', 'supply_pressure': 'median'}).reset_index()
    train_lookup.columns = ['object_id', 'supply_phase', 'median_supply_flow', 'median_supply_pressure']
    df_train = df_train.merge(train_lookup, on=['object_id', 'supply_phase'], how='left').sort_values(by='timestamp')
    df_test = df_test.merge(train_lookup, on=['object_id', 'supply_phase'], how='left').sort_values(by='timestamp')

    # Engineer phase-level features on train, validation, and test sets
    print('Engineering features on train, ' + val_or_test + ' sets...')
    processed_train_data = engineer_features(df_train, start_times)
    processed_val_data = engineer_features(df_test, start_times)
    print('Successfully engineered features.')

    # Fill nas with 0, where appropriate
    for model_type in ['acid', 'int_rinse', 'pre_rinse', 'caustic']:
        cols = list(filter(lambda x: re.search(r'(?=.*' + model_type + ')', x), list(processed_train_data.columns)))
        processed_train_data.loc[pd.notna(processed_train_data['row_count_' + model_type]), cols] =\
            processed_train_data.loc[pd.notna(processed_train_data['row_count_' + model_type]), cols].fillna(0)
        # processed_train_data.loc[:, cols] = processed_train_data.loc[:, cols].fillna(-1)

    # Drop features that make no sense (produce mostly 0 or nan)
    keep_cols = processed_val_data.apply(lambda x: (x.isnull()).sum() / len(x)) <= 0.9
    processed_train_data = processed_train_data[list(keep_cols[keep_cols].index)]
    processed_val_data = processed_val_data[list(keep_cols[keep_cols].index)]

    # Remove outliers from training data
    processed_train_data = remove_outliers(processed_train_data)

    # Bring in labels and metadata for train and validation data
    processed_train_data = processed_train_data.merge(labels, on='process_id').merge(metadata, on='process_id')
    processed_val_data = processed_val_data.merge(metadata, on='process_id')
    if val_or_test == 'validation':
        processed_val_data = processed_val_data.merge(labels, on='process_id')

    # Write datasets out to local
    if val_or_test == 'validation':
        processed_train_data.to_csv(path + 'modeling_data.csv')
    elif val_or_test == 'test':
        processed_train_data.to_csv(path + 'full_modeling_data.csv')
        processed_val_data.to_csv(path + 'processed_test_data.csv')

    # Convert object id to category
    # Ensure that categories are consistent across training, validation, and test sets
    for col in ['object_id', 'recipe_type']:
        processed_train_data[col] = processed_train_data[col].astype('category')
        processed_val_data[col] = processed_val_data[col].astype('category', categories=processed_train_data[
            col].cat.categories)

    processed_val_data = processed_val_data.sort_values(by='process_id')

    return processed_train_data, processed_val_data


def engineer_features(df, timestamps):

    # Normalize flows using historical averages
    df['norm_supply_flow'] = df.supply_flow / df.median_supply_flow
    df['norm_return_flow'] = df.return_flow / df.median_return_flow
    df['norm_turb'] = df.norm_return_flow * df.return_turbidity

    df['norm_supply_pressure'] = df.supply_pressure - df.median_supply_pressure
    df['norm_conductivity'] = df.return_conductivity - df.median_conductivity

    # Return-phase-level features
    group_cols = ['process_id', 'object_id', 'pipeline', 'return_phase']
    df_groupby = df.groupby(group_cols)

    df_output_phase = calculate_features(df_groupby, level='return_phase')

    df_output_phase = pd.pivot_table(df_output_phase,
                                     index=['process_id', 'object_id', 'pipeline'],
                                     columns='return_phase',
                                     values=list(set(df_output_phase.columns) - set(group_cols))).reset_index()

    df_output_phase.columns = [' '.join(col).strip() for col in df_output_phase.columns.values]
    df_output_phase.columns = df_output_phase.columns.str.replace(' ', '_')

    # Supply-phase-level features
    group_cols = ['process_id', 'object_id', 'pipeline', 'supply_phase']
    df_groupby = df.groupby(group_cols)

    df_output_phase2 = calculate_features(df_groupby, level='supply_phase')

    df_output_phase2 = pd.pivot_table(df_output_phase2,
                                      index=['process_id', 'object_id', 'pipeline'],
                                      columns='supply_phase',
                                      values=list(set(df_output_phase2.columns) - set(group_cols))).reset_index()

    df_output_phase2.columns = [' '.join(col).strip() for col in df_output_phase2.columns.values]
    df_output_phase2.columns = df_output_phase2.columns.str.replace(' ', '_')

    df_output_phase = df_output_phase2.merge(df_output_phase, on=['process_id', 'object_id', 'pipeline'])

    # Phase-level features
    group_cols = ['process_id', 'object_id', 'pipeline', 'phase']
    df_groupby = df.groupby(group_cols)
    df_output_phase2 = calculate_features(df_groupby, level='phase')

    df_output_phase2 = pd.pivot_table(df_output_phase2,
                                      index=['process_id', 'object_id', 'pipeline'],
                                      columns='phase',
                                      values=list(set(df_output_phase2.columns) - set(group_cols))).reset_index()

    df_output_phase2.columns = [' '.join(col).strip() for col in df_output_phase2.columns.values]
    df_output_phase2.columns = df_output_phase2.columns.str.replace(' ', '_')

    df_output_phase = df_output_phase2.merge(df_output_phase, on=['process_id', 'object_id', 'pipeline'])

    # Process-level aggregations of phase-level features
    group_cols = ['process_id', 'object_id', 'pipeline']
    df_groupby = df.groupby(group_cols)

    df_output_process = calculate_features(df_groupby, level='process')

    df_final_output = df_output_phase.merge(df_output_process, on=group_cols)

    # Other process-level features
    df_final_output = df_final_output.merge(timestamps, on='process_id')
    df_final_output = df_final_output.sort_values(by=['object_id', 'start_time'])
    #
    # df_final_output['hour_of_day'] = df_final_output.start_time.dt.hour

    # df_final_output['weekday_name'] = df_final_output.start_time.dt.dayofweek

    # df_final_output['cumulative_runs_day'] = df_final_output.groupby(['pipeline', 'day_of_week']).\
    #                                                          cumcount()

    return df_final_output


def calculate_features(df_groupby, level):
    if level == 'return_phase':
        output = pd.DataFrame({'return_turb': df_groupby.norm_turb.sum(),
                               #'return_turb_max': df_groupby.rolling_turb.max(),
                               'return_residue': df_groupby.return_residue.sum(),
                               'return_cond': df_groupby.norm_conductivity.min(),
                               'return_duration': (df_groupby.timestamp.max() -
                                                   df_groupby.timestamp.min()).astype('timedelta64[s]'),
                               }).reset_index()
    elif level == 'supply_phase':
        output = pd.DataFrame({'supply_flow': df_groupby.supply_flow.sum(),
                               'supply_pressure': df_groupby.norm_supply_pressure.min(),
                               'supply_duration': (df_groupby.timestamp.max() -
                                                   df_groupby.timestamp.min()).astype('timedelta64[s]'),
                               }).reset_index()
    elif level == 'phase':
        output = pd.DataFrame({'row_count': df_groupby.phase.count(),

                               'end_turb': df_groupby.end_turb.mean(),
                               'end_residue': df_groupby.end_residue.sum(),

                               'return_temp': df_groupby.return_temperature.min(),
                               'obj_low_lev': df_groupby.object_low_level.sum() / (df_groupby.timestamp.max() -
                                                                                     df_groupby.timestamp.min()).astype(
                                   'timedelta64[s]'),
                               'lsh_caus': df_groupby.tank_lsh_caustic.sum() / (df_groupby.timestamp.max() -
                                                                                df_groupby.timestamp.min()).astype(
                                   'timedelta64[s]')
                               }).reset_index()
    else:
        output = pd.DataFrame({'phase_duration': (df_groupby.timestamp.max() -
                                                  df_groupby.timestamp.min()).astype('timedelta64[s]')
                               }).reset_index()

    return output


def remove_outliers(processed_train_data):
    # Remove processed with too short or long of train duration
    processed_train_data = processed_train_data[(processed_train_data.phase_duration > 30) &
                                                (processed_train_data.phase_duration < 10000)]

    return processed_train_data

