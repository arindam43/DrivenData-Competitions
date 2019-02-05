import pandas as pd


def create_model_datasets(df_train, df_test, start_times, labels, metadata, path, val_or_test='validation'):

    # Engineer phase-level features on train, validation, and test sets
    print('Engineering features on train, ' + val_or_test + ' sets...')
    processed_train_data = engineer_features(df_train, start_times)
    processed_val_data = engineer_features(df_test, start_times)
    print('Successfully engineered features.')

    # Drop features that make no sense (produce mostly 0 or nan)
    keep_cols = processed_train_data.apply(lambda x: (x.isnull()).sum() / len(x)) <= 0.8
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
    # Return-phase-level features
    group_cols = ['process_id', 'object_id', 'pipeline', 'return_phase']
    df_groupby = df.groupby(group_cols)

    df_output_phase = calculate_features(df, df_groupby, group_cols, level='return_phase')

    df_output_phase = pd.pivot_table(df_output_phase,
                                     index=['process_id', 'object_id', 'pipeline'],
                                     columns='return_phase',
                                     values=list(set(df_output_phase.columns) - set(group_cols))).reset_index()

    df_output_phase.columns = [' '.join(col).strip() for col in df_output_phase.columns.values]
    df_output_phase.columns = df_output_phase.columns.str.replace(' ', '_')

    # Phase-level features
    group_cols = ['process_id', 'object_id', 'pipeline', 'phase']
    df_groupby = df.groupby(group_cols)
    df_output_phase2 = calculate_features(df, df_groupby, group_cols, level='phase')

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

    df_output_process = calculate_features(df, df_groupby, group_cols, level='process')

    df_final_output = df_output_phase.merge(df_output_process, on=group_cols)

    # Other process-level features
    df_final_output = df_final_output.merge(timestamps, on='process_id')
    df_final_output = df_final_output.sort_values(by=['object_id', 'start_time'])

    # df_final_output['hour_of_day'] = df_final_output.timestamp.dt.hour * 60 + df_final_output.timestamp.dt.minute

    # df_final_output['weekday_name'] = df_final_output.timestamp.dt.dayofweek
    #
    # df_final_output['cumulative_runs_day'] = df_final_output.groupby(['pipeline', 'day_of_week']).\
    #                                                          cumcount()

    # cols_to_shift = list(filter(lambda x: re.search(r'(?=.*pre_rinse|.*caustic|.*intermediate|.*acid)', x), list(df_final_output.columns)))
    #
    # for col in cols_to_shift:
    #     df_final_output['previous_' + col] = df_final_output.groupby(['pipeline', 'object_id'])[col].shift(1)

    return df_final_output


def calculate_features(df, df_groupby, group_cols, level):
    if level == 'return_phase':
        output = pd.DataFrame({'total_flow': df_groupby.total_turbidity.sum(),
                               }).reset_index()
    elif level == 'phase':
        output = pd.DataFrame({'phase_duration': (df_groupby.timestamp.max() -
                                                  df_groupby.timestamp.min()).astype('timedelta64[s]'),
                               'row_count': df_groupby.phase.count(),
                               'end_flow': df_groupby.end_turbidity.sum(),
                               'min_temp': df_groupby.return_temperature.min(),
                               'obj_low_level': df_groupby.object_low_level.sum() / (df_groupby.timestamp.max() -
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
    processed_train_data = processed_train_data[(processed_train_data.phase_duration > 20) &
                                                (processed_train_data.phase_duration < 10000)]

    return processed_train_data

