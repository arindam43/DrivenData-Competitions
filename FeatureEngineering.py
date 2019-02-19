import pandas as pd
import numpy as np
import re


def create_model_datasets(df_train, df_test, start_times, labels, response, metadata, path, val_or_test='validation'):

    # Create normalization lookup tables
    print('Creating and merging normalization lookup tables...')

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

    print('Normalization lookup tables finished.')

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

    # Bring in labels and metadata for train and validation data
    processed_train_data = processed_train_data.merge(labels, on='process_id').merge(metadata, on='process_id')
    processed_val_data = processed_val_data.merge(metadata, on='process_id')
    if val_or_test == 'validation':
        processed_val_data = processed_val_data.merge(labels, on='process_id')

    # Remove outliers from training data
    processed_train_data = remove_outliers(processed_train_data, response)

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
    df['norm_return_flow'] = df.return_flow / df.median_return_flow
    df['norm_turb'] = df.norm_return_flow * df.return_turbidity

    df['norm_supply_pressure'] = df.supply_pressure - df.median_supply_pressure
    df['norm_conductivity'] = df.return_conductivity - df.median_conductivity

    group_cols = ['process_id', 'object_id', 'pipeline']

    # Calculate features at various levels of aggregation
    df_return_phase = calculate_features(df, group_cols, 'return_phase')
    df_supply_phase = calculate_features(df, group_cols, 'supply_phase', df_return_phase)
    df_full_phase = calculate_features(df, group_cols, 'phase', df_supply_phase)
    df_final_output = calculate_features(df, group_cols, 'process', df_full_phase)

    # Bring in start times for processed data
    df_final_output = df_final_output.merge(timestamps, on='process_id')
    df_final_output = df_final_output.sort_values(by=['object_id', 'start_time'])

    return df_final_output


def calculate_features(df, base_group_cols, level='process', existing_features=None):

    full_group_cols = base_group_cols + [level] if level != 'process' else base_group_cols
    df_groupby = df.groupby(full_group_cols)

    if level == 'return_phase':
        features = pd.DataFrame({'return_turb': df_groupby.norm_turb.sum(),
                                 'return_residue': df_groupby.return_residue.sum(),
                                 'return_cond': df_groupby.norm_conductivity.min(),
                                 'return_duration': (df_groupby.timestamp.max() -
                                                     df_groupby.timestamp.min()).astype('timedelta64[s]')
                                 }).reset_index()
    elif level == 'supply_phase':
        features = pd.DataFrame({'supply_flow': df_groupby.supply_flow.sum(),
                                 'supply_pressure': df_groupby.norm_supply_pressure.min(),
                                 'supply_duration': (df_groupby.timestamp.max() -
                                                     df_groupby.timestamp.min()).astype('timedelta64[s]'),
                                 }).reset_index()
    elif level == 'phase':
        features = pd.DataFrame({'row_count': df_groupby.phase.count(),

                                 'end_turb': df_groupby.end_turb.mean(),
                                 'end_residue': df_groupby.end_residue.sum(),

                                 'return_temp': df_groupby.return_temperature.min(),

                                 'lsh_caus': df_groupby.tank_lsh_caustic.sum() / (df_groupby.timestamp.max() -
                                                                                  df_groupby.timestamp.min()).astype(
                                     'timedelta64[s]'),
                                 'obj_low_lev': df_groupby.object_low_level.sum() / (df_groupby.timestamp.max() -
                                                                                     df_groupby.timestamp.min()).astype(
                                     'timedelta64[s]')
                                 }).reset_index()
    else:
        features = pd.DataFrame({'total_duration': (df_groupby.timestamp.max() -
                                                    df_groupby.timestamp.min()).astype('timedelta64[s]')
                                 }).reset_index()

    if level != 'process':
        features = pd.pivot_table(features,
                                  index=base_group_cols,
                                  columns=level,
                                  values=list(set(features.columns) - set(full_group_cols))).reset_index()

        features.columns = [' '.join(col).strip() for col in features.columns.values]
        features.columns = features.columns.str.replace(' ', '_')

    output = features if existing_features is None else features.merge(existing_features, on=base_group_cols)

    return output


def remove_outliers(processed_train_data, response):
    # Remove processed train data with unusually short or long train duration
    output = processed_train_data[(processed_train_data.total_duration > 30) &
                                  (processed_train_data.total_duration < 10000)]

    print('Number of outliers removed: ' + str(processed_train_data.shape[0] - output.shape[0]))

    # Clipping experiments
    quantiles = (output.groupby('object_id')[response].quantile(0.2) / 10).reset_index()
    quantiles.columns = ['object_id', 'response_thresh']
    output = output.merge(quantiles, on='object_id')

    print('Number of outliers clipped: ' + str(output[output.response_thresh > output[response]].shape[0]))
    # output = output[output.response_thresh < output[response]]
    output[response] = np.where(output.response_thresh > output[response],
                                output.response_thresh - (output.response_thresh - output[response])/5,
                                output[response])

    return output
