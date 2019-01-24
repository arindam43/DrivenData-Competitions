import pandas as pd
import numpy as np


def calculate_start_times(df):
    output = pd.DataFrame(df.groupby(['process_id']).timestamp.min()).reset_index()
    output.timestamp = output.timestamp.astype('datetime64[ns]')

    return output


def remove_phases(validation_data):
    output = validation_data[((validation_data.type == 'pre_rinse') &
                                (validation_data.phase == 'pre_rinse')) |
                             ((validation_data.type == 'caustic') &
                                (validation_data.phase.isin(['pre_rinse', 'caustic']))) |
                             ((validation_data.type == 'int_rinse') &
                                (validation_data.phase.isin(['pre_rinse', 'caustic', 'intermediate_rinse']))) |
                             (validation_data.type == 'acid')]

    return output


def engineer_features_phases(df, colname, func):
    phases = ['pre_rinse', 'caustic', 'intermediate_rinse', 'acid']

    for x in phases:
        col = colname + '_' + x
        df[col] = func(df, x, colname)

    return df


def engineer_features_process(df_groupby, cols):
    phases = ['', 'pre_rinse', 'caustic', 'intermediate_rinse', 'acid']

    output_dict = {}

    for x in cols:
        for y in phases:
            col = x + ('' if y == '' else '_') + y
            output_dict[col] = df_groupby[col].sum().astype(float)

    output_df = pd.DataFrame(output_dict).reset_index()

    return output_df


def engineer_features(df, timestamps):
    for col in ['timestamp']:
        df[col] = df[col].astype('datetime64[ns]')

    df = df[df.phase != 'final_rinse']

    df['total_turbidity'] = df.return_turbidity * df.return_flow
    df['lagged_turbidity'] = df.groupby('process_id').return_turbidity.shift(1)
    df['delta_turbidity'] = df.lagged_turbidity - df.return_turbidity
    #df['start_time'] = df.groupby(['process_id', 'phase']).timestamp.transform('min')
    #df['duration'] = (df.timestamp - df.start_time).astype('timedelta64[s]')

    group_cols = ['process_id', 'object_id', 'pipeline', 'phase']
    df_groupby = df.groupby(group_cols)

    df_output = pd.DataFrame({'phase_duration': (df_groupby.timestamp.max() -
                                                 df_groupby.timestamp.min()).astype('timedelta64[s]'),
                              'total_turbidity': df_groupby.total_turbidity.sum(),
                              'total_turbidity_rate': df_groupby.total_turbidity.sum() / (df_groupby.timestamp.max() -
                                                      df_groupby.timestamp.min()).astype('timedelta64[s]'),
                              'cov_return_turbidity': df_groupby.return_turbidity.std() / df_groupby.return_turbidity.mean(),
                              'max_turbidity': df_groupby.total_turbidity.quantile(0.95),
                              'min_turbidity': df_groupby.total_turbidity.quantile(0.1)
                              }).reset_index()

    col_list = list(set(df_output) - set(group_cols))

    for colname in col_list:
        df_output = engineer_features_phases(df_output, colname, lambda df, x, y: np.where(df.phase == x, df[y], 0))

    df_groupby = df_output.groupby(['process_id', 'pipeline', 'object_id'])

    df_output_2 = engineer_features_process(df_groupby, col_list)

    df_final_output = df_output_2.copy()

    df_final_output = df_final_output.replace(0, np.nan)

    df_final_output = df_final_output.merge(timestamps, on='process_id')
    df_final_output['day_of_week'] = df_final_output.timestamp.dt.date
    df_final_output = df_final_output.sort_values(by=['pipeline','timestamp'])
    df_final_output['cumulative_runs_day'] = df_final_output.groupby(['pipeline', 'day_of_week']).\
                                                             cumcount()
    df_final_output['previous_object'] = df_final_output.groupby(['pipeline', 'day_of_week'])['object_id'].shift(1)
    df_final_output['previous_run_start_time'] = df_final_output.groupby(['pipeline', 'day_of_week']).timestamp.shift(1)
    df_final_output['previous_run_delta'] = (df_final_output.timestamp - df_final_output.previous_run_start_time).astype('timedelta64[s]')

    return df_final_output


#def convert_categorial_features(train_data, test_data, cat_cols):