import pandas as pd
import numpy as np

threshold = 290000
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

def calculate_start_times(df):
    output = pd.DataFrame(df.groupby(['process_id']).timestamp.min()).reset_index()
    output.timestamp = output.timestamp.astype('datetime64[ns]')

    return output


def engineer_features(df):
    for col in ['timestamp']:
        df[col] = df[col].astype('datetime64[ns]')
    for col in ['process_id', 'object_id']:
        df[col] = df[col].astype('category')

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
                              'cov_return_turbidity': df_groupby.return_turbidity.std() / df_groupby.return_turbidity.mean()
                              }).reset_index()

    col_list = list(set(df_output) - set(group_cols))

    for colname in col_list:
        df_output = engineer_features_phases(df_output, colname, lambda df, x, y: np.where(df.phase == x, df[y], 0))

    df_groupby = df_output.groupby(['process_id', 'pipeline', 'object_id'])

    df_output_2 = engineer_features_process(df_groupby, col_list)

    df_final_output = df_output_2.copy()

    return df_final_output

