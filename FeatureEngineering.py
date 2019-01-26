import pandas as pd
import numpy as np
import re

def calculate_start_times(df):
    output = pd.DataFrame(df.groupby(['process_id']).timestamp.min()).reset_index()
    output.timestamp = output.timestamp.astype('datetime64[ns]')

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

    group_cols = ['process_id', 'object_id', 'pipeline', 'phase']
    df_groupby = df.groupby(group_cols)

    df_output = pd.DataFrame({'phase_duration': (df_groupby.timestamp.max() -
                                                 df_groupby.timestamp.min()).astype('timedelta64[s]'),
                              'total_turbidity': df_groupby.total_turbidity.sum(),
                              'max_supply_flow': df_groupby.supply_flow.quantile(0.9),
                              'min_supply_flow': df_groupby.supply_flow.quantile(0.2)
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
    df_final_output['weekday_name'] = df_final_output.timestamp.dt.dayofweek

    df_final_output = df_final_output.sort_values(by=['pipeline', 'timestamp'])
    df_final_output['cumulative_runs_day'] = df_final_output.groupby(['pipeline', 'day_of_week']).\
                                                             cumcount()
    #
    # cols_to_shift = list(filter(lambda x: re.search(r'(?=.*pre_rinse|.*caustic|.*intermediate|.*acid)', x), list(df_final_output.columns)))
    #
    # df_final_output['previous_object_id'] = df_final_output.groupby('pipeline')['object_id'].shift(1).fillna(-10000).astype(int)
    #
    # for col in cols_to_shift:
    #     df_final_output['previous_' + col] = df_final_output.groupby('pipeline')[col].shift(1)
    #

    return df_final_output

