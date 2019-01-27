import pandas as pd
import numpy as np
import re


def calculate_start_times(df):
    output = pd.DataFrame(df.groupby(['process_id']).timestamp.min()).reset_index()
    output.timestamp = output.timestamp.astype('datetime64[ns]')

    return output


def calculate_features(df_groupby):
    return pd.DataFrame({'phase_duration': (df_groupby.timestamp.max() -
                                            df_groupby.timestamp.min()).astype('timedelta64[s]'),
                         'total_turbidity': df_groupby.total_turbidity.sum(),
                         'total_return_flow': df_groupby.return_flow.sum(),
                         'prop_object_low_level': df_groupby.object_low_level.sum() / ((df_groupby.timestamp.max() -
                                            df_groupby.timestamp.min()).astype('timedelta64[s]')),
                         'prop_lsh_caustic': df_groupby.tank_lsh_caustic.sum() / ((df_groupby.timestamp.max() -
                                            df_groupby.timestamp.min()).astype('timedelta64[s]')),
                         'prop_return_drain': df_groupby.return_drain.sum() / ((df_groupby.timestamp.max() -
                                            df_groupby.timestamp.min()).astype('timedelta64[s]')),
                        }).reset_index()


def engineer_features(df, timestamps):
    for col in ['timestamp']:
        df[col] = df[col].astype('datetime64[ns]')

    # Row level features
    df['total_turbidity'] = df.return_turbidity * df.return_flow

    # Phase-level features
    group_cols = ['process_id', 'object_id', 'pipeline', 'phase']
    df_groupby = df.groupby(group_cols)

    df_output_phase = calculate_features(df_groupby)

    df_output_phase = pd.pivot_table(df_output_phase,
                                     index=['process_id', 'object_id', 'pipeline'],
                                     columns='phase',
                                     values=list(set(df_output_phase.columns) - set(group_cols))).reset_index()

    df_output_phase.columns = [' '.join(col).strip() for col in df_output_phase.columns.values]
    df_output_phase.columns = df_output_phase.columns.str.replace(' ', '_')

    # Process-level aggregations of phase-level features
    group_cols = ['process_id', 'object_id', 'pipeline']
    df_groupby = df.groupby(group_cols)

    df_output_process = calculate_features(df_groupby)

    df_final_output = df_output_phase.merge(df_output_process, on=group_cols)

    # Other process-level features

    df_final_output = df_final_output.merge(timestamps, on='process_id')
    df_final_output = df_final_output.sort_values(by=['object_id', 'timestamp'])

    # df_final_output['hour_of_day'] = df_final_output.timestamp.dt.hour * 60 + df_final_output.timestamp.dt.minute

    # df_final_output['weekday_name'] = df_final_output.timestamp.dt.dayofweek
    #
    # df_final_output['cumulative_runs_day'] = df_final_output.groupby(['pipeline', 'day_of_week']).\
    #                                                          cumcount()
    #
    # df_final_output['starting_phase'] = np.where(pd.notnull(df_final_output.phase_duration_pre_rinse), "pre_rinse",
    #                                              np.where(pd.notnull(df_final_output.phase_duration_caustic), "caustic",
    #                                                       np.where(pd.notnull(df_final_output.phase_duration_intermediate_rinse), "int_rinse",
    #                                                                np.where(pd.notnull(df_final_output.phase_duration_acid), 'acid', 'other'))))
    #
    #
    # cols_to_shift = list(filter(lambda x: re.search(r'(?=.*pre_rinse|.*caustic|.*intermediate|.*acid)', x), list(df_final_output.columns)))
    #
    # for col in cols_to_shift:
    #     df_final_output['previous_' + col] = df_final_output.groupby(['pipeline', 'object_id'])[col].shift(1)

    return df_final_output


def remove_outliers(processed_train_data):
    # Remove processed with too short or long of train duration
    processed_train_data = processed_train_data[(processed_train_data.phase_duration > 20) & (processed_train_data.phase_duration < 8000)]

    return processed_train_data

#
# sort_col = 'total_return_flow'
# processed_test_data = processed_test_data.sort_values(by=sort_col)
# processed_train_data = processed_train_data.sort_values(by=sort_col)
