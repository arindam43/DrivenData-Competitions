import pandas as pd
import numpy as np
import re


def calculate_start_times(df):
    output = pd.DataFrame(df.groupby(['process_id']).timestamp.min()).reset_index()
    output.timestamp = output.timestamp.astype('datetime64[ns]')
    output['day_number'] = output.timestamp.dt.dayofyear - 51
    output.columns = ['process_id', 'start_time', 'day_number']

    return output


def calculate_features(df_groupby):
    return pd.DataFrame({'phase_duration': (df_groupby.timestamp.max() -
                                            df_groupby.timestamp.min()).astype('timedelta64[s]'),

                         'caus_flow': df_groupby.caustic_flow.sum(),
                         'ac_flow': df_groupby.acid_flow.sum(),
                         'rec_water_flow': df_groupby.recovery_water_flow.sum(),
                         'drain_flow': df_groupby.drain_flow.sum(),

                         'caus_flow_end': df_groupby.caustic_flow_end.sum(),
                         'ac_flow_end': df_groupby.acid_flow_end.sum(),
                         'rec_water_flow_end': df_groupby.recovery_water_flow_end.sum(),
                         'drain_flow_end': df_groupby.drain_flow_end.sum(),

                         'caus_temp': df_groupby.caustic_temp.min(),
                         'ac_temp': df_groupby.acid_temp.min(),
                         'rec_water_temp': df_groupby.recovery_water_temp.min(),
                         'drain_temp': df_groupby.drain_temp.min(),

                         'obj_low_level': df_groupby.object_low_level.sum() / (df_groupby.timestamp.max() -
                                                                               df_groupby.timestamp.min()).astype('timedelta64[s]'),
                         'lsh_caus': df_groupby.tank_lsh_caustic.sum() / (df_groupby.timestamp.max() -
                                                                          df_groupby.timestamp.min()).astype('timedelta64[s]')
                        }).reset_index()


def engineer_features(df, timestamps):
    df.timestamp = df.timestamp.astype('datetime64[s]')
    df = df.merge(timestamps, on='process_id')

    # Row level features
    df['return_flow'] = np.maximum(0, df.return_flow)
    # df['total_turbidity'] = df.return_flow * df.return_turbidity
    df['caustic_flow'] = df.return_flow * df.return_caustic * df.return_turbidity
    df['acid_flow'] = df.return_flow * df.return_acid * df.return_turbidity
    df['recovery_water_flow'] = df.return_flow * df.return_recovery_water * df.return_turbidity
    df['drain_flow'] = df.return_flow * df.return_drain * df.return_turbidity

    df['caustic_temp'] = df.return_temperature * df.return_caustic
    df['acid_temp'] = df.return_temperature * df.return_acid
    df['recovery_water_temp'] = df.return_temperature * df.return_recovery_water
    df['drain_temp'] = df.return_temperature * df.return_drain

    df['phase_elapse_end'] = (df.groupby(['process_id', 'phase']).timestamp.transform('max') - df.timestamp).dt.seconds
    df['caustic_flow_end'] = df.return_flow * df.return_caustic * df.return_turbidity * (df.phase_elapse_end <= 40)
    df['acid_flow_end'] = df.return_flow * df.return_acid * df.return_turbidity * (df.phase_elapse_end <= 40)
    df['recovery_water_flow_end'] = df.return_flow * df.return_recovery_water * df.return_turbidity * (df.phase_elapse_end <= 40)
    df['drain_flow_end'] = df.return_flow * df.return_drain * df.return_turbidity * (df.phase_elapse_end <= 40)

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

    # Drop columns that don't make sense

    # Process-level aggregations of phase-level features
    group_cols = ['process_id', 'object_id', 'pipeline']
    df_groupby = df.groupby(group_cols)

    df_output_process = calculate_features(df_groupby)

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


def remove_outliers(processed_train_data):
    # Remove processed with too short or long of train duration
    processed_train_data = processed_train_data[(processed_train_data.phase_duration > 20) &
                                                (processed_train_data.phase_duration < 10000)]

    return processed_train_data

#
# sort_col = 'total_return_flow'
# processed_test_data = processed_test_data.sort_values(by=sort_col)
# processed_train_data = processed_train_data.sort_values(by=sort_col)
