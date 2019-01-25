import lightgbm as lgb


def build_lgbm_validation_datasets(train_data, val_data, response, cols_to_drop=None, cols_to_include=None):
    # Model training
    drop_cols = [response]
    val_data_acid = val_data[val_data.phase_duration_acid.notnull()]
    val_data_int_rinse = val_data[val_data.phase_duration_intermediate_rinse.notnull()]
    val_data_caustic = val_data[val_data.phase_duration_caustic.notnull()]
    val_data_pre_rinse = val_data[val_data.phase_duration_pre_rinse.notnull()]

    y_train = train_data.ix[:, response]
    y_val_acid = val_data_acid.ix[:, response]
    y_val_pre_rinse = val_data_pre_rinse.ix[:, response]
    y_val_caustic = val_data_caustic.ix[:, response]
    y_val_int_rinse = val_data_int_rinse.ix[:, response]

    if cols_to_drop is None and cols_to_include is None:
        print('You dun goofed')

    elif cols_to_include is None:
        x_train = train_data.drop(drop_cols + cols_to_drop, axis=1)
        x_val_acid = val_data_acid.drop(drop_cols + cols_to_drop, axis=1)
        x_val_pre_rinse = val_data_pre_rinse.drop(drop_cols + cols_to_drop, axis=1)
        x_val_caustic = val_data_caustic.drop(drop_cols + cols_to_drop, axis=1)
        x_val_int_rinse = val_data_int_rinse.drop(drop_cols + cols_to_drop, axis=1)

    elif cols_to_drop is None:
        x_train = train_data[cols_to_include]
        x_val_acid = val_data_acid[cols_to_include]
        x_val_pre_rinse = val_data_pre_rinse[cols_to_include]
        x_val_caustic = val_data_caustic[cols_to_include]
        x_val_int_rinse = val_data_int_rinse[cols_to_include]
    else:
        print('How did you even get here?')

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval_acid = lgb.Dataset(x_val_acid, y_val_acid, reference=lgb_train)
    lgb_eval_pre_rinse = lgb.Dataset(x_val_pre_rinse, y_val_pre_rinse, reference=lgb_train)
    lgb_eval_caustic = lgb.Dataset(x_val_caustic, y_val_caustic, reference=lgb_train)
    lgb_eval_int_rinse = lgb.Dataset(x_val_int_rinse, y_val_int_rinse, reference=lgb_train)

    return {'train': lgb_train,
            'eval_acid': lgb_eval_acid,
            'eval_pre_rinse': lgb_eval_pre_rinse,
            'eval_caustic': lgb_eval_caustic,
            'eval_int_rinse': lgb_eval_int_rinse
            }


def build_lgbm_test_datasets(full_train_data, test_data, response, cols_to_drop=None, cols_to_include=None):
    # Model training
    drop_cols = [response]
    test_data_acid = test_data[test_data.phase_duration_acid.notnull()]
    test_data_int_rinse = test_data[test_data.phase_duration_acid.isnull() &
                                    test_data.phase_duration_intermediate_rinse.notnull()]
    test_data_caustic = test_data[test_data.phase_duration_acid.isnull() &
                                  test_data.phase_duration_intermediate_rinse.isnull() &
                                  test_data.phase_duration_caustic.notnull()]
    test_data_pre_rinse = test_data[test_data.phase_duration_acid.isnull() &
                                    test_data.phase_duration_intermediate_rinse.isnull() &
                                    test_data.phase_duration_caustic.isnull()]

    y_train = full_train_data.ix[:, response]

    if cols_to_drop is None and cols_to_include is None:
        print('You dun goofed')

    elif cols_to_include is None:
        x_train = full_train_data.drop(drop_cols + cols_to_drop, axis=1)
        x_test_acid = test_data_acid.drop(drop_cols + cols_to_drop, axis=1)
        x_test_pre_rinse = test_data_pre_rinse.drop(drop_cols + cols_to_drop, axis=1)
        x_test_caustic = test_data_caustic.drop(drop_cols + cols_to_drop, axis=1)
        x_test_int_rinse = test_data_int_rinse.drop(drop_cols + cols_to_drop, axis=1)

    elif cols_to_drop is None:
        x_test_acid = test_data_acid[cols_to_include]
        x_test_pre_rinse = test_data_pre_rinse[cols_to_include]
        x_test_caustic = test_data_caustic[cols_to_include]
        x_test_int_rinse = test_data_int_rinse[cols_to_include]

        if 'process_id' in cols_to_include:
            cols_to_include.remove('process_id')
        x_train = full_train_data[cols_to_include]
    else:
        print('How did you even get here?')

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(x_train, y_train)

    return {'full_train': lgb_train,
            'test_acid': x_test_acid,
            'test_pre_rinse': x_test_pre_rinse,
            'test_caustic': x_test_caustic,
            'test_int_rinse': x_test_int_rinse
            }
