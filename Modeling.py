import lightgbm as lgb


def build_lgbm_validation_datasets(train_data, val_data, response, cols_to_drop=None, cols_to_include=None):
    # Model training
    drop_cols = [response]

    y_train = train_data.ix[:, response]
    y_val = val_data.ix[:, response]

    if cols_to_drop is None and cols_to_include is None:
        print('You dun goofed')

    elif cols_to_include is None:
        x_train = train_data.drop(drop_cols + cols_to_drop, axis=1)
        x_val = val_data.drop(drop_cols + cols_to_drop, axis=1)

    elif cols_to_drop is None:
        x_train = train_data[cols_to_include]
        x_val = val_data[cols_to_include]

    else:
        print('How did you even get here?')

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_val, y_val, reference=lgb_train)

    return {'train': lgb_train,
            'eval': lgb_eval
            }


def build_lgbm_test_datasets(full_train_data, test_data, response, cols_to_drop=None, cols_to_include=None):
    # Model training
    drop_cols = [response]

    y_train = full_train_data.ix[:, response]

    if cols_to_drop is None and cols_to_include is None:
        print('You dun goofed')

    elif cols_to_include is None:
        x_train = full_train_data.drop(drop_cols + cols_to_drop, axis=1)
        x_test = test_data.drop(drop_cols + cols_to_drop, axis=1)

    elif cols_to_drop is None:
        x_train = full_train_data[cols_to_include]
        x_test = test_data[cols_to_include]

    else:
        print('How did you even get here?')

    # create dataset for lightgbm
    lgb_full_train = lgb.Dataset(x_train, y_train)

    return {'full_train': lgb_full_train,
            'test': x_test
            }