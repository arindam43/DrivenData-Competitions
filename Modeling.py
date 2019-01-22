import lightgbm as lgb


def build_lgbm_datasets(train_data, test_data, train_ratio, response, cols_to_drop=None, cols_to_include=None):
    # Model training
    num_rows = train_data.shape[0]
    drop_cols = [response]

    train_size = int(round(train_ratio * num_rows))
    df_train = train_data.iloc[0:train_size]
    df_val = train_data.iloc[train_size:num_rows]

    y_train = df_train.ix[:, response]
    y_val = df_val.ix[:, response]

    if cols_to_drop is None and cols_to_include is None:
        print('You dun goofed')

    elif cols_to_include is None:
        x_train = df_train.drop(drop_cols + cols_to_drop, axis=1)
        x_val = df_val.drop(drop_cols + cols_to_drop, axis=1)
        x_test = test_data.copy().drop(cols_to_drop, axis=1)

        x_full = train_data.copy().drop(drop_cols + cols_to_drop, axis=1)
        y_full = train_data.copy().ix[:, response]

    elif cols_to_drop is None:
        x_train = df_train[cols_to_include]
        x_val = df_val[cols_to_include]
        x_test = test_data.copy()[cols_to_include]

        x_full = train_data.copy()[cols_to_include]
        y_full = train_data.copy().ix[:, response]

    else:
        print('How did you even get here?')


    # create dataset for lightgbm
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_val, y_val, reference=lgb_train)
    lgb_train_full = lgb.Dataset(x_full, y_full)

    return {'train': lgb_train,
            'eval': lgb_eval,
            'train_full': lgb_train_full,
            'test': x_test}

