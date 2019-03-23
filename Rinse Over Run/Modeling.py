import lightgbm as lgb
import pandas as pd
import re
import shap
import matplotlib


def subset_df_cols(regex, df):
    return set(filter(lambda x: re.search(regex, x), list(df.columns)))


def select_model_columns(processed_train_data, cols_subset=None):
    # For each of the four models, identify which columns should be kept from overall set
    # Simulates data censoring in test data

    pre_rinse_cols = subset_df_cols(r'(?=.*caustic|.*int_rinse|.*acid|.*other|.*residue|.*cond|.*temp)', processed_train_data)
    caustic_cols = subset_df_cols(r'(?=.*turb_pre_rinse|.*int_rinse|.*acid|.*other|.*residue|.*cond|.*temp)', processed_train_data)
    int_rinse_cols = subset_df_cols(r'(?=.*turb_pre_rinse|.*acid|.*other|.*flow|.*residue|recipe.*)', processed_train_data)
    acid_cols = subset_df_cols(r'(?=turb_acid|.*turb_caustic|.*residue_pre_rinse|.*turb_pre_rinse|.*residue_int_rinse|.*turb_int_rinse|.*flow|.*sup|recipe.*)', processed_train_data)

    base_cols = list(subset_df_cols(r'(?=.*row_count|total.*|.*none)', processed_train_data))
    misc_cols = ['response_thresh', 'day_number', 'start_time', 'process_id', 'pipeline']

    exclude_cols = set(misc_cols + base_cols) \
        if cols_subset is None \
        else set(list(subset_df_cols(r'(?=.*' + cols_subset + ')', processed_train_data)) + misc_cols + base_cols)
    all_cols = set(processed_train_data.columns)

    cols_to_include = {
        'pre_rinse': list(all_cols - exclude_cols - pre_rinse_cols),
        'caustic':   list(all_cols - exclude_cols - caustic_cols),
        'int_rinse': list(all_cols - exclude_cols - int_rinse_cols),
        'acid':      list(all_cols - exclude_cols - acid_cols)
    }

    return cols_to_include


def build_lgbm_validation_datasets(train_data, val_data, model_type, response, cols_to_include=None):
    # Model training
    val_data = val_data[val_data['row_count_' + model_type].notnull()]

    y_train = train_data.loc[:, response]
    y_val = val_data.loc[:, response]

    x_train = train_data[cols_to_include]
    x_val = val_data[cols_to_include]

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_val, y_val, reference=lgb_train)

    return {'train': lgb_train,
            'eval': lgb_eval
            }


def build_lgbm_test_datasets(full_train_data, test_data, response, cols_to_include=None):
    # Model training
    cols_to_include = cols_to_include + ['process_id']

    test_data_acid = test_data[test_data.row_count_acid.notnull()]
    test_data_int_rinse = test_data[test_data.row_count_acid.isnull() &
                                    test_data.row_count_int_rinse.notnull()]
    test_data_caustic = test_data[test_data.row_count_acid.isnull() &
                                  test_data.row_count_int_rinse.isnull() &
                                  test_data.row_count_caustic.notnull()]
    test_data_pre_rinse = test_data[test_data.row_count_acid.isnull() &
                                    test_data.row_count_int_rinse.isnull() &
                                    test_data.row_count_caustic.isnull()]

    y_train = full_train_data.ix[:, response]

    if cols_to_include is None:
        print('You dun goofed')
        quit()
    else:
        x_test_acid = test_data_acid[cols_to_include]
        x_test_pre_rinse = test_data_pre_rinse[cols_to_include]
        x_test_caustic = test_data_caustic[cols_to_include]
        x_test_int_rinse = test_data_int_rinse[cols_to_include]

        if 'process_id' in cols_to_include:
            cols_to_include.remove('process_id')
        x_train = full_train_data[cols_to_include]

        # create dataset for lightgbm
        lgb_train = lgb.Dataset(x_train, y_train)

        return {'full_train': lgb_train,
                'test_acid': x_test_acid,
                'test_pre_rinse': x_test_pre_rinse,
                'test_caustic': x_test_caustic,
                'test_int_rinse': x_test_int_rinse
                }


def build_models(model_type, processed_train_data, processed_val_data, params, response, cols_to_include,
                 train_ratio, max_train_ratio, tuning_params, validation_results, cols, visualize, predictions):

    # Build lightgbm datasets from train and test data
    # Must be repeated for each model to properly simulate data censoring ('cols_to_include' parameter)
    modeling_data = build_lgbm_validation_datasets(processed_train_data, processed_val_data, model_type, response,
                                                   cols_to_include=cols_to_include)

    # Train model
    print('Training ' + model_type + ' model...')
    # train
    gbm_train = lgb.train(params,
                          modeling_data['train'],
                          num_boost_round=5000,
                          valid_sets=modeling_data['eval'],
                          verbose_eval=False,
                          early_stopping_rounds=150,
                          keep_training_booster=True
                          )

    preds = gbm_train._Booster__inner_predict(data_idx=1)
    output_preds = processed_val_data[processed_val_data['row_count_' + model_type].notnull()].reset_index()
    output_preds = output_preds[['process_id', response]]
    output_preds['train_ratio'] = train_ratio
    output_preds['model_type'] = model_type
    output_preds['predicted_response'] = pd.Series(preds)

    predictions = predictions.append(output_preds)

    modeling_data = build_lgbm_validation_datasets(processed_train_data, processed_val_data, model_type, response,
                                                   cols_to_include=cols_to_include)

    if train_ratio == max_train_ratio and visualize is True and model_type == 'acid':
        # explain the model's predictions using SHAP values
        # (same syntax works for LightGBM, CatBoost, and scikit-learn models)
        # matplotlib.pyplot.close()
        matplotlib.pyplot.figure()
        explainer = shap.TreeExplainer(gbm_train)
        shap_data = modeling_data['eval'].data.copy()
        shap_values = explainer.shap_values(shap_data[shap_data.end_residue_acid < 1.5e6])

        # visualize the first prediction's explanation
        # shap.force_plot(explainer.expected_value, shap_values[0, 1], modeling_data['eval'].data.iloc[0, 1],
        #                 matplotlib=True)

        shap.dependence_plot('end_residue_acid', shap_values, shap_data[shap_data.end_residue_acid < 1.5e6])
        # matplotlib.pyplot.figure()
        # shap.dependence_plot('ret_return_acid_ac', shap_values, modeling_data['eval'].data, interaction_index=None)
        # matplotlib.pyplot.figure()
        # shap.dependence_plot('end_residue_acid', shap_values, modeling_data['eval'].data, interaction_index=None)
        # matplotlib.pyplot.figure()
        # shap.dependence_plot('ret_residue_caustic_drain', shap_values, modeling_data['eval'].data, interaction_index=None)
        # matplotlib.pyplot.figure()
        # shap.summary_plot(shap_values, shap_data, plot_type='bar', max_display=500)

    if cols is None:
        cols = 'NA'

    validation_results = validation_results.append(pd.DataFrame([[model_type,
                                                                  train_ratio,
                                                                  cols,
                                                                  tuning_params[0],
                                                                  tuning_params[1],
                                                                  str(tuning_params[2]),
                                                                  round(gbm_train.best_score['valid_0']['mape'], 5),
                                                                  gbm_train.best_iteration]],
                                                   columns=validation_results.columns))

    return predictions, validation_results, modeling_data['eval'].data


def build_test_models(model_type, processed_full_train_data, processed_test_data, response, params, test_iterations,
                      cols_to_include, y_test_pred):

    # Build lgbm data sets on full train and test data
    prediction_data = build_lgbm_test_datasets(processed_full_train_data, processed_test_data, response,
                                               cols_to_include=cols_to_include)

    # Build model on full training data to make predictions for test set
    print('Building model on full training data for ' + model_type + ' model...')

    gbm_full = lgb.train(params,
                         prediction_data['full_train'],
                         num_boost_round=test_iterations[model_type])

    # Make predictions on test set and save to .csv
    print('Making test set predictions for ' + model_type + ' model...')
    print('')

    y_test_pred.append(pd.DataFrame({'process_id': prediction_data['test_' + model_type].process_id,
                                     response: gbm_full.predict(prediction_data['test_' + model_type])}
                                    ))

    return y_test_pred


def calculate_validation_metrics(val_summary):
    test_iterations = {}
    est_test_errors = {}
    phases = ['pre_rinse', 'caustic', 'int_rinse', 'acid']

    print(val_summary)

    for phase in phases:
        test_iterations[phase] = int(val_summary[val_summary.Model_Type == phase].Best_Num_Iters)
        est_test_errors[phase] = round(float(val_summary[val_summary.Model_Type == phase].Best_MAPE), 4)

    for phase in phases:
        print('Best Iterations, ' + phase + ' model: ' + str(test_iterations[phase]))

    print('')
    for phase in phases:
        print('Estimated error for ' + phase + ' predictions: ' + str(est_test_errors[phase]))

    print('')
    print('Estimated total error for all predictions: ' + str(round(292 / 2967 * est_test_errors['pre_rinse'] +
                                                                    1205 / 2967 * est_test_errors['caustic'] +
                                                                    672 / 2967 * est_test_errors['int_rinse'] +
                                                                    798 / 2967 * est_test_errors['acid'], 4)))

    return test_iterations
