import lightgbm as lgb
import pandas as pd
import re
import numpy as np
import shap
import matplotlib


def subset_modeling_columns(processed_train_data, cols_subset=None):
    # For each of the four models, identify which columns should be kept from overall set
    # Simulates data censoring in test data
    non_phase_cols_short = ['object_id', 'recipe_type']
    non_phase_cols_full = ['object_id']

    if cols_subset is None:
        pre_rinse_cols = set(filter(lambda x: re.search(r'(?=.*residue|.*cond|.*temp)', x), list(processed_train_data.columns)))
        caustic_cols = set(filter(lambda x: re.search(r'(?=.*residue|.*cond|.*temp)', x), list(processed_train_data.columns)))
        int_rinse_cols = set(filter(lambda x: re.search(r'(?=.*flow|.*residue)', x), list(processed_train_data.columns)))
        acid_cols = set(filter(lambda x: re.search(r'(?=.*flow|.*turb|.*supply)', x), list(processed_train_data.columns)))
        none_cols = set(filter(lambda x: re.search(r'(?=.*none|row_count.*)', x), list(processed_train_data.columns)))

        cols_to_include = {'pre_rinse': list(set(filter(lambda x: re.search(r'(?=.*pre_rinse)', x),
                                                        list(processed_train_data.columns))) - none_cols - pre_rinse_cols) + non_phase_cols_short,
                           'caustic': list(set(filter(lambda x: re.search(r'(?=.*pre_rinse|.*caustic)', x),
                                                      list(processed_train_data.columns))) - none_cols - caustic_cols) + non_phase_cols_short,
                           'int_rinse': list(set(filter(lambda x: re.search(r'(?=.*pre_rinse|.*caustic|.*int_rinse)', x),
                                          list(processed_train_data.columns))) - none_cols - int_rinse_cols) + non_phase_cols_full,
                           'acid': list(set(filter(lambda x: re.search(r'(?=.*pre_rinse|.*caustic|.*int_rinse|.*acid|.*other)', x),
                                   list(processed_train_data.columns))) - none_cols - acid_cols) + non_phase_cols_full
                           }

    else:
        none_cols = set(filter(lambda x: re.search(r'(?=.*none|row_count.*)', x), list(processed_train_data.columns)))
        exclude_cols = set(filter(lambda x: re.search(r'(?=.*' + cols_subset + ')', x), list(processed_train_data.columns)))

        cols_to_include = {'pre_rinse': list(set(filter(lambda x: re.search(r'(?=.*pre_rinse)', x),
                                                        list(processed_train_data.columns))) - exclude_cols - none_cols) + non_phase_cols_short,
                           'caustic': list(set(filter(lambda x: re.search(r'(?=.*pre_rinse|.*caustic)', x),
                                                      list(processed_train_data.columns))) - exclude_cols - none_cols) + non_phase_cols_short,
                           'int_rinse': list(set(filter(lambda x: re.search(r'(?=.*pre_rinse|.*caustic|.*int_rinse)', x),
                                          list(processed_train_data.columns))) - exclude_cols - none_cols) + non_phase_cols_full,
                           'acid': list(set(filter(lambda x: re.search(r'(?=.*pre_rinse|.*caustic|.*int_rinse|.*acid|.*other)', x),
                                   list(processed_train_data.columns))) - exclude_cols - none_cols) + non_phase_cols_full
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
                 train_ratio, max_train_ratio, tuning_params, validation_results, cols, visualize):

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
                          early_stopping_rounds=40
                          )

    modeling_data = build_lgbm_validation_datasets(processed_train_data, processed_val_data, model_type, response,
                                                   cols_to_include=cols_to_include)

    if train_ratio == max_train_ratio and visualize is True:
        # explain the model's predictions using SHAP values
        # (same syntax works for LightGBM, CatBoost, and scikit-learn models)
        # matplotlib.pyplot.close()
        matplotlib.pyplot.figure()
        explainer = shap.TreeExplainer(gbm_train)
        shap_values = explainer.shap_values(modeling_data['eval'].data)

        # visualize the first prediction's explanation
        # shap.force_plot(explainer.expected_value, shap_values[0, :], modeling_data['eval_acid'].data.iloc[0, :], matplotlib=True)
        # shap.dependence_plot('total_turbidity_acid', shap_values, modeling_data['eval_acid'].data)
        # shap.summary_plot(shap_values, modeling_data['eval_' + model_type].data)
        shap_title = 'Model Type: ' + model_type
        shap.summary_plot(shap_values, modeling_data['eval'].data, plot_type='bar', max_display=500,
                          title=shap_title)

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

    return validation_results


def build_test_models(model_type, processed_full_train_data, processed_test_data, response, params, test_iterations,
                      cols_to_include, y_test_pred):

    # Build lgbm data sets on full train and test data
    prediction_data = build_lgbm_test_datasets(processed_full_train_data, processed_test_data, response,
                                               cols_to_include=cols_to_include)

    # Build model on full training data to make predictions for test set
    print('Building model on full training data...')

    gbm_full = lgb.train(params,
                         prediction_data['full_train'],
                         num_boost_round=test_iterations[model_type])

    # Make predictions on test set and save to .csv
    print('Making test set predictions...')

    y_test_pred.append(pd.DataFrame({'process_id': prediction_data['test_' + model_type].process_id,
                                     response: gbm_full.predict(prediction_data['test_' + model_type])}
                                    ))

    return y_test_pred


def calculate_validation_metrics(validation_summary):
    test_iterations = {'pre_rinse': int(validation_summary[validation_summary.Model_Type == 'pre_rinse'].Best_Num_Iters),
                       'caustic': int(validation_summary[validation_summary.Model_Type == 'caustic'].Best_Num_Iters),
                       'int_rinse': int(validation_summary[validation_summary.Model_Type == 'int_rinse'].Best_Num_Iters),
                       'acid': int(validation_summary[validation_summary.Model_Type == 'acid'].Best_Num_Iters)
                       }

    est_error_pre_rinse = round(float(validation_summary[validation_summary.Model_Type == 'pre_rinse'].Best_MAPE), 4)
    est_error_acid = round(float(validation_summary[validation_summary.Model_Type == 'acid'].Best_MAPE), 4)
    est_error_caustic = round(float(validation_summary[validation_summary.Model_Type == 'caustic'].Best_MAPE), 4)
    est_error_int_rinse = round(
        float(validation_summary[validation_summary.Model_Type == 'int_rinse'].Best_MAPE), 4)

    print(validation_summary)
    print('Best Iterations, pre-rinse model: ' + str(test_iterations['pre_rinse']))
    print('Best Iterations, caustic model: ' + str(test_iterations['caustic']))
    print('Best Iterations, intermediate-rinse model: ' + str(test_iterations['int_rinse']))
    print('Best Iterations, acid model: ' + str(test_iterations['acid']))
    print('')
    print('Estimated error for pre-rinse predictions: ' + str(est_error_pre_rinse))
    print('Estimated error for caustic predictions: ' + str(est_error_caustic))
    print('Estimated error for intermediate-rinse predictions: ' + str(est_error_int_rinse))
    print('Estimated error for acid predictions: ' + str(est_error_acid))
    print('')
    print('Estimated total error for all predictions: ' + str(round(292 / 2967 * est_error_pre_rinse +
                                                                    1205 / 2967 * est_error_caustic +
                                                                    672 / 2967 * est_error_int_rinse +
                                                                    798 / 2967 * est_error_acid, 4)))

    return test_iterations

