import lightgbm as lgb
import pandas as pd
import shap
from IPython.core.display import HTML


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
    cols_to_include = cols_to_include + ['process_id']

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


def build_models(model_type, processed_train_data, processed_val_data, params, response, cols_to_include,
                 train_ratio, max_train_ratio, validation_results):


    # Modeling
    modeling_data = build_lgbm_validation_datasets(processed_train_data, processed_val_data, response,
                                                   cols_to_include=cols_to_include)

    print('Starting training...')
    # train
    gbm_train = lgb.train(params,
                          modeling_data['train'],
                          num_boost_round=2000,
                          valid_sets=modeling_data['eval_' + model_type],
                          verbose_eval=2000,
                          early_stopping_rounds=30)

    modeling_data = build_lgbm_validation_datasets(processed_train_data, processed_val_data, response,
                                                   cols_to_include=cols_to_include)

    if train_ratio == max_train_ratio and model_type == 'acid':
        #lgb.plot_importance(gbm_train)

        # explain the model's predictions using SHAP values
        # (same syntax works for LightGBM, CatBoost, and scikit-learn models)
        explainer = shap.TreeExplainer(gbm_train)
        shap_values = explainer.shap_values(modeling_data['eval_' + model_type].data)

        # visualize the first prediction's explanation
        # shap.force_plot(explainer.expected_value, shap_values[0, :], modeling_data['eval_acid'].data.iloc[0, :], matplotlib=True)
        # shap.dependence_plot('total_turbidity_acid', shap_values, modeling_data['eval_acid'].data)
        # shap.summary_plot(shap_values, modeling_data['eval_' + model_type].data)
        shap.summary_plot(shap_values, modeling_data['eval_' + model_type].data, plot_type='bar')

    validation_results = validation_results.append(pd.DataFrame([[model_type,
                                                                  train_ratio,
                                                                  round(gbm_train.best_score['valid_0']['mape'], 5),
                                                                  gbm_train.best_iteration]],
                                                   columns=validation_results.columns))

    return validation_results


def calculate_validation_metrics(validation_results):
    test_iterations_pre_rinse = int(
        round(validation_results[validation_results.Model_Type == 'pre_rinse'].Best_Num_Iters.mean()))
    test_iterations_acid = int(round(validation_results[validation_results.Model_Type == 'acid'].Best_Num_Iters.mean()))
    test_iterations_caustic = int(
        round(validation_results[validation_results.Model_Type == 'caustic'].Best_Num_Iters.mean()))
    test_iterations_int_rinse = int(
        round(validation_results[validation_results.Model_Type == 'int_rinse'].Best_Num_Iters.mean()))
    test_iterations = {'pre_rinse': test_iterations_pre_rinse,
                       'caustic': test_iterations_caustic,
                       'int_rinse': test_iterations_int_rinse,
                       'acid': test_iterations_acid}

    est_error_pre_rinse = round(validation_results[validation_results.Model_Type == 'pre_rinse'].Best_MAPE.mean(), 4)
    est_error_acid = round(validation_results[validation_results.Model_Type == 'acid'].Best_MAPE.mean(), 4)
    est_error_caustic = round(validation_results[validation_results.Model_Type == 'caustic'].Best_MAPE.mean(), 4)
    est_error_int_rinse = round(
        validation_results[validation_results.Model_Type == 'int_rinse'].Best_MAPE.mean(), 4)

    print(validation_results.sort_values(by=['Model_Type', 'Train_Ratio']))
    print('Best Iterations, pre-rinse model: ' + str(test_iterations_pre_rinse))
    print('Best Iterations, caustic model: ' + str(test_iterations_caustic))
    print('Best Iterations, intermediate-rinse model: ' + str(test_iterations_int_rinse))
    print('Best Iterations, acid model: ' + str(test_iterations_acid))
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


def build_test_models(model_type, processed_full_train_data, processed_test_data, response, params, test_iterations,
                      cols_to_include, y_test_pred):

    # Build lgbm data sets on full train and test data
    prediction_data = build_lgbm_test_datasets(processed_full_train_data, processed_test_data, response, cols_to_include=cols_to_include)

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
