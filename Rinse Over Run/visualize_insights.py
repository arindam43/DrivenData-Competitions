import shap
import matplotlib.pyplot
import logging as logger


def plot_shap(model, data, dependence_predictor=None, interaction='auto', cutoff=None):
    matplotlib.pyplot.figure()
    explainer = shap.TreeExplainer(model)
    shap_data = data['eval'].data.copy()

    if dependence_predictor is not None and cutoff is not None:
        shap_data = shap_data[shap_data[dependence_predictor] < cutoff]

    shap_values = explainer.shap_values(shap_data)

    if dependence_predictor is not None:
        logger.info('Plotting SHAP dependence plot for predictor ' + dependence_predictor + '...')
        shap.dependence_plot(dependence_predictor, shap_values, shap_data, interaction_index=interaction)
        matplotlib.pyplot.figure()

    logger.info('Plotting SHAP summary plot (variable importances)...')
    shap.summary_plot(shap_values, shap_data, plot_type='bar', max_display=500)
    logger.info('SHAP plots created successfully.')
