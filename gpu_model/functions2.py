from gpu_model.gpu_pso import *
from gpu_model.gpu import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm

#Adapted from TFT to GPU
def plot_prediction(prediction, observed, temperature=None, precipitation=None, freq=None, target_label='Discharge [m³/s]',
                    add_markers=False):                   
    '''
    Plot the model predictions against the observed data, while syncronizing also precipitation and temperature.

    Input:
        > prediction - represent the predictions of the model
        > observed - represent the observed data
        > temperature - represent the temperature data
        > precipitation - represent the precipitation data
'''
    
    color_observed = 'darkturquoise'
    color_median = 'crimson'
    
    try:
        training = prediction.loc[:, ['Training']]
    except Exception:
        training = pd.DataFrame(np.zeros((prediction.shape[0], 1), dtype=bool), index=prediction.index)
    try:
        validation = prediction.loc[:, ['Validation']]
    except Exception:
        validation = pd.DataFrame(np.zeros((prediction.shape[0], 1), dtype=bool), index=prediction.index)
 
    quantiles = prediction.columns
    quantiles_idx = [i for i, j in enumerate(quantiles) if type(j) != str]  # Remove any string entries if present
    quantiles = quantiles[quantiles_idx]
    
    complete_idx = pd.date_range(observed.index.min(), observed.index.max(), freq=freq)
    observed = observed.reindex(complete_idx, fill_value=np.nan)
    observed_ = observed
    prediction = prediction.reindex(complete_idx, fill_value=np.nan)
    missing = observed.isna()
    training = training.reindex(complete_idx, fill_value=False)
    validation = validation.reindex(complete_idx, fill_value=False)
    
    fig, (ax_training, ax_temperature, ax_precipitation, ax_forecast) = plt.subplots(4, 1, 
                                                                                     gridspec_kw={'height_ratios': [0.15, 1, 1, 2]}, 
                                                                                     constrained_layout=True, 
                                                                                     sharex=True, 
                                                                                     sharey=False, 
                                                                                     figsize=[11.17,  6.26])

    # Forecast plot
   
    legendSamples = []
    legendLabels = []
    if len(quantiles)>1:
        color_len = len(quantiles)//2
        color = np.linspace(0.87, 0, color_len)
        colors = np.repeat(np.expand_dims(color, 1), 3, axis=1)

        i0 = 0
        predictions_ = prediction.drop(columns=['Training', 'Validation'], errors='ignore')
        predictions_ = predictions_.drop(columns=[ci for ci in predictions_.columns if ci not in quantiles], errors='ignore')
        for i0 in np.arange(0, len(quantiles)//2):
            tmp = predictions_.iloc[:,[i0, len(quantiles)-i0-1]]
            ax_forecast.fill_between(predictions_.index, tmp.iloc[:, 0], tmp.iloc[:, -1], facecolor=colors[i0,], linewidth=0, alpha=1, zorder=i0)
            legendSamples.append(plt.Rectangle((0, 0), 1, 1, fc=colors[i0,], linewidth=0))
            quantiles_ = tmp.columns
            legendLabels.append('{f:03.1f}-{t:03.1f}%'.format(f=quantiles_[0]*100, t=quantiles_[-1]*100))

        center_ = predictions_.loc[:, :]
        center = median_prediction(center_, multiIndex=False)
        ax_forecast.plot(center, linestyle='--', color=color_median, zorder=i0+1)
    
        legendLabels.append('Median prediction')
        
    observed.plot(ax=ax_forecast, color=color_observed, zorder=prediction.shape[1]//2, alpha=0.75)
        
    legendSamples.append(Line2D([0], [0], color=color_observed, linestyle='-'))
    legendLabels.append('Observed')

    ax_forecast.legend(legendSamples, legendLabels, fontsize=10, numpoints=1, loc=2, frameon=False, ncol=len(legendLabels)//2)
    ax_forecast.set_ylabel(target_label)
    ax_forecast.grid(True)
    
    # Precipitation plot
    if precipitation is not None:
        legendSamples = [Line2D([0], [0], color='k')]
        legendLabels = ['Precipitation']
        
        ax_precipitation.plot(precipitation.index, precipitation.sum(axis=1), color='blue', label='Precipitation')
        ax_precipitation.set_ylabel('Precipitation [mm]')
        ax_precipitation.grid(True)
        ax_precipitation.set_ylim(0, 1.2* precipitation.sum(axis=1).max()) 
        ax_precipitation.invert_yaxis()
         # Reverse y-axis

    # Temperature plot
    if temperature is not None:
        legendSamples = [Line2D([0], [0], color='r')]
        ax_temperature.plot(temperature.index, temperature.mean(axis=1), color='r', zorder=2)
        ax_temperature.set_ylabel('Temperature [°C]')
        ax_temperature.grid(True)
        ax_temperature.yaxis.label.set_color('r')
        ax_temperature.set_ylim(temperature.min().min() - 5, temperature.max().max() + 5)
    if True:
        training_mask = training.values.ravel().astype(bool)
        validation_mask = validation.values.ravel().astype(bool)
        # test_mask = test.values.ravel().astype(bool)
        no_input_mask = missing.values.ravel().astype(bool)
 
        ax_training.fill_between(prediction.index, 0, 1, where=training_mask, transform=ax_training.get_xaxis_transform(), color='snow')
        ax_training.fill_between(prediction.index, 0, 1, where=validation_mask, transform=ax_training.get_xaxis_transform(), color='steelblue')
        # ax_training.fill_between(prediction.index, 0, 1, where=test_mask, transform=ax_training.get_xaxis_transform(), color='darkslategray')
        ax_training.fill_between(prediction.index, 0, 1, where=no_input_mask, transform=ax_training.get_xaxis_transform(), color='lightcoral')


        legendSamples = [plt.Rectangle((0, 0), 1, 1, fc='snow', ec='k', linewidth=1),
                         plt.Rectangle((0, 0), 1, 1, fc='steelblue', ec='k', linewidth=1),
                        #  plt.Rectangle((0, 0), 1, 1, fc='darkslategray', ec='k', linewidth=1),
                         plt.Rectangle((0, 0), 1, 1, fc='lightcoral', ec='k', linewidth=1)]
        legendLabels = ['Training', 'Validation', 'Missing'] # 'Test' is not used in this context, but can be added back if needed

        ax_training.legend(legendSamples, legendLabels, ncol=len(legendLabels), frameon=False, loc='lower left', bbox_to_anchor=(0, 1.05))
        _ = ax_training.set_yticks([])

    tmp = prediction.index
    min_x = tmp.min()
    max_x = tmp.max()
    ax_forecast.set_xlim(min_x, max_x)

    return fig

def median_prediction(predictions, multiIndex=True):
    try:
        prob_levels = predictions.columns.get_level_values('Probability').unique()
    except:
        prob_levels = predictions.columns.unique()

    lower_prob = prob_levels[prob_levels <= 0.5].max()
    upper_prob = prob_levels[prob_levels >= 0.5].min()

    try:
        lower_pred = predictions.xs(lower_prob, level='Probability', axis=1)
        upper_pred = predictions.xs(upper_prob, level='Probability', axis=1)
    except:
        lower_pred = predictions[lower_prob]
        upper_pred = predictions[upper_prob]
    
    if lower_prob == upper_prob:
        median_pred = lower_pred
    else:
        median_pred = lower_pred  + (upper_pred - lower_pred) * (0.5 - lower_prob) / (upper_prob - lower_prob)
    
    if multiIndex:
        median_pred.columns = pd.MultiIndex.from_tuples([(pd.Timedelta('0D'), 'Simulated')], names=['Leadtime', 'Deterministic'])

    return median_pred
        
     