from gpu_pso import *
from gpu import *
from error.errorOpenCL import *
from regression.annOpenCL import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
import matplotlib.cm as cm
import pickle
import os
from performance import ForecastPerformance, Results

def toCustomLogSpace(x):
	with np.errstate(divide='ignore'):
		y = np.empty_like(x)
		tmp = x<0.5
		y[tmp] = np.log(x[tmp])-np.log(0.5)
		tmp = np.logical_not(tmp)
		y[tmp] = -np.log(1-x[tmp])+np.log(0.5)
		
	return y

def fromCustomLogSpace(y):
	'''
	Transforms band probabilities back from a custom log space for easier interpolation of the extremes and calculation of quantiles
	'''
	with np.errstate(divide='ignore'):
		x = np.empty_like(y)
		tmp = y < 0
		x[tmp] = np.exp(y[tmp] + np.log(0.5))
		tmp = np.logical_not(tmp)
		x[tmp] = 1 - np.exp(-y[tmp] + np.log(0.5))
	return x


def timeSeriesPvalues(aggregated, targets, bands):
	# make targets into a numpy array (if needed)
	if isinstance(targets, (pd.core.frame.DataFrame, pd.core.series.Series)):
		targets = targets.values
	
	# Main loop
	bands = toCustomLogSpace(np.array(bands)[::-1])
	pValues = np.empty(targets.shape[0]) # on the custom log space
	for i0 in range(pValues.shape[0]):
		# drop nans
		tmp = np.isfinite(aggregated[i0, :])
		tmpBands = bands[tmp]
		tmpAggregated = aggregated[i0, tmp]
		
		# sort and discard repeated values
		sims, idxs = np.unique(tmpAggregated, return_index=True)
		
		# calculate the p-value of the observation (on the custom log space)
		#Debugging
		try:
			if targets[i0]<sims[0]:
				# targets overestimated
				pValues[i0] = np.inf
			elif targets[i0]>sims[-1]:
				# targets underestimated
				pValues[i0] = -np.inf
			else:
				pValues[i0] = interp1d(sims, tmpBands[idxs], kind='linear', assume_sorted=True)(targets[i0])
				
		except IndexError:
			# print('Lack of models to calculate Aggregated')
			# pValues[pValues<0] = 0
			# pValues[pValues>1] = 1
			# return pValues
			raise Exception('Pvalues Calculation Error')    
	
	# restore p-values to their original space
	pValues = fromCustomLogSpace(pValues)
	# pValues[pValues<0] = 0
	# pValues[pValues>1] = 1
	return 1-pValues

def plotPvalues(pValues, targets, ax=None, filename=None):
	# make targets into a numpy array (if needed)
	
	# plot
	if ax is None:
		fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
		ax[0].plot(targets.index, pValues, '.', label='p-values')
		ax[0].set_xlabel('Theoretical pValues U[0, 1]')
		ax[0].set_ylabel('Observed p-values')
		ax[1].plot(targets.index, targets, linestyle='solid', color='red', label='Observed', alpha=0.5)
		plt.tight_layout()
		if filename is not None:
			plt.savefig(filename)
		plt.show(block=False)
	else:
		ax.plot(targets.index, pValues, '.', label='p-values')
		ax.set_ylabel('Observed p-value')

#Adapted from TFT to GPU
def plot_prediction(prediction, observed, temperature, precipitation, basin_areas=None, best_deterministic=None, hype=None, lead=pd.Timedelta('0D'),
                    snow=None, freq=None, target_label='Discharge [m³/s]', snow_label=None,
                    plot_persistence=False, add_markers=False, filename=None):                   
    '''
    Plot the model predictions against the observed data, while syncronizing also precipitation and temperature.

    Input:
        > prediction - represent the predictions of the model
        > observed - represent the observed data
        > temperature - represent the temperature data
        > precipitation - represent the precipitation data
        > basin_areas - a dataframe with the areas of the subbasins associated with its IDs, this makes possible to compute the PAveage, important for the precipitation
        > others - come from the original form of this function and are not explicitly defined here.
    '''
    
    color_observed = 'darkturquoise'
    color_median = 'crimson'
    color_snow = 'skyblue'
    
    try:
        training = prediction.loc[:, ['Training']]
    except Exception:
        training = pd.DataFrame(np.zeros((prediction.shape[0], 1), dtype=bool), index=prediction.index)
    try:
        validation = prediction.loc[:, ['Validation']]
    except Exception:
        validation = pd.DataFrame(np.zeros((prediction.shape[0], 1), dtype=bool), index=prediction.index)
    # try:
    #     test = prediction.loc[:, ['Test']]
    # except Exception:
    #     test = pd.DataFrame(np.zeros((prediction.shape[0], 1), dtype=bool), index=prediction.index)
    quantiles = prediction.columns.get_level_values('Probability').unique().to_numpy()
    quantiles_idx = [i for i, j in enumerate(quantiles) if type(j) != str]  # Remove any string entries if present
    quantiles = quantiles[quantiles_idx]
        #Remove band 0.1% to 99.9%
    if 0.001 in quantiles and 0.999 in quantiles:
        quantiles = quantiles[(quantiles>0.001) & (quantiles<0.999)]
    
    complete_idx = pd.date_range(observed.index.min(), observed.index.max(), freq=freq)
    observed = observed.reindex(complete_idx, fill_value=np.nan)
    observed_ = observed
    prediction = prediction.reindex(complete_idx, fill_value=np.nan)
    missing = observed.isna()
    training = training.reindex(complete_idx, fill_value=False)
    validation = validation.reindex(complete_idx, fill_value=False)
    # test = test.reindex(complete_idx, fill_value=False)
    
    
    cm2inch = lambda x, y: (x/2.54, y/2.54)  # Convert cm to inches for matplotlib
    fig, (ax_training, ax_temperature, ax_precipitation, ax_forecast) = plt.subplots(4, 1, gridspec_kw={'height_ratios': [0.15, 1, 1, 2]}, constrained_layout=True, sharex=True, sharey=False, figsize=[11.17,  6.26])

    # Forecast plot
    if True:
        legendSamples = []
        legendLabels = []
        if len(quantiles)>1:
            color_len = len(quantiles)//2
            color = np.linspace(0.87, 0, color_len)
            #coverage = np.array([j-i for i, j in zip(quantiles[:color_len], quantiles[::-1][:color_len])])
            #coverage -= coverage.min()
            #coverage += 0.01 * coverage.max()
            #coverage /= coverage.max()
            #colors = np.repeat(np.expand_dims(coverage, 1), 3, axis=1)
            colors = np.repeat(np.expand_dims(color, 1), 3, axis=1)
            
            #=======================================================================
            # colors = np.repeat(np.expand_dims(np.linspace(0.95, 0, len(quantiles)//2), 1), 3, axis=1)
            #=======================================================================
            i0 = 0
            predictions_ = prediction.drop(columns=['Training', 'Validation'], level='Leadtime', errors='ignore')
            predictions_ = predictions_.drop(columns=[ci for ci in predictions_.columns.get_level_values('Probability') if ci not in quantiles], level='Probability', errors='ignore')
            for i0 in np.arange(0, len(quantiles)//2):
                tmp = predictions_.iloc[:,[i0, len(quantiles)-i0-1]]
                ax_forecast.fill_between(predictions_.index, tmp.iloc[:, 0], tmp.iloc[:, -1], facecolor=colors[i0,], linewidth=0, alpha=1, zorder=i0)
                legendSamples.append(plt.Rectangle((0, 0), 1, 1, fc=colors[i0,], linewidth=0))
                quantiles_ = tmp.columns.get_level_values('Probability')
                legendLabels.append('{f:03.1f}-{t:03.1f}%'.format(f=quantiles_[0]*100, t=quantiles_[-1]*100))

            center_ = prediction.loc[:, (predictions_.columns.get_level_values('Leadtime')[0], prediction.columns.get_level_values('Probability'))]
            center = median_prediction(center_, multiIndex=True)
            ax_forecast.plot(center, linestyle='--', color=color_median, zorder=i0+1)
            
            if len(quantiles)>1:
                ax_forecast.plot(prediction.iloc[:, 0], linestyle=':', color=[0.85] * 3)
                ax_forecast.plot(prediction.iloc[:, -1], linestyle=':', color=[0.85] * 3)
        
            legendSamples.append(Line2D([0], [0], color=color_median, linestyle='--'))
            legendLabels.append('Median prediction')
        else:
            if add_markers:
                ax_forecast.plot(prediction.loc[:, prediction.columns.get_level_values('Type')!='Observed'], linestyle='-', marker='o', markersize=3, color=color_median, zorder=1)
            else:
                ax_forecast.plot(prediction.loc[:, prediction.columns.get_level_values('Type')!='Observed'], linestyle='-', color=color_median, zorder=1)
            legendSamples.append(Line2D([0], [0], color=color_median, linestyle='-'))
            legendLabels.append('Prediction')
            
        observed.plot(ax=ax_forecast, color=color_observed, zorder=prediction.shape[1]//2, alpha=0.75)
          
        #=======================================================================
        # if not not_observed.empty:
        #     not_observed.plot(ax=ax_forecast, color=color_no_observed, zorder=pred.shape[1]//2)
        #=======================================================================
              
        legendSamples.append(Line2D([0], [0], color=color_observed, linestyle='-'))
        legendLabels.append('Observed')

        if best_deterministic is not None:
            best_deterministic.plot(ax=ax_forecast, color='orange', linestyle='-.', zorder=prediction.shape[1]//2, alpha=0.75)
            legendSamples.append(Line2D([0], [0], color='orange', linestyle='-.'))
            legendLabels.append(f'Best deterministic nexc={best_deterministic.loc[best_deterministic.values>observed.values].shape[0]/best_deterministic.shape[0]:.2f}')

        if hype is not None:
            hype.plot(ax=ax_forecast, color='orange', linestyle='-.', zorder=prediction.shape[1]//2, alpha=0.75)
            legendSamples.append(Line2D([0], [0], color='orange', linestyle='-.'))
            legendLabels.append(f'Results from automatic calibration')
        #=======================================================================
        # legendSamples.append(Line2D([0], [0], color=color_no_observed, linestyle='-'))
        # legendLabels.append('Observed NA')
        #=======================================================================
        
        if plot_persistence:
            persistance = observed.copy()
            persistance.index += lead
            ax_forecast.plot(persistance, linestyle='-.', color=color_observed, zorder=prediction.shape[1]//2, alpha=0.25)
            legendSamples.append(Line2D([0], [0], color=color_observed, linestyle='-.', alpha=0.25))
            legendLabels.append('Persistence')

        ax_forecast.legend(legendSamples, legendLabels, fontsize=10, numpoints=1, loc=2, frameon=False, ncol=len(legendLabels)//2) #ncol=len(legendLabels)//2
        ax_forecast.set_ylabel(target_label)
        ax_forecast.grid(True)
    
    # Precipitation plot
    if True:
        legendSamples = [Line2D([0], [0], color='k')]
        legendLabels = ['Precipitation']
        
        try:
            if 'Leadtime' == observed_.columns.names[1]:
                ax_precipitation.plot(observed_.index, observed_.loc[:, (precipitation, pd.Timedelta('0d'))], color='k')
                ax_precipitation.plot(observed_.index, observed_.loc[:, (precipitation, lead)], color='lightblue', linestyle='--')
                legendSamples += [Line2D([0], [0], color='lightblue')]
                legendLabels += ['Precipitation forecast']
            else:
                raise(Exception())
        except Exception:
            if np.all(basin_areas) != None:
                #Match types of the SUBID in both dataframes
                basin_areas.index = basin_areas.index.astype(int)
                precipitation.columns = precipitation.columns.astype(int)
                legendLabels = ['Ponderated average precipitation across all subbasins']

                ax_precipitation.plot(precipitation.index, (precipitation*basin_areas['AREA']).sum(axis=1)/basin_areas['AREA'].sum(axis=0), color='k')
                ax_precipitation.set_ylim(0, 1.2*precipitation.max().max())
            else:
                # Assuming precipitation is a DataFrame with multiple columns (e.g., subbasins)
                cmap = cm.get_cmap('Blues')
                colors = cmap(np.linspace(0, 1, len(precipitation.columns)))
                for i, col in enumerate(precipitation.columns):
                    ax_precipitation.plot(precipitation.index, precipitation[col], color=colors[i], label=col)
                # Try to keep all labels on a single row, but if there are too many,
                # split into multiple rows with near-equal lengths.
                n_items = len(precipitation.columns)
                max_cols_per_row = 7
                n_rows = int(np.ceil(max(1, n_items) / max_cols_per_row))
                ncol = int(np.ceil(max(1, n_items) / n_rows))
                ax_precipitation.legend(frameon=False, ncol=8, fontsize=10, loc='lower center')
        ax_precipitation.set_ylabel('Precipitation [mm]')
        ax_precipitation.grid(True)
        ax_precipitation.set_ylim(0, 1.2*precipitation.max().max()) 
        ax_precipitation.invert_yaxis()
         # Reverse y-axis

    # Temperature plot
    if True:
        legendSamples = [Line2D([0], [0], color='r')]
        legendLabels = ['Temperature']
        
        try:
            if 'Leadtime' == observed_.columns.names[1]:
                ax_temperature.plot(observed_.index, observed_.loc[:, (temperature, pd.Timedelta('0d'))], color='r', zorder=2)
                ax_temperature.plot(observed_.index, observed_.loc[:, (temperature, lead)], color='orange', linestyle='--')

                legendSamples.append(Line2D([0], [0], color='orange', linestyle='--'))
                legendLabels.append('Temperature Forecast')
            else:
                raise(Exception())
        except Exception:
            ax_temperature.plot(temperature.index, temperature.mean(axis=1), color='r', zorder=2)
            ax_temperature.set_ylabel('Temperature [C]')
        ax_temperature.grid(True)
        ax_temperature.yaxis.label.set_color('r')
        ax_temperature.set_ylim(temperature.min().min() - 5, temperature.max().max() + 5)

        if snow:
            legendSamples.append(plt.Rectangle((0, 0), 1, 1, fc=color_snow, ec=color_snow, linewidth=1))
            legendLabels.append('Snow')
            
            ax_snow = ax_temperature.twinx()
            ax_snow.yaxis.label.set_color(color_snow)
            
            try:
                if 'Leadtime' == observed_.columns.names[1]:
                    tmp = observed_.loc[:, (snow, pd.Timedelta('0d'))].applymap(lambda x: 0 if x<0 else x)
            except Exception:
                tmp = observed_.loc[:, snow].applymap(lambda x: 0 if x<0 else x)
            
            ax_snow.plot(tmp.index, tmp, color=color_snow, zorder=-1)
            plt.fill_between(tmp.index, tmp.values.ravel(), color=color_snow, alpha=0.5, zorder=1)
            
            ax_snow.set_ylabel(snow_label)

        ax_temperature.legend(legendSamples, legendLabels, frameon=False)
    
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

    plt.show(block=False)
    
    if filename:
        pickle.dump(fig, open(filename, 'wb'))

    return fig

#Adapted from TFT to GPU
def plot_prediction_with_MLP(prediction, observed, MLP, temperature, precipitation, basin_areas=None, best_deterministic=None, hype=None, lead=pd.Timedelta('0D'),
                    snow=None, freq=None, target_label='Discharge [m³/s]', snow_label=None,
                    plot_persistence=False, add_markers=False, filename=None):                   
    '''
    Plot the model predictions against the observed data, while syncronizing also precipitation and temperature.

    Input:
        > prediction - represent the predictions of the model
        > observed - represent the observed data
        > temperature - represent the temperature data
        > precipitation - represent the precipitation data
        > MLP - represent the MLP model predictions
        > others - come from the original form of this function and are not explicitly defined here.
    '''
    
    color_observed = 'darkturquoise'
    color_median = 'crimson'
    color_snow = 'skyblue'
    
    try:
        training = prediction.loc[:, ['Training']]
    except Exception:
        training = pd.DataFrame(np.zeros((prediction.shape[0], 1), dtype=bool), index=prediction.index)
    try:
        validation = prediction.loc[:, ['Validation']]
    except Exception:
        validation = pd.DataFrame(np.zeros((prediction.shape[0], 1), dtype=bool), index=prediction.index)
    # try:
    #     test = prediction.loc[:, ['Test']]
    # except Exception:
    #     test = pd.DataFrame(np.zeros((prediction.shape[0], 1), dtype=bool), index=prediction.index)
    quantiles = prediction.columns.get_level_values('Probability').unique().to_numpy()
    quantiles_idx = [i for i, j in enumerate(quantiles) if type(j) != str]  # Remove any string entries if present
    quantiles = quantiles[quantiles_idx]
        #Remove band 0.1% to 99.9%
    if 0.001 in quantiles and 0.999 in quantiles:
        quantiles = quantiles[(quantiles>0.001) & (quantiles<0.999)]
    
    
    complete_idx = pd.date_range(observed.index.min(), observed.index.max(), freq=freq)
    observed = observed.reindex(complete_idx, fill_value=np.nan)
    observed_ = observed
    prediction = prediction.reindex(complete_idx, fill_value=np.nan)
    missing = observed.isna()
    training = training.reindex(complete_idx, fill_value=False)
    validation = validation.reindex(complete_idx, fill_value=False)
    # test = test.reindex(complete_idx, fill_value=False)
    
    
    cm2inch = lambda x, y: (x/2.54, y/2.54)  # Convert cm to inches for matplotlib
    fig, (ax_training, ax_temperature, ax_precipitation, ax_forecast, ax_MLP_forecast) = plt.subplots(5, 1, gridspec_kw={'height_ratios': [0.15, 1, 1, 2, 2]}, constrained_layout=True, sharex=True, sharey=False, figsize=cm2inch(40, 30))

    # Forecast plot
    if True:
        legendSamples = []
        legendLabels = []
        if len(quantiles)>1:
            color_len = len(quantiles)//2
            color = np.linspace(0.87, 0, color_len)
            #coverage = np.array([j-i for i, j in zip(quantiles[:color_len], quantiles[::-1][:color_len])])
            #coverage -= coverage.min()
            #coverage += 0.01 * coverage.max()
            #coverage /= coverage.max()
            #colors = np.repeat(np.expand_dims(coverage, 1), 3, axis=1)
            colors = np.repeat(np.expand_dims(color, 1), 3, axis=1)
            
            #=======================================================================
            # colors = np.repeat(np.expand_dims(np.linspace(0.95, 0, len(quantiles)//2), 1), 3, axis=1)
            #=======================================================================
            i0 = 0
            predictions_ = prediction.drop(columns=['Training', 'Validation'], level='Leadtime', errors='ignore')
            predictions_ = predictions_.drop(columns=[ci for ci in predictions_.columns.get_level_values('Probability') if ci not in quantiles], level='Probability', errors='ignore')
            for i0 in np.arange(0, len(quantiles)//2):
                tmp = predictions_.iloc[:,[i0, len(quantiles)-i0-1]]
                ax_forecast.fill_between(predictions_.index, tmp.iloc[:, 0], tmp.iloc[:, -1], facecolor=colors[i0,], linewidth=0, alpha=1, zorder=i0)
                legendSamples.append(plt.Rectangle((0, 0), 1, 1, fc=colors[i0,], linewidth=0))
                quantiles_ = tmp.columns.get_level_values('Probability')
                legendLabels.append('{f:03.1f}-{t:03.1f}%'.format(f=quantiles_[0]*100, t=quantiles_[-1]*100))

            center_ = prediction.loc[:, (predictions_.columns.get_level_values('Leadtime')[0], prediction.columns.get_level_values('Probability'))]
            center = median_prediction(center_, multiIndex=True)
            ax_forecast.plot(center, linestyle='--', color=color_median, zorder=i0+1)
            
            if len(quantiles)>1:
                ax_forecast.plot(prediction.iloc[:, 0], linestyle=':', color=[0.85] * 3)
                ax_forecast.plot(prediction.iloc[:, -1], linestyle=':', color=[0.85] * 3)
        
            legendSamples.append(Line2D([0], [0], color=color_median, linestyle='--'))
            legendLabels.append('Median prediction')
        else:
            if add_markers:
                ax_forecast.plot(prediction.loc[:, prediction.columns.get_level_values('Type')!='Observed'], linestyle='-', marker='o', markersize=3, color=color_median, zorder=1)
            else:
                ax_forecast.plot(prediction.loc[:, prediction.columns.get_level_values('Type')!='Observed'], linestyle='-', color=color_median, zorder=1)
            legendSamples.append(Line2D([0], [0], color=color_median, linestyle='-'))
            legendLabels.append('Prediction')
            
        observed.plot(ax=ax_forecast, color=color_observed, zorder=prediction.shape[1]//2, alpha=0.75)
          
        #=======================================================================
        # if not not_observed.empty:
        #     not_observed.plot(ax=ax_forecast, color=color_no_observed, zorder=pred.shape[1]//2)
        #=======================================================================
              
        legendSamples.append(Line2D([0], [0], color=color_observed, linestyle='-'))
        legendLabels.append('Observed')

        if best_deterministic is not None:
            best_deterministic.plot(ax=ax_forecast, color='orange', linestyle='-.', zorder=prediction.shape[1]//2, alpha=0.75)
            legendSamples.append(Line2D([0], [0], color='orange', linestyle='-.'))
            legendLabels.append(f'Best deterministic nexc={best_deterministic.loc[best_deterministic.values>observed.values].shape[0]/best_deterministic.shape[0]:.2f}')

        if hype is not None:
            hype.plot(ax=ax_forecast, color='orange', linestyle='-.', zorder=prediction.shape[1]//2, alpha=0.75)
            legendSamples.append(Line2D([0], [0], color='orange', linestyle='-.'))
            legendLabels.append(f'Results from automatic calibration')
        #=======================================================================
        # legendSamples.append(Line2D([0], [0], color=color_no_observed, linestyle='-'))
        # legendLabels.append('Observed NA')
        #=======================================================================
        
        if plot_persistence:
            persistance = observed.copy()
            persistance.index += lead
            ax_forecast.plot(persistance, linestyle='-.', color=color_observed, zorder=prediction.shape[1]//2, alpha=0.25)
            legendSamples.append(Line2D([0], [0], color=color_observed, linestyle='-.', alpha=0.25))
            legendLabels.append('Persistence')

        ax_forecast.legend(legendSamples, legendLabels, fontsize=10, numpoints=1, loc=2, frameon=False, ncol=len(legendLabels)//2)
        ax_forecast.set_ylabel(target_label)
        ax_forecast.grid(True)
    
    #MLP Forecast plot
    if True:
        legendSamples = []
        legendLabels = []
        if len(quantiles)>1:
            color_len = len(quantiles)//2
            color = np.linspace(0.87, 0, color_len)
            colors = np.repeat(np.expand_dims(color, 1), 3, axis=1)
            i0 = 0
            for i0 in np.arange(0, len(quantiles)//2):
                tmp = MLP.iloc[:,[i0, len(quantiles)-1-i0]]
                ax_MLP_forecast.fill_between(MLP.index, tmp.iloc[:, 0], tmp.iloc[:, -1], facecolor=colors[i0,], linewidth=0, alpha=1, zorder=i0)
                legendSamples.append(plt.Rectangle((0, 0), 1, 1, fc=colors[i0,], linewidth=0))
                quantiles_ = tmp.columns.get_level_values('Probability')
                legendLabels.append('{f:03.1f}-{t:03.1f}%'.format(f=quantiles_[0]*100, t=quantiles_[-1]*100))

            center_ = MLP.loc[:, (MLP.columns.get_level_values('Leadtime')[0], MLP.columns.get_level_values('Probability'))]
            center = median_prediction(center_, multiIndex=True)
            ax_MLP_forecast.plot(center, linestyle='--', color=color_median, zorder=i0+1)
            
            if len(quantiles)>1:
                ax_MLP_forecast.plot(MLP.iloc[:, 0], linestyle=':', color=[0.85] * 3)
                ax_MLP_forecast.plot(MLP.iloc[:, -1], linestyle=':', color=[0.85] * 3)
        
        else:
            if add_markers:
                ax_MLP_forecast.plot(MLP.loc[:, MLP.columns.get_level_values('Type')!='Observed'], linestyle='-', marker='o', markersize=3, color=color_median, zorder=1)
            else:
                ax_MLP_forecast.plot(MLP.loc[:, MLP.columns.get_level_values('Type')!='Observed'], linestyle='-', color=color_median, zorder=1)
            legendSamples.append(Line2D([0], [0], color=color_median, linestyle='-'))
            legendLabels.append('Prediction')
            
        observed.plot(ax=ax_MLP_forecast, color=color_observed, zorder=MLP.shape[1]//2, alpha=0.75)
        

        ax_MLP_forecast.set_ylim(ax_forecast.get_ylim())
        ax_MLP_forecast.legend().set_visible(False)
        ax_MLP_forecast.set_ylabel(target_label)
        ax_MLP_forecast.grid(True)
          
    # Precipitation plot
    if True:
        legendSamples = [Line2D([0], [0], color='k')]
        legendLabels = ['Precipitation']
        
        try:
            if 'Leadtime' == observed_.columns.names[1]:
                ax_precipitation.plot(observed_.index, observed_.loc[:, (precipitation, pd.Timedelta('0d'))], color='k')
                ax_precipitation.plot(observed_.index, observed_.loc[:, (precipitation, lead)], color='lightblue', linestyle='--')
                legendSamples += [Line2D([0], [0], color='lightblue')]
                legendLabels += ['Precipitation forecast']
            else:
                raise(Exception())
        except Exception:
            if np.all(basin_areas) != None:
                #Match types of the SUBID in both dataframes
                basin_areas.index = basin_areas.index.astype(int)
                precipitation.columns = precipitation.columns.astype(int)
                legendLabels = ['Ponderated average precipitation across all subbasins']

                ax_precipitation.plot(precipitation.index, (precipitation*basin_areas['AREA']).sum(axis=1)/basin_areas['AREA'].sum(axis=0), color='k')
                ax_precipitation.set_ylim(0, 1.2*precipitation.max().max())
            else:
                # Assuming precipitation is a DataFrame with multiple columns (e.g., subbasins)
                cmap = cm.get_cmap('Blues')
                colors = cmap(np.linspace(0, 1, len(precipitation.columns)))
                for i, col in enumerate(precipitation.columns):
                    ax_precipitation.plot(precipitation.index, precipitation[col], color=colors[i], label=col)
                # Try to keep all labels on a single row, but if there are too many,
                # split into multiple rows with near-equal lengths.
                n_items = len(precipitation.columns)
                max_cols_per_row = 10
                n_rows = int(np.ceil(max(1, n_items) / max_cols_per_row))
                ncol = int(np.ceil(max(1, n_items) / n_rows))
                ax_precipitation.legend(frameon=False, ncol=ncol, fontsize=12, loc='lower center')
        ax_precipitation.set_ylabel('Precipitation [mm]')
        ax_precipitation.grid(True)
        ax_precipitation.set_ylim(0, 1.2*precipitation.max().max()) 
        ax_precipitation.invert_yaxis()

         # Reverse y-axis

    # Temperature plot
    if True:
        legendSamples = [Line2D([0], [0], color='r')]
        legendLabels = ['Temperature']
        
        try:
            if 'Leadtime' == observed_.columns.names[1]:
                ax_temperature.plot(observed_.index, observed_.loc[:, (temperature, pd.Timedelta('0d'))], color='r', zorder=2)
                ax_temperature.plot(observed_.index, observed_.loc[:, (temperature, lead)], color='orange', linestyle='--')

                legendSamples.append(Line2D([0], [0], color='orange', linestyle='--'))
                legendLabels.append('Temperature Forecast')
            else:
                raise(Exception())
        except Exception:
            ax_temperature.plot(temperature.index, temperature.mean(axis=1), color='r', zorder=2)
            ax_temperature.set_ylabel('Temperature [C]')
        ax_temperature.grid(True)
        ax_temperature.yaxis.label.set_color('r')
        ax_temperature.set_ylim(temperature.min().min() - 5, temperature.max().max() + 5)

        if snow:
            legendSamples.append(plt.Rectangle((0, 0), 1, 1, fc=color_snow, ec=color_snow, linewidth=1))
            legendLabels.append('Snow')
            
            ax_snow = ax_temperature.twinx()
            ax_snow.yaxis.label.set_color(color_snow)
            
            try:
                if 'Leadtime' == observed_.columns.names[1]:
                    tmp = observed_.loc[:, (snow, pd.Timedelta('0d'))].applymap(lambda x: 0 if x<0 else x)
            except Exception:
                tmp = observed_.loc[:, snow].applymap(lambda x: 0 if x<0 else x)
            
            ax_snow.plot(tmp.index, tmp, color=color_snow, zorder=-1)
            plt.fill_between(tmp.index, tmp.values.ravel(), color=color_snow, alpha=0.5, zorder=1)
            
            ax_snow.set_ylabel(snow_label)

        ax_temperature.legend(legendSamples, legendLabels, frameon=False)
    
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

    plt.show(block=False)
    
    if filename:
        pickle.dump(fig, open(filename, 'wb'))

    return fig


def median_prediction(predictions, multiIndex=True):
    prob_levels = predictions.columns.get_level_values('Probability').unique()
    lower_prob = prob_levels[prob_levels <= 0.5].max()
    upper_prob = prob_levels[prob_levels >= 0.5].min()
    
    lower_pred = predictions.xs(lower_prob, level='Probability', axis=1)
    upper_pred = predictions.xs(upper_prob, level='Probability', axis=1)
    
    if lower_prob == upper_prob:
        median_pred = lower_pred
    else:
        median_pred = lower_pred  + (upper_pred - lower_pred) * (0.5 - lower_prob) / (upper_prob - lower_prob)
    
    if multiIndex:
        median_pred.columns = pd.MultiIndex.from_tuples([(pd.Timedelta('0D'), 'Simulated')], names=['Leadtime', 'Deterministic'])

    return median_pred

def aggregated_to_df(aggregate, index, bands, leadtime=[pd.Timedelta('0D')]):
    '''Transforms the array of predictions into a dataframe with multiindex columns, necessary for the performance module'''
    columns = pd.MultiIndex.from_product([leadtime, bands], names=['Leadtime', 'Probability'])
    df = pd.DataFrame(aggregate, index=index, columns=columns)
    return df

def calculate_metrics(qp, results, leadtime, zone, climatology=False, model_name='Model', median=None, best_gpu=None, best_HYPE=None):
    '''Calculates a set of metrics and appends them to the results object, this function hevealy relies on the performance module'''
    det_metrics = [ForecastPerformance.NSE,        
                   ForecastPerformance.KGE,
                   ForecastPerformance.relative_bias,
                   ForecastPerformance.MAE,
                   ForecastPerformance.MSE,]
    
    for metric in det_metrics:   
        results.append(Model=model_name, Zone=zone, Metric=metric.__name__, Leadtime=str(leadtime),
                       Value=qp.deterministic(metric, model_name, leadtime=leadtime))
        if median is not None: 
            results.append(Model=model_name + '_Median', Zone=zone, Metric=metric.__name__, Leadtime=str(leadtime),
                           Value=qp.deterministic(metric, model_name + '_Median', leadtime=leadtime))
            
        if best_gpu is not None:
            results.append(Model='Best_GPU', Zone=zone, Metric=metric.__name__, Leadtime=str(leadtime),
                           Value=qp.deterministic(metric, 'Best_GPU', leadtime=leadtime))

        if best_HYPE is not None:
            results.append(Model='Best_HYPE', Zone=zone, Metric=metric.__name__, Leadtime=str(leadtime),
                           Value=qp.deterministic(metric, 'Best_HYPE', leadtime=leadtime))

        if climatology:
            results.append(Model='Climatology', Zone=zone, Metric=metric.__name__, Leadtime=str(leadtime),
                           Value=qp.deterministic(metric, 'Climatology', leadtime=pd.Timedelta('0D')))

    results.append(Model=model_name, Zone=zone, Metric='CRPS', Leadtime=str(leadtime),
                       Value=qp.fairCRPS(model_name, leadtime=leadtime))
    results.append(Model=model_name, Zone=zone, Metric='reliability', Leadtime=str(leadtime),
                       Value=qp.reliability(model_name, leadtimes=leadtime))
    results.append(Model=model_name, Zone=zone, Metric='resolution', Leadtime=str(leadtime),
                       Value=qp.resolution(model_name, leadtimes=leadtime, relative=True))

    if climatology:
        results.append(Model='Climatology', Zone=zone, Metric='CRPS', Leadtime=str(leadtime),
                        Value=qp.fairCRPS('Climatology', leadtime=leadtime))
        results.append(Model='Climatology', Zone=zone, Metric='reliability', Leadtime=str(leadtime),
                        Value=qp.reliability('Climatology', leadtimes=leadtime))
        results.append(Model='Climatology', Zone=zone, Metric='resolution', Leadtime=str(leadtime),
                        Value=qp.resolution('Climatology', leadtimes=leadtime, relative=True))
        
     