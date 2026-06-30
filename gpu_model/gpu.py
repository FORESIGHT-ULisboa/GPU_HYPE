'''
Created on 02/12/2018

@author: zepedro
'''
import time
import pickle
import functools
# import mpld3
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from celery import current_task
from sklearn import preprocessing
from gpu_model.domination import convexSorting
from gpu_model.crowding import phenCrowdingNSGAII
from gpu_model.functions import mpld3Correct, processBands
from scipy.interpolate.interpolate import interp1d
# from django.conf import settings

#HYPE
from conceptual.HYPE import HYPE, predictHYPEdec, metaHYPE
#from linear_model import linearModel

def timer(name):
    '''
    Decorator used to time the execution of the calculations
    '''
    def wrapper0(function):
        @functools.wraps(function)
        def wrapper1(self, *args, **kwargs):
            start=time.time()
            result = function(self, *args, **kwargs)
            self.timing[name] = (time.time()-start)
            return result
        return wrapper1
    return wrapper0

class Display(object):
    '''
    Class used to display results
    '''
    def __init__(self, y, opt, bandBounds, dataType='Observed and simulated', info=''):
        '''
        Declaration
        '''

        self.y = y
        self.opt = opt
        self.bandBounds = bandBounds
        self.bandProbabilities = self.bandBounds[1, :]-self.bandBounds[0, :]
        self.dataType = dataType
        self.performance = []
        self.plot = None
        self.info = info
        if self.info != '':
            self.performance.append(self.info)
            print(self.info)

    def prepare(self, data, verbose=True, save_fig=None):
        '''
        Prepares the display
        '''
        # metrics
        aggregated = GPU._aggregateByBand(data['simulations'], data['fit'], self.bandBounds, self.opt['minModels'], self.opt['forcePositive'])
        if np.any(np.isnan(aggregated)):
            aggregated = processBands(aggregated, self.opt['bands'])
        pValues = GPU._pValues(aggregated, self.y, self.opt['bands'])
        pValues = np.sort(pValues)
        alpha, xi, pi, piRel, sigma = GPU._metrics(pValues, aggregated, self.bandProbabilities)
        
        timing = ['%s: %.5f' % (k, v) for k, v in  data['timing'].items()]
        self.performance.append('% 4u | alpha: %.3f, xi: %.3f, pi: %.3f, pi(rel): %.3f, sigma: %.3f \n       %s' % (data['epoch'], alpha, xi, pi, piRel, sigma, ', '.join(timing)))
        if verbose:
            print(self.performance[-1])
        
        if save_fig:
            self.showPlots(data, aggregated, save=save_fig)
        return self.performance

    def showPlots(self, data, aggregated, save=None):

        if np.any(np.isnan(aggregated)):
            aggregated = processBands(aggregated, self.opt['bands'])
        pValues = GPU._pValues(aggregated, self.y, self.opt['bands'])
        if isinstance(pValues, str):
            pValues = pValues = np.empty(self.y.shape[0])
        pValues = np.sort(pValues)

        if isinstance(self.plot, type(None)):
            self.plot = PlotGPU(self.y, self.opt['bands'])
        self.plot.pareto(data['fit'], data['rejected'])
        self.plot.qq(pValues)
        self.plot.timeseries(aggregated)
        
        fig = plt.gcf()
        fig.set_size_inches([10 , 3])
        axs = fig.get_axes()
        axs[0].set_ylabel('Log10(Error_Metric)')
        axs[0].set_xlabel('Non-exceedance probability')
        axs[0].legend(fontsize=9, frameon=False)

        axs[2].set_ylabel('Discharge [m³/s]')
        axs[2].set_xlabel('Time [days]')

        for ax in axs:
            ax.xaxis.label.set_fontsize(9)
            ax.yaxis.label.set_fontsize(9)
            ax.tick_params(axis='both', labelsize=9)
        line = ax.get_lines()[0]
        line.set(linewidth=2)
        lines = axs[0].get_lines()
        lines[0].set_markersize(5)
        lines[1].set_markersize(5)

        
        plt.tight_layout()

        if save:
            fig.savefig(save, dpi=300)
        return aggregated

    @classmethod
    def save_pickle(self, fig, save):
        with open(save, 'wb') as file:
            pickle.dump(fig, file)

    @classmethod    
    def show_pickle(self, save):
        with open(save, 'rb') as file:
            fig = pickle.load(file)
            plt.show(block=True)

    def close(self):
        plt.close(self.plot.fig)
        
    
class PlotGPU(object):
    '''
    GPU plot class
    '''
    def __init__(self, y, bands, timeSeriesIdxs=None, whatToPlot=['pareto','qq','timeseries'], dataType='Observed and simulated'):
        '''
        Constructor
        '''        
        
        self.y = y
        self.bands = bands
        self.timeSeriesIdxs = timeSeriesIdxs
        if not isinstance(whatToPlot, (list, tuple)):
            whatToPlot = [whatToPlot]
        self.whatToPlot = whatToPlot
        
        # create plot
        self.fig, self.plotAx = self.__create()
        self.hRejected = None
        self.hKept = None
        self.qqKeys = None
        self.uniform = None
        self.hQQ = None
        self.hTimeSeries = None
        self.dataType = dataType
        
    def __create(self):
        '''
        Create the plot
        '''
        cm=1/2.54
        if self.whatToPlot == ['timeseries']:
            fig = plt.figure(figsize=(34*cm, 11*cm))
        else:
            fig = plt.figure(figsize=(0.1+13.4/3*len(self.whatToPlot), 4.2))            
        plotAx = {}
        for i0, k0 in enumerate(self.whatToPlot):
            plotAx[k0] = fig.add_subplot(1, len(self.whatToPlot), i0+1)
    
        return fig, plotAx
    
    def pareto(self, fit, rejected):
        '''
        Plots the gpu space (non-exceedance vs error)
        '''
        
        if isinstance(self.hRejected, type(None)):
            self.hRejected, = self.plotAx['pareto'].plot(rejected[:, 0], rejected[:, 1], 'ok', label='rejected')
            self.hKept, = self.plotAx['pareto'].plot(fit[:, 0], fit[:, 1], 'or', label='kept')
            self.plotAx['pareto'].set_xlim((0, 1))
            self._updateYLim(fit[:,1], self.plotAx['pareto'])
            self.plotAx['pareto'].grid(True)
            self.plotAx['pareto'].legend(fontsize=10, numpoints=1, title='')
            self.plotAx['pareto'].set_xlabel('Exceedance (% of observations over the simulated values)')
            self.plotAx['pareto'].set_ylabel('Log10(error function) (z-scored values)')
        else:
            self.hRejected.set_xdata(rejected[:, 0])
            self.hRejected.set_ydata(rejected[:, 1])
            self.hKept.set_xdata(fit[:, 0])
            self.hKept.set_ydata(fit[:, 1])
            
            tmp = self.plotAx['pareto'].get_ylim()
            if np.min(self.hKept.get_ydata())<tmp[0]:
                self.plotAx['pareto'].set_ylim((np.min(self.hKept.get_ydata()), tmp[1]))
    
    def qq(self, pValues):
        '''
        Draws the predictive QQ plot
        '''
        
        if isinstance(self.hQQ, type(None)):
            if not isinstance(pValues, type(None)):
                self.qqKeys = np.linspace(0, pValues.shape[0]-1, 200, dtype=np.int32)
                self.uniform = np.linspace(0, 1, 200)
                self.hQQ, = self.plotAx['qq'].plot(self.uniform, pValues[self.qqKeys], '-r', label='Observations')
            self.plotAx['qq'].plot([0, 1], [0, 1], '--k')
            self.plotAx['qq'].set_xlim((0,1))
            self.plotAx['qq'].set_ylim((0,1))
            self.plotAx['qq'].grid(True)
            self.plotAx['qq'].set_xlabel('Theoretical quantile of U[0,1]')
            self.plotAx['qq'].set_ylabel('Quantile of the observed p-value')
        else:
            if not isinstance(pValues, type(None)):
                self.hQQ.set_ydata(pValues[self.qqKeys])
        
    def timeseries(self, aggregated):
        '''
        Plots the time series view
        '''
        if isinstance(self.timeSeriesIdxs, type(None)):
            self.timeSeriesIdxs = np.arange(min(aggregated.shape[0], 730))
        
        if isinstance(self.hTimeSeries, type(None)):
            
            self.hTimeSeries = []
            for i0 in range(int(np.floor(len(self.bands)/2))-1, -1, -1):
                tmp = self.plotAx['timeseries'].fill_between(self.timeSeriesIdxs, 
                                                  aggregated[self.timeSeriesIdxs, i0], 
                                                  aggregated[self.timeSeriesIdxs, len(self.bands)-1-i0], 
                                                  facecolor='black', linewidth=0.0, alpha=0.2)
                self.hTimeSeries.append(tmp)
            self.plotAx['timeseries'].plot(self.timeSeriesIdxs, self.y.values[self.timeSeriesIdxs], '-r', alpha=0.5)
            self.plotAx['timeseries'].grid(True)
            self.plotAx['timeseries'].set_xlim(0, len(self.timeSeriesIdxs)-1)
            self.plotAx['timeseries'].set_xlabel('Time')
            self.plotAx['timeseries'].set_ylabel(self.dataType)

        else:
            self.plotAx['timeseries'].clear()
            for i0 in range(int(np.floor(len(self.bands)/2))-1, -1, -1):
                self.plotAx['timeseries'].fill_between(self.timeSeriesIdxs,
                                            aggregated[self.timeSeriesIdxs, i0], 
                                            aggregated[self.timeSeriesIdxs, len(self.bands)-1-i0],
                                            facecolor='black', linewidth=0.0, alpha=0.4)
            self.plotAx['timeseries'].plot(self.timeSeriesIdxs, self.y.values[self.timeSeriesIdxs], '-r')
            self.plotAx['timeseries'].grid(True)
            self.plotAx['timeseries'].set_xlim(0, len(self.timeSeriesIdxs)-1)
            self.plotAx['timeseries'].set_xlabel('Time')
            self.plotAx['timeseries'].set_ylabel(self.dataType)
            
    def _updateYLim(self, data, axis):
        axis.set_ylim((np.floor(np.min(data)-0.25), np.ceil(np.max(data)+1)))
    
    def plot_bands(self):
        return self.fig
    
    def close(self):
        plt.close(self.fig)
        self = None
    
class GPU(object):
    '''
    GPU class
    '''

    def __init__(self,
                 modelObject, 
                 errorObject, 
                 variables, 
                 population=1000, 
                 epochs=400,
                 bands=[0.001, 0.01, 0.025, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 0.975, 0.99, 0.999],
                 bandWidth=0.025, 
                 minModels=1, 
                 displayEach=25,
                 forcePositive=False,
                 transformWeights=True,
                 forceNonExceedance=None,
                 bounds=[-30, 30], #Bounds changed to match the tethys ones, In tethys the bounds are set in the ForecastManagerSingleLeadtime model
                 regularizationCoefficient=None,
                 regularizationL=1,
                 regularizationWeights=None,
                 *args, 
                 **kwargs):
        '''
        Constructor
        '''
       
        # save properties
        self.XScaler = None
        self.yScaler = None
        self.trained = False
        self.minBounds = None
        self.maxBounds = None
        self.timing = {}

        self.modelObject = modelObject
        self.errorObject = errorObject
        self.regularizationCoefficient = regularizationCoefficient  # regularization coefficent used to regularize the regression models
        self.regularizationL = regularizationL                      # type of regularization applied (L1 or L2: 1 or 2)
        self.regularizationWeights = regularizationWeights          # weights of the regression model to be regularized

        self.opt = {}
        self.opt['variables'] = variables                           # number of problem variables
        self.opt['population'] = population                         # number of models to train
        self.opt['epochs'] = epochs                                 # number of training iterations
        self.opt['bands'] = bands                                   # the bands to consider by default
        self.opt['bandWidth'] = bandWidth                           # the width of each band (models to characterize the band are searched for within this distance)
        self.opt['forcePositive'] = forcePositive                   # force positive results
        self.opt['minModels'] = minModels                           # minimum number of models for an estimate to be produced
        self.opt['displayEach'] = displayEach                       # interval between progress displays
        self.opt['transformWeights'] = transformWeights             # transformation of weights (may help convergence)
        self.opt['forceNonExceedance'] = forceNonExceedance         # slope used to force non-exceedance
        self.opt['bounds'] = bounds                                 # (2, #variables) bounds used for each variable. If (2,), the values are tiled to the number of variables
        # process band bounds
        self.bandBounds = self._establishBandBounds(self.opt['bands'])
        self.dataType = 'Unknown'
        
        # process variable bounds
        self.__processVariableBounds()
    
    @timer('Training')
    def fit(self, y, X=None , block=False, epochs=None, info='', **kwargs):
        '''
        Fits models parameters
        y: 1D vector of targets
        X: nD matrix of input patterns
        '''
        
        if epochs==None:
            epochs = self.opt['epochs']
        
        # store the data label
        if isinstance(y, (pd.core.frame.DataFrame, pd.core.series.Series)):
            self.dataType = y.columns[0]
        
        # Details to run with HYPE
        if not isinstance(self.modelObject, HYPE):
            # Remove missing data
            tmp = np.hstack((X, y))
            valid = np.where(np.isfinite(np.sum(tmp, axis=1)))[0]
            if isinstance(X, (pd.core.frame.DataFrame, pd.core.series.Series)):
                X = X.iloc[valid, :]
            else:
                X = X[valid, :]
            if isinstance(X, (pd.core.frame.DataFrame, pd.core.series.Series)):
                y = y.iloc[valid, :]
            else:
                y = y[valid, :]
        
            # Normalize inputs and outputs
            sX = self.__normX(X)
            self.__normY(y)
            
            # Normalize the data within the regression and error model objects
            self.modelObject.setData(sX)
            
        self.errorObject.setTargets(y)

        # Generate the initial population
        if not self.trained:
            self.population = self.__initPopulation()
        
        # Perform the first evaluation
        self.simulations, self.fitted = self.__eval(self.population, first=True) # regression and error models, including regularization
        
        # Initialize display object
        display = Display(y, self.opt, self.bandBounds, info=info)
        
        # Main loop
        for i0 in range(epochs):
            self.timing = {}    # reset the timing dictionary
            
            #i0 in the following expression is for debugging purposes
            self.__iteration()

            if i0 % self.opt['displayEach']==0 or i0==epochs-1:
                # update display
                
                data = {'simulations': self.simulations, 'fit': self.fitted, 'epoch': i0, 'timing': self.timing, 'rejected': self.rejected}
                save_fig = kwargs.get('save', None)
                if  save_fig is not None:
                    save_fig = save_fig.parent / (save_fig.stem + f'_epoch_{i0}.png')
                display.prepare(data , save_fig=save_fig)

        if isinstance(self.modelObject, HYPE):
            self.modelObject.setWeights(self.population)
            # Cleanup the multiprocessing pool
            self.modelObject.delete_Pool() 
        else:
            self.modelObject.setWeights(self.__fWeights(self.population))
        

        result = {'parameters': self.population,'fitness': self.fitted, 'states': display.performance[-1], 'aggregated': self.simulations}
        display.close()
        performance = display.performance
        self.trained = True
        return result, performance
    
    @timer('Prediction')
    @predictHYPEdec
    def predict(self, X, bands=None, postProcess=False, all=False):
        '''
        Predicts model outputs based on new data
        '''
        self.timing={}
        
        # remove missing data
        if isinstance(X, (pd.core.frame.DataFrame, pd.core.series.Series)):
            Xv = X.dropna(inplace=False)
            valid = np.isfinite(np.sum(Xv, axis=1))
            Xv = Xv.loc[valid, :]
        else:
            valid = np.isfinite(np.sum(X, axis=1))
            Xv = X[valid, :]
        
        # Normalize the new data
        XNorm = self.__normX(Xv)
        
        # perform simulations
        simulations = self.modelObject.compute(XNorm)
        
        # rescale simulations to their original space
        simulations = self.yScaler.inverse_transform(simulations)

        if all:
            return simulations
        
        # aggregate results by band, de-normalize, and enforce positiveness (if required)
        aggregated = self.__aggregateByBandWrapper(simulations, bands)
        
        # post-process results (sort bands and fill missing data)
        if postProcess:
            aggregated = self.__postProcessBandsWrapper(aggregated, bands)
        
        # put results into a dataframe
        if isinstance(X, (pd.core.frame.DataFrame, pd.core.series.Series)):
            aggregated = pd.DataFrame(aggregated, index=Xv.index).reindex(X.index)
            if isinstance(bands, type(None)):
                bands = self.opt['bands']
            aggregated.columns = ['%6.1f%%' % (100*b) for b in bands]
        else:
            tmp = np.empty((X.shape[0], aggregated.shape[1]))*np.nan
            tmp[valid, :] = aggregated
            tmp = aggregated

        return aggregated
    
    def prepareDump(self):
        self.modelObject.prepareDump()
        self.errorObject.prepareDump()
        #self.simulations = None
    
    def save(self, fileName):
        '''
        Saves the model to a pickle object
        '''
        self.prepareDump()
        
        with open(fileName, 'wb') as file:
            pickle.dump(self, file)
        
    @classmethod    
    def load(cls, fileName):
        '''
        Loads the model from a pickle object
        '''
        import sys
        # To make the trained versions from the paper being able to run in the repository, the following imports are necessary
        if 'HYPE' not in sys.modules:
            import conceptual.HYPE
            sys.modules['HYPE'] = conceptual.HYPE
        if 'GPU_PSO' not in sys.modules:
            import gpu_model.gpu_pso
            sys.modules['gpu_pso'] = gpu_model.gpu_pso
        if 'GPU_Model' not in sys.modules:
            import gpu_model.gpu  
            sys.modules['gpu'] = gpu_model.gpu
        if 'new_error_model' not in sys.modules:
            import error.error_model
            sys.modules['new_error_model'] = error.error_model

        with open(fileName, 'rb') as file:
            self = pickle.load(file)
        return self
    
    @timer('Generation')
    def __generateWrapper(self, fit, population):
        '''
        Wrapper for the generate (abstract) function
        (allows timing)
        '''
        return self._generate(fit, population)
    
    def _generate(self, fit, population):
        '''
        Abstract method to be implemented in a class implementing a specific optimization strategy
        Should return a matrix of candidades
        '''
        pass
    
    @timer('Selection')
    def __selectWrapper(self, fit, frontLvls, crowdDist, population):
        '''
        Wrapper for the select (abstract) function
        (allows timing)
        '''
        return self._select(fit, frontLvls, crowdDist, population)
    
    def _select(self, fit, frontLvls, crowdDist, population):
        '''
        Abstract method to be implemented in a class implementing a specific optimization strategy
        Should return an array of solutions to be kept
        '''
        pass
    
    @timer('Iteration')
    def __iteration(self):
        '''
        Manages one GPU iteration
        '''
        
        # generate new candidates
        candidates = self.__generateWrapper(self.fitted, self.population)
        
        # evaluate solutions
        newSimulations, newFit = self.__eval(candidates)
        
        # aggregate old and new parameter sets
        jointPopulation = np.vstack((self.population, candidates))
        jointFit = np.vstack((self.fitted, newFit))
        jointSimulations = np.hstack((self.simulations, newSimulations))
        
        jointFit_ = self._penalize_distant_errors(jointFit)

        # domination
        jointFrontLevels = self.__domination(jointFit_)
        
        # crowding
        jointCrowdDist = self.__crowding(jointSimulations, jointFit_, jointFrontLevels)

        # select the parameters to be kept 
        toKeepIdxs = self.__selectWrapper(jointFit_, jointFrontLevels, jointCrowdDist, jointPopulation)
        # update variables
            # update fit
        self.fitted = jointFit[toKeepIdxs,]
            # update population
        self.population = jointPopulation[toKeepIdxs,]
            # update simulations
        self.simulations = jointSimulations[:,toKeepIdxs]
        # update rejected
        tmp = np.ones(jointFit.shape[0], dtype=bool)
        tmp[toKeepIdxs] = False
        self.rejected = jointFit[tmp,]
        # update frontLevels
    
    def __normX(self, X):
        '''
        Normalizes inputs
        '''
            
        if isinstance(self.XScaler, type(None)):
            self.XScaler = preprocessing.StandardScaler().fit(X)

        return self.XScaler.transform(X, copy=True)

    def __normY(self, y):
        '''
        Normalizes outputs
        '''
                    
        if isinstance(self.yScaler, type(None)):
            self.yScaler = preprocessing.StandardScaler().fit(y)
            
        return self.yScaler.transform(y, copy=True)
    
    def _establishBandBounds(self, bands):
        '''
        Calculates the bounds associated with each band
        '''
        bands0 = bands.copy()
        bands0.extend([0,1])
        bands0 = np.unique(bands0)
                
        interval = bands0[1:]-bands0[0:-1]
        interval[1:-1] /= 2
        interval = np.minimum(self.opt['bandWidth'], pd.Series(interval).rolling(window=2).min().dropna()).values
        
        bounds = np.zeros((2, len(bands)))
        bounds[0, :] = bands - interval
        bounds[1, :] = bands + interval

        return bounds
    
    def __processVariableBounds(self):
        '''
        Processes the variable bounds by tiling the bound vector if required
        Also prepares the minBounds and maxBounds variables
        
        If transformWeights is true, the bounds are transformed to the GPU space
        
        HYPE Bounds: When GPU is working with HYPE, the bounds must be specific for every parameter!
        This part now is in an early version, after would be good to refine it!
        '''

        #Bounds for HYPE because the interval was normalize according to each parameter limits
        if isinstance(self.modelObject, HYPE) or isinstance(self.modelObject, metaHYPE):
            #HYPE normalizes the parameters to be between 0 and 1
            if self.modelObject.opts['Normalization']:
                maxBounds = np.ones((self.opt['population'], self.opt['variables']))
                minBounds = np.zeros((self.opt['population'], self.opt['variables']))
                self.minBounds = minBounds
                self.maxBounds = maxBounds
                return None
            else:
                #HYPE doesn't normalize the parameters
                if isinstance(self.minBounds, type(None)):
                    self.minBounds = np.tile(np.array([]), (self.opt['population'], 1))
                    self.maxBounds = np.tile(np.array([]), (self.opt['population'], 1))
                for par in self.modelObject.parNames:
                    if self.modelObject.parametersBounds[par[0]][3] == 'multiply':
                        minBounds = np.ones((self.opt['population'], 1))*self.modelObject.parametersBounds[par[0]][0]
                        maxBounds = np.ones((self.opt['population'], 1))*self.modelObject.parametersBounds[par[0]][1]    
                    else: #substitute
                        minBounds = np.ones((self.opt['population'], par[1]))*self.modelObject.parametersBounds[par[0]][0]
                        maxBounds = np.ones((self.opt['population'], par[1]))*self.modelObject.parametersBounds[par[0]][1]
                    self.minBounds = np.concatenate((self.minBounds, minBounds), axis=1)
                    self.maxBounds = np.concatenate((self.maxBounds, maxBounds), axis=1)
                if isinstance(self.modelObject, metaHYPE):
                    #add Limit to m and b
                    self.minBounds = np.concatenate((self.minBounds, np.ones((self.opt['population'], 1))*self.modelObject.mBounds[0]), axis=1)
                    self.minBounds = np.concatenate((self.minBounds, np.ones((self.opt['population'], 1))*self.modelObject.bBounds[0]), axis=1)
                    self.maxBounds = np.concatenate((self.maxBounds, np.ones((self.opt['population'], 1))*self.modelObject.mBounds[1]), axis=1)
                    self.maxBounds = np.concatenate((self.maxBounds, np.ones((self.opt['population'], 1))*self.modelObject.bBounds[1]), axis=1)
                return None
            
        #Other models (ANNs)
        if isinstance(self.opt['bounds'], (list, tuple)):
            self.opt['bounds'] = np.array(self.opt['bounds'])
    
        if len(self.opt['bounds'].shape)==1:
            self.opt['bounds'] = np.expand_dims(self.opt['bounds'], axis=1)


        if self.opt['bounds'].shape[1] != self.opt['variables']:
            if self.opt['bounds'].shape[1] > 1:
                raise('The number of bounds is not equal to the number of variables.')
            self.opt['bounds'] = np.tile(self.opt['bounds'], (1, self.opt['variables']))
    
        self.minBounds = np.tile(self.opt['bounds'][0,], (self.opt['population'], 1))
        self.maxBounds = np.tile(self.opt['bounds'][1,], (self.opt['population'], 1))
    
        if self.opt['transformWeights']:
            self.minBounds = self.__fWeightsInv(self.minBounds)
            self.maxBounds = self.__fWeightsInv(self.maxBounds)
          
    def __initPopulation(self, forcePositive=True, random=True):
        '''
        Initializes the population
        '''
        #Generate the population for meta_HYPE
        if isinstance(self.modelObject, metaHYPE) and not self.modelObject.opts['init_random']:
                population = np.tile(self.modelObject.parList, (self.opt['population'], 1))
                if self.modelObject.opts['Normalization']:
                    population = self.modelObject._normPar(population)
                    m = np.random.uniform(0, 1, (self.opt['population'], 1))
                    b = np.random.uniform(0, 1, (self.opt['population'], 1))
                    #Debug
                    if np.any(population>1) or np.any(population<0):
                        raise ValueError("Error in the normalization process")
                    
                else:
                    m = np.random.uniform(self.modelObject.mBounds[0], self.modelObject.mBounds[1], (self.opt['population'], 1))
                    b = np.random.uniform(self.modelObject.bBounds[0], self.modelObject.bBounds[1], (self.opt['population'], 1))
                population = np.concatenate((population, m, b), axis=1)
                return population
        
        #Generate the population for HYPE
        elif isinstance(self.modelObject, HYPE) and not self.modelObject.opts['init_random']:
            population = np.random.uniform(0, 1, (self.opt['population']-1, self.opt['variables']))
            if self.modelObject.opts['Normalization']:
                forced_element = self.modelObject._normPar(self.modelObject.parList[0].reshape((1, self.modelObject.parList[0].shape[0])))
                population = np.concatenate((forced_element, population), axis=0)
                if np.any(population>1) or np.any(population<0):
                    raise ValueError("Error in the normalization process")
            else:
                tmp = population*(self.maxBounds[:-1]-self.minBounds[:-1])+self.minBounds[:-1]
                population[np.isfinite(tmp)] = tmp[np.isfinite(tmp)]
                population = np.concatenate((population, self.modelObject.parList[0].reshape((1, self.modelObject.parList[0].shape[0]))), axis=0)
            #Debug
            
            return population
        
        #Restant types of models
        else:
            population = np.random.uniform(0, 1, (self.opt['population'], self.opt['variables']))
            tmp = population*(self.maxBounds-self.minBounds)+self.minBounds
            population[np.isfinite(tmp)] = tmp[np.isfinite(tmp)]
            #Debug
            if isinstance(self.modelObject, HYPE) and self.modelObject.opts['Normalization'] == True:
                if np.any(population>1) or np.any(population<0):
                    raise ValueError("Error in the normalization process")
            return population
        
    def __fWeights(self, pop):
        '''
        Transforms weights from the GPU space to the regression space 
        '''     
        return np.power(pop/4,5)
        
    def __fWeightsInv(self, pop):
        '''
        Transforms weights from the regression space to the GPU space
        '''        
        return 4*np.copysign(np.power(np.abs(pop), 1/5), pop)
    
    def __eval(self, W, first=False):
        '''
        Performs one model evaluation for the whole population and returns a tuple:
        (simulations, fit)
        The log10 of the error is returned as part of the fit
        '''

        # tranform the population into the regression space (if needed)
        if self.opt['transformWeights']:
            W = self.__fWeights(W)
        
        # perform simulations
        simulations = self.__regression(W)
        
        # evaluate fitness
        if first:
            self.errorObject.reshapeData(simulations)
        fit = self.__fit(simulations, W)
        
        return simulations, fit
    
    @timer('Run models')
    def __regression(self, W):
        '''
        Run regression models
        Returns a matrix with the simulations made by each model
        '''
        
        # update model weights
        self.modelObject.setWeights(W)
        # run the models
        simulations = self.modelObject.compute()
        
        #Maintain the simulations in the original space
        #If HYPER is used, the simulations are already in the original space
        if isinstance(self.modelObject, HYPE):
            return simulations
        
        # return simulations to the original space
        else:
            return self.yScaler.inverse_transform(simulations)
        
    @timer('Evaluate fitness')
    def __fit(self, simulations, W):
        '''
        Evaluate fitness associated with each model
        Returns a fit column matrix [non-exceedance, error]
        '''
        
        # perform base calculations
        error, nonexceedance = self.errorObject.compute(simulations)
        
        # apply the logarithm to the error
        error = np.log10(error)
        
        # compute regularization (if needed)
        if self.regularizationCoefficient!=None and self.regularizationCoefficient>0:
            if isinstance(self.regularizationWeights, type(None)):
                self.regularizationWeights = self.modelObject.getWeightsToRegularize()
            reg = self.fRegularization(W, self.regularizationWeights, self.regularizationCoefficient, self.regularizationL)
            error += reg
    
        # aggregate errors and nonexceedance under a fit matrix
        fit = np.vstack((nonexceedance, error)).T
        return fit
    
    @timer('Domination')
    def __domination(self, fit):
        '''
        Performs the domination calculations that are specific to GPU
        Returns a frontLevels (an array containing the level of the front that each parameter set falls into)
        '''
        fronts = convexSorting(fit[:, 0], fit[:, 1])
            
        frontLevels = np.nan*np.empty((fit.shape[0],))
        for i0 in range(len(fronts)):
            frontLevels[fronts[i0]]=i0
            
        return frontLevels
    
    def _penalize_distant_errors(self, fit):
        '''This function intents to give a fit penalization equal 
        to the module of the difference of the distance for each particle'''

        if  self.opt['forceNonExceedance'] == None:
            return fit
        
        tmpMin = fit[np.argmin(fit[:, 1]), 0]   # non-exceedance at the point of minimum error
        tmpDiff = np.abs(fit[:, 0]-tmpMin)      # distance to the best point in terms of non-exceedance
        fit[:, 1] += self.opt['forceNonExceedance']*tmpDiff
        return fit



    @timer('Crowding')
    def __crowding(self, simulations, fit, frontLvls):
        '''
        Phenotype crowding based on correlations with neighboring series [1 - high crowding; 0 - low crowding]
        '''
        # crowding NSGAII
        phenotype = phenCrowdingNSGAII(fit[:,0], fit[:,1], fronts=frontLvls)
        
        with np.errstate(invalid='ignore'):
            phenotype *= np.abs(self.toCustomLogSpace(fit[:,0]))
        
        #Why is necessary to return a matrix?
        return np.vstack((phenotype, phenotype)).T
    
    def _enforceBounds(self, newPopulation):
        '''
        Used by the 'generate' function to guarantee that new parameter sets fall within the desired bounds
        '''
        boundedPopulation = np.max(np.dstack((newPopulation, self.minBounds)), axis=2)
        boundedPopulation = np.min(np.dstack((boundedPopulation, self.maxBounds)), axis=2)
        changed = boundedPopulation != newPopulation
        
        return (boundedPopulation, changed)
    
    @timer('Aggregate by band')
    def __aggregateByBandWrapper(self, simulations, bands=None):
        '''
        Wrapper for the _aggregateByBand function
        (allows timing and default bands)
        '''
        
        # Evaluate band bounds (if required)
        if isinstance(bands, type(None)):
            bandBounds = self.bandBounds
        else:
            bandBounds = self._establishBandBounds(bands)
        
        return self._aggregateByBand(simulations, self.fitted, bandBounds, self.opt['minModels'], self.opt['forcePositive'])
    
    @classmethod
    def _aggregateByBand(cls, simulations, fit, bandBounds, minModels, forcePositive):
        '''
        Aggregates the simulations according to band
        Also de-normalizes and forces values to be positive (if required)
        '''
        # Average results
        aggregated = np.empty((simulations.shape[0], bandBounds.shape[1]))*np.nan
        for i0 in range(bandBounds.shape[1]):
            tmpValidNE = np.where(np.logical_and(fit[:, 0] >= bandBounds[0, i0], fit[:, 0] <= bandBounds[1, i0]))[0] 
            
            if len(tmpValidNE) >= minModels:
                # check for a minimum number of models
                aggregated[:,i0] = np.median(simulations[:, tmpValidNE], axis=1)
        
        # enforce positiveness (if required)
        if forcePositive :
            aggregated = np.maximum(0, aggregated)
        
        return aggregated
    
    @timer('Report status')
    def __report(self, plot=True):
        '''
        Reports the status of the optimization
        '''
        
    @timer('p-values')
    def __pValuesWrapper(self, aggregated, targets, bands=None):
        '''
        Wrapper for the pValues function
        (allows timing and default bands)
        '''
        
        if isinstance(bands, type(None)):
            bands = self.opt['bands']
        
        return self._pValues(aggregated, targets, bands)
       
    @classmethod
    def _pValues(cls, aggregated, targets, bands):
        '''
        Computes the p-values of the targets based on the aggregated simulations
        p-value=1 for target<simulations
        p-value=0 for target>simulations
        
        This operation takes a long time
        '''
        
        # make targets into a numpy array (if needed)
        if isinstance(targets, (pd.core.frame.DataFrame, pd.core.series.Series)):
            targets = targets.values
        
        # Main loop
        bands = cls.toCustomLogSpace(np.array(bands)[::-1])
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
                print('Lack of models to calculate Aggregated')
                pValues[pValues<0] = 0
                pValues[pValues>1] = 1
                return pValues     
        
        # restore p-values to their original space
        pValues = cls.fromCustomLogSpace(pValues)
        pValues[pValues<0] = 0
        pValues[pValues>1] = 1
        
        return 1-pValues
    
    @timer('metrics')
    def __metricsWrapper(self, pValues, aggregated, bandProbabilities=None):
        '''
        Wrapper for the metrics function
        (allows timing and default bandProbabilities)
        '''
        
        if isinstance(bandProbabilities, type(None)):
            bandProbabilities = self.bandBounds[1, :]-self.bandBounds[0, :]
        
        return self._metrics(pValues, aggregated, bandProbabilities)
        
    @classmethod
    def _metrics(cls, pValues, aggregated,  bandProbabilities):
        '''
        Metrics for the probabilistic prediction
        Based on Renard et al. 2010 in Water Resources Research (doi:10.1029/2009WR008328)
        '''
        if  isinstance(pValues, str):
            return (np.nan, np.nan, np.nan, np.nan, np.nan)
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            with np.errstate(divide='ignore'):
                # alpha
                alpha = 1-2*np.mean(np.abs(np.linspace(0, 1, pValues.shape[0]) - pValues))
                
                # xi
                xi = np.zeros_like(pValues)
                xi[np.logical_or(pValues==1, pValues==0)] = 1
                xi = 1-np.mean(xi)
                
                # pi
                if np.any(np.isnan(aggregated)):
                    aggregated[np.where(np.isnan(aggregated))] = 0
                represented = np.sum(bandProbabilities)
                tmp = np.tile(bandProbabilities, (aggregated.shape[0], 1))
                EX = np.sum(aggregated * tmp, axis=1) / represented
                EX2 = np.sum(np.square(aggregated) * tmp, axis=1) / represented
                pi = np.nanmean(1/np.sqrt(EX2-np.square(EX)))
                
                # pi relative
                piRel = np.nanmean(EX/np.sqrt(EX2-np.square(EX)))
                
                # sigma (not in the paper cited above, average standard deviation)
                sigma = np.nanmean(np.sqrt(EX2-np.square(EX)))
        
        return (alpha, xi, pi, piRel, sigma)
    
    @timer('Post-process')
    def __postProcessBandsWrapper(self, aggregated, bands=None):
        '''
        Wrapper for the postProcessBands function
        (allows timing and default bands)
        '''
        
        # Evaluate bands from the base object
        if isinstance(bands, type(None)):
            bands = self.opt['bands']
        
        return self._postProcessBands(aggregated, bands)
        
    @classmethod
    def _postProcessBands(cls, aggregated, bands):
        '''
        Corrects the order of the bands and interpolates missing data 
        '''

        # transform the bands into a log space to allow for a better interpolation of extremes
        bands = cls.toCustomLogSpace(np.array(bands))
          
        # order information for each time step and interpolate missing data
        for i0 in range(aggregated.shape[0]):
            tmpBase = aggregated[i0,:]
            tmp = ~np.isnan(tmpBase)
            tmp2 = ~np.isnan(tmpBase)
            tmpBase[tmp] = np.sort(tmpBase[tmp])
            if np.sum(tmp)>=2:
                tmp = tmpBase[-1]
                for i1 in range(len(tmpBase)-2, 0, -1):
                    if np.isnan(tmp):
                        tmp = tmpBase[i1]
                    else:
                        if ~np.isnan(tmpBase[i1]):
                           
                            if np.round(tmp,6)<=np.round(tmpBase[i1],6):
                                tmpBase[i1] = np.nan
                            else:
                                tmp = tmpBase[i1]
                tmp = ~np.isnan(tmpBase)
                if np.sum(tmp)>1:
                    aggregated[i0,:] = np.interp(bands, bands[tmp], tmpBase[tmp])
                else:
                    aggregated[i0,:] = np.nan
            else:
                aggregated[i0,:] = np.nan
    
        return aggregated
    
    @staticmethod
    def toCustomLogSpace(x):
        '''
        Transforms band probabilities into a custom log space for easier interpolation of the extremes and calculation of quantiles
        '''
        with np.errstate(divide='ignore'):
            y = np.empty_like(x)
            tmp = x<0.5
            y[tmp] = np.log(x[tmp])-np.log(0.5)
            tmp = np.logical_not(tmp)
            y[tmp] = -np.log(1-x[tmp])+np.log(0.5)
        
        return y
    
    @staticmethod
    def fromCustomLogSpace(y):
        '''
        Transforms band probabilities back from a custom log space for easier interpolation of the extremes and calculation of quantiles
        '''

        with np.errstate(divide='ignore'):
            x = np.empty_like(y)
            tmp = y<0
            x[tmp] = np.exp(y[tmp]+np.log(0.5))
            tmp = np.logical_not(tmp)
            x[tmp] = 1-np.exp(-y[tmp]+np.log(0.5))
        
        return x

    @staticmethod
    def fRegularization(w, wToRegularize, regularizationCoefficient, L=1):
        '''
        Regularization function
        '''
        
        if len(w.shape)==1:
            w = np.expand_dims(w, 0)
        if L==1:
            reg = np.sum(np.abs(w[:, np.where(wToRegularize)[0]]), axis=1) # L1
        elif L==2:
            reg = np.sum(np.square(w[:, np.where(wToRegularize)[0]]), axis=1) # L2
        else:
            reg = np.power(np.sum(np.power(np.abs(w[:, np.where(wToRegularize)[0]]),L), axis=1),1/L) # Lp
            
        return regularizationCoefficient*reg
    
    def get_par_distribution(self, save_path=None, show_plots=False):
        '''
        Returns the distribution of the parameters in the current population
        '''
        if not self.trained:
            raise ValueError("Model not trained yet. No parameter distribution available.")
        
        if isinstance(self.modelObject, HYPE):
            return self.modelObject._get_par_distribution(self.population, self.fitted, save_path=save_path, show_plots=show_plots)
        
        return self.population
    
    def get_best_model(self):
        '''
        Returns the best model parameters found during the optimization
        '''
        if not self.trained:
            raise ValueError("Model not trained yet. No best model available.")
        best_idx = np.argmin(self.fitted[:,1])
        self.modelObject.setWeights(weights=self.population[best_idx,:].reshape(1,-1))
        predictions = self.modelObject.compute()
        if isinstance(self.modelObject, HYPE):
            return pd.DataFrame(predictions, index=self.modelObject.records)
        return predictions