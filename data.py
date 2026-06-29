'''
Created on 21 dec. 2018

@author: MAJO
'''

import re
import mpld3
import warnings
import json
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from functions import mpld3Correct, ClickInfo # @UnresolvedImport
from scipy import interpolate
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from dateutil.relativedelta import relativedelta

from django.utils.translation import gettext_lazy as _
# from tethys.functions import get_translation_in, fSortMetorologyColumns

def fSecond(ref, shift):
    return ref + dt.timedelta(seconds=shift)

def fMinute(ref, shift):
    return ref + dt.timedelta(minutes=shift)

def fHour(ref, shift):
    return ref + dt.timedelta(hours=shift)

def fDay(ref, shift):
    return ref + dt.timedelta(days=shift)

def fWeek(ref, shift):
    return ref + dt.timedelta(days=7*shift)

def fMonth(ref, shift):
    #===========================================================================
    # return ref.replace(year=ref.year+np.int(ref.month+shift/12), month=np.mod(ref.month+shift,12))
    #===========================================================================
    return ref + relativedelta(months=shift)

def fYear(ref, shift):
    #===========================================================================
    # return ref.replace(year=ref.year+shift)
    #===========================================================================
    return ref + relativedelta(years=shift)

class Data(object):
    '''
    Prepares and stores data for use within GPU
    '''

    # To perform operations based on the data series period
    __PERIOD__ = {'seconds': fSecond,
                  'minutes': fMinute,
                  'hours': fHour,
                  'days': fDay,
                  'weeks': fWeek,
                  'months': fMonth,
                  'years': fYear,
                  }

    __frequencies__ = {'seconds': 'S',
                       'minutes': 'T',
                       'hours': 'H',
                       'days': 'D',
                       'weeks': 'W',
                       'months': 'M',
                       'years': 'Y'}

    __TICKSTYLE__ = {'seconds': '%Y.%m.%d %H:%M:%S',
                     'minutes': '%Y.%m.%d %H:%M:%S',
                     'hours': '%Y.%m.%d %H:%M',
                     'days': '%Y.%m.%d %H',
                     'weeks': '%Y.%m.%d %H',
                     'months': '%Y.%m.%d',
                     'years': '%Y.%m',
                  }

    # To parse input and output functions
    _CUSTOMFUNCTIONS_ = [{('targets', 'lead(targets)', 'the target series'),
                            ('extra', 'lead(extra', 'additional series used as input (identify as extra[0], extra[1], etc. according to the order in which they appear in the selections below.)'),
                            (']', '])'),
                            },

                           {('known(lead', '('),
                            ('shift(lead', 'shift('),
                            ('fSum', 's._fFrcst_sum_(l,', 's'),
                            ('fFilter', 's._fFrcst_filter_(l,', 'f'),
                            },

                           {('lead(', 's._fLead_(l,', 'shifts the series by the lead time of the forecast or a desired number of time steps: lead(*series*) or lead(*series*, *steps*)'),
                            ('shift(', 's._fShift_(l,', 'shifts the series by the lead time of the forecast minus a desired number of time steps (the lead time of the *series*): shift(*series*, *steps*)'),
                            ('filter(','s._fmFilter_(', 'applies a fading-memory filter to the series (exponential decay). A filter constant k between 0 and 1 can be specified filter(*series*, k)'),
                            ('targets', 't'),
                            ('extra', 'e'),
                            ('meteo', 'm', 'additional information used as input ((identify as meteorology[0], meteorology[1], etc. according to the order in which they appear in the selections below.)'),
                            ('forecast', 's._fForecast_(l,'),
                            #('meteorology', 's._fForecast_(l,m', 'additional information used as input ((identify as meteorology[0], meteorology[1], etc. according to the order in which they appear in the selections below.)'),
                            #('meteo_sum', 's._fForecastSum_(l,m,', 'sum of additional information used as input ((identify as meteorology[0], meteorology[1], etc. according to the order in which they appear in the selections below.): meteo_sum[0]'),
                            ('cycle()', '*s._cycle_(t)', 'informs the model about a periodic cycle: cycle for the forecast period and cycle(*key*) for a different period (allowed *keys* are "seconds", "minutes", "hours", "days", "weeks", "months" and "years"'),
                            ('cycle(seconds)', '*s._cycle_(t, "seconds")'),
                            ('cycle(minutes)', '*s._cycle_(t, "minutes")'),
                            ('cycle(hours)', '*s._cycle_(t, "hours")'),
                            ('cycle(days)', '*s._cycle_(t, "days")'),
                            ('cycle(weeks)', '*s._cycle_(t, "weeks")'),
                            ('cycle(months)', '*s._cycle_(t, "months")'),
                            ('cycle(years)', '*s._cycle_(t, "years")'),
                            ('sum(', 's._fSum_(l,', 'sums the values within with a window the size of the lead time: sum(*series*)'),
                            ('known', '', 'identify a series of values known at the date of the forecast: known(*series*)'),
                            ('log(', 'np.log(', 'applies the log function to the series: log(*series*)'),
                            ('exp(', 'np.exp(', 'applies the exp function to the series: exp(*series*)'),
                            ('rand', 'np.random.randn(s.values.shape[0])', 'creates a random series: rand'),
                            ('leadtime', 'l', 'leadtime constant'),
                            ('extend(', 's._fExtend_(', 'extends the series in case of missing data'),
                            ('linTrend(', 's._fLinTrend_(l,', 'applies a linear trend from the last observed value forward based on 1 or several steps: linTrend(*series*) or linTrend(*series*, *steps*) or linTrend(*series*, *steps*, *max leadtime*)'),
                            ('fSum', 'fSum'),
                            ('fFilter', 'fFilter'),
                           }]


    def __init__(self,
                 targets,               # pandas data frame with the reference series
                 leadtime,              # leadtime
                 inputFunction,         # function to process input data
                 outputFunction,        # function to process output data
                 timeStepSize,          # how many timeStepUnits are there in each step
                 timeStepUnit,          # unit of the time step
                 period,                # how many timeStepUnits are there in a period (integer or datetime format string)
                 reference,             # reference time for the start of a period
                 extra=None,            # pandas data frame with extra series
                 meteorology=None,      # pandas data frame with meteorological forecasts
                 seasons=1,             # number of seasons the period is split in or definition of seasonality
                 seasonDefinition=None, # definition of seasons
                 trainingPeriods=0.4,   # (or e.g. [-10, -9, -8]) fraction of the periods to be used for training or a list of training periods (numeric or in json format)
                 outputLabel=None,       # output label that overrides that of the targets
                 ):
        '''
        Constructor
        '''
        self.targets = targets
        self.leadtime = leadtime
        self.inputFunctionString = inputFunction
        self.outputFunctionString = outputFunction
        self.timeStepSize = timeStepSize
        self.timeStepUnit = timeStepUnit
        self.period = period
        self.reference = reference
        self.extra = extra
        self.meteorology = meteorology
        self.seasons = seasons
        self.seasonDefinition = seasonDefinition
        self.trainingPeriods = trainingPeriods
        self.outputLabel = outputLabel

        self.inputs = None
        self.outputs = None

        if self.outputLabel==None:
            self.outputLabel = self.targets.columns[0]

        # parse output function
        self.outputFunctionStr = self.__parseFunction(self.outputFunctionString)

        # parse input function
        self.inputFunctionStr = self.__parseFunction(self.inputFunctionString)

        # define datetime operation functions
        self.fPeriodJump = self.__PERIOD__[self.period]
        #=======================================================================
        # self.fTimeJump = lambda x, y: self.__PERIOD__[self.timeStepUnit](x, y*self.timeStepSize)
        #=======================================================================

        # synchronize targets and extra
        self.targets, self.extra, self.meteorology = self.__synchronizeInputs(self.targets, self.extra, self.meteorology)

        # break extra data frame into series
        if not isinstance(self.extra, type(None)):
            self.extra = [self.extra.iloc[:, i] for i in range(self.extra.shape[1])]

        # identify seasons
        if isinstance(self.seasonDefinition, type(None)):
            # transform outputs (if required)
            if isinstance(self.outputs, type(None)):
                self.outputs = self.__transformData(self.outputFunctionStr)

            self.seasonDefinition = self.__splitSeasons()
        else:
            self.seasonDefinition = pd.read_json(self.seasonDefinition, convert_axes=False, orient='index', dtype=np.float64)
            self.seasonDefinition.index = np.float64(self.seasonDefinition.index)
            seasons = self.seasonDefinition.shape[1]
            if seasons != self.seasons:
                message = _('The specified number of seasons (%(seasonsNumber)u) was overrided by the specified season definition (%(seasons)u') % {
                    'seasonsNumber': self.seasons, 
                    'seasons': seasons
                }
                warnings.warn(message)
                self.seasons = seasons

        # identify periods
        self.__parseTrainingPeriods(self.trainingPeriods) 

    def fTimeJump(self, x, y):
        '''
        Function to jump between series dates
        '''
        key = get_translation_in(self.timeStepUnit, 'en')
        return self.__PERIOD__[key](x, y*self.timeStepSize)

    def forTraining(self, season=None):
        '''
        Function that prepares data for training (uses data already stored in the object)
        '''

        if season==None and self.seasons != 0:
            raise(_('The season must be specified.'))

        # transform inputs and outputs (if required)
        if isinstance(self.inputs, type(None)):
            self.inputs = self.__transformData(self.inputFunctionStr)
        if isinstance(self.outputs, type(None)):
            self.outputs = self.__transformData(self.outputFunctionStr)

        # convert dates to a the series' period
        x = self.__splitPeriods(self.outputs.index)

        # handle periods
        floored = np.floor(x)
        dataPeriods = np.unique(floored)
        if isinstance(self.trainingPeriods, type(None)):
            periods = dataPeriods
        else:
            if not np.min(np.isin(self.trainingPeriods, dataPeriods)):
                warnings.warn(_('Some of the specified period indexes were not found in the dataset. Have you changed the values, the period, or the reference of the series?'))

        # chose selected periods
        toSelect = np.array(self.trainingPeriods)[np.isin(self.trainingPeriods, dataPeriods)]
        idxs = np.where(np.isin(floored, toSelect))[0]

        inputs = self.inputs.iloc[idxs, :]
        outputs = self.outputs.iloc[idxs, :]

        # remove incomplete patterns
        idxs = np.where(np.isfinite(np.sum(pd.concat((inputs, outputs), axis=1).values, axis=1)))[0]
        inputs = inputs.iloc[idxs, :]
        outputs = outputs.iloc[idxs, :]

        # select season
        seasonWeights = self.__seasonWeights(inputs.index, season)
        inputs = inputs.loc[seasonWeights.loc[:,'To interpolate'].values, :]
        outputs = outputs.loc[seasonWeights.loc[:,'To interpolate'].values, :]
        weights = seasonWeights.loc[seasonWeights.loc[:,'To interpolate'].values, 'Weights']

        return inputs, outputs, weights

    def forPrediction(self, targets, extra, meteorology, season=None):
        '''
        Function that prepares data for prediction (uses new data)
        '''

        # synchronize targets and extra
        targets, extra, meteorology = self.__synchronizeInputs(targets, extra, meteorology)

        # break extra data frame into series
        if not isinstance(extra, type(None)) and not isinstance(extra, list):
            extra = [extra.iloc[:, i] for i in range(extra.shape[1])]

        # transform inputs and output
        inputs = self.__transformData(self.inputFunctionStr, targets=targets, extra=extra, meteorology=meteorology)
        outputs = self.__transformData(self.outputFunctionStr, targets=targets, extra=extra, meteorology=meteorology)
        outputs.columns = [self.outputLabel]

        # select season
        if season!=None:
            seasonWeights = self.__seasonWeights(inputs.index, season)
            inputs = inputs.loc[seasonWeights.loc[:,'To interpolate'].values, :]
            outputs = outputs.loc[seasonWeights.loc[:,'To interpolate'].values, :]
            weights = seasonWeights.loc[seasonWeights.loc[:,'To interpolate'].values, 'Weights']
        else:
            weights = None

        # remove nans
        tmp = inputs.sum(axis=1, skipna=False).notnull().values
        inputs = inputs.loc[tmp, :]
        outputs = outputs.loc[tmp, :]
        if season!=None:
            weights = weights.loc[tmp]

        return inputs, outputs, weights

    def getSeasons(self):
        '''
        Returns an array defining seasonality in the form of a json string
        '''
        return self.seasonDefinition.to_json(orient='index')

    def plotSeasons(self, data=None, mlpd3=False, block=True):
        '''
        Plots the seasons
        data should be a 1D data frame
        '''

        if isinstance(data, type(None)):
            if isinstance(self.outputs, type(None)):
                data = self.__transformData(self.outputFunctionStr)
            else:
                data = self.outputs

        # order by period
        columns = self.__intoPeriods(data)

        # discard incomplete and fill missing data
        columns = self.__fillMissing(columns, discard=0.2, periodWeight=15)

        # compute percentiles
        percentiles = pd.DataFrame({'p=05%': np.nanpercentile(columns, 5, axis=1),
                                    'p=50%': np.nanpercentile(columns, 50, axis=1),
                                    'p=95%': np.nanpercentile(columns, 95, axis=1)},
                                    index=columns.index)

        # plot
        fig = plt.figure(figsize=(12, 4))
        plotAx = fig.add_subplot(1, 1, 1)
        plotAx.fill_between(percentiles.index, percentiles.loc[:, 'p=05%'], percentiles.loc[:, 'p=95%'], alpha=0.3, color='gray', label='p=05%-95%')
        plotAx.plot(percentiles.index.values, percentiles.loc[:, 'p=50%'], 'k', label='p=50%')

        sMin = np.min(columns.values)
        sMax = np.max(columns.values)
        cmap = plt.cm.Blues_r(np.arange(self.seasonDefinition.shape[1]+2)/(self.seasonDefinition.shape[1]+1))   # @UndefinedVariable
        for i0 in range(self.seasonDefinition.shape[1]):
            plotAx.fill_between(self.seasonDefinition.index, sMin, sMin + self.seasonDefinition.iloc[:, i0] * (sMax - sMin), alpha=0.3, label='Season %u' % i0, color=cmap[i0,:])

        plt.xlabel('Time (period fraction)')
        plt.ylabel(self.targets.columns[0])
        plt.legend()
        plt.tight_layout()

        # prepare mpld3 plot
        plotDict = mpld3.fig_to_dict(fig)
        plotDict = mpld3Correct(plotDict)
        if mlpd3:
            plt.close(fig)
        else:
            plt.show(block=block)

        return json.dumps(plotDict)

    def getTrainingPeriods(self):
        '''
        Returns an array of periods selected for training in json format
        '''

        return json.dumps(self.trainingPeriods)

    def plotPeriods(self, mlpd3=False, block=True):
        '''
        Generate a violin plot
        '''

        # transform outputs (if required)
        if isinstance(self.outputs, type(None)):
            self.outputs = self.__transformData(self.outputFunctionStr)

        # convert dates to a the series' period
        x = self.__splitPeriods(self.outputs.index)

        # handle periods
        floored = np.floor(x)
        dataPeriods = np.unique(floored)
        #=======================================================================
        # if not np.min(np.isin(self.trainingPeriods, dataPeriods)):
        #     warnings.warn('Some of the specified period indexes were not found in the dataset. Have you changed the values, the period, or the reference of the series?')
        #=======================================================================

        # create groups and attach a datetime to each period
        groups = []
        dates = []
        periods = []
        for g0 in dataPeriods:
            tmp = self.outputs.iloc[floored==g0, :].dropna().values
            if tmp.shape[0]>1:
                groups.append(tmp.flatten())
                tmp = [self.fPeriodJump(self.reference, int(g0+1)), self.fPeriodJump(self.reference, int(g0))]
                dates.append(pd.to_datetime(tmp).values.astype('timedelta64[ns]').mean().astype('datetime64[ns]'))
                periods.append(g0)
        dates = pd.to_datetime(dates)

        # scale with with the number of points
        widths = np.array([g0.shape[0] for g0 in groups])
        widths = widths/widths.max()

        # create plot
        fig = plt.figure(figsize=(12, 3))
        plotAx = fig.add_subplot(1, 1, 1)
        violinParts = plotAx.violinplot(groups, periods, points=40, widths=widths, showmeans=False, showextrema=False, showmedians=False)

        # add labels
        plotAx.set_ylabel(self.targets.columns[0])
        plotAx.set_xlabel(_('Periods (the "reference date for the period" marks the 0)'))

        # replace x labels
        #=======================================================================
        # plotAx.set_xticks(periods[::2])
        # plotAx.set_xticklabels(dates[::2].strftime(self.__TICKSTYLE__[self.period]))
        #=======================================================================
        plt.tight_layout()

        # change colors
        styles = {True: {'fill': '#417690'},
                  False: {'fill': '#cccccc'}}
        for i0, g0 in enumerate(periods):
            if np.isin(g0, self.trainingPeriods):
                # Training
                violinParts['bodies'][i0].set_facecolor(styles[True]['fill'])
            else:
                # Others
                violinParts['bodies'][i0].set_facecolor(styles[False]['fill'])
            violinParts['bodies'][i0].set_alpha(1)

        # connect to a plugin function
        for i0, p0 in enumerate(violinParts['bodies']):
            marker = periods[i0]
            usedForTraining = np.isin(periods[i0], self.trainingPeriods)
            mpld3.plugins.connect(fig, ClickInfo(p0, usedForTraining, styles, marker))

        # prepare mpld3 plot
        plotDict = mpld3.fig_to_dict(fig)
        plotDict = mpld3Correct(plotDict)
        if mlpd3:
            plt.close(fig)
        else:
            plt.show(block=block)

        return json.dumps(plotDict)

    @classmethod
    def functionDescriptions(cls):
        '''
        Returns a list with the description of the allowed functions for data processing
        '''

        pattern = re.compile('[\W_]+')

        descriptions = []
        for f0 in cls._CUSTOMFUNCTIONS_:
            for f1 in f0:
                if len(f1)==3:
                    descriptions.append('%s:   %s.' % (pattern.sub('', f1[0]), f1[2]))

        return descriptions

    def __parseTrainingPeriods(self, trainingPeriods):
        '''
        Parses the training periods (from a fraction, a numeric array, or a json array
        '''

        # a fraction was passed
        if isinstance(trainingPeriods, (float, int)):
            if trainingPeriods<0:
                trainingPeriods = 0
                warnings.warn(_('Training period too small. Set to 0 (one single period).'))
            elif trainingPeriods>1:
                trainingPeriods = 1
                warnings.warn(_('Training period too large. Set to 1 (all available periods).'))

            # transform outputs (if required)
            if isinstance(self.outputs, type(None)):
                self.outputs = self.__transformData(self.outputFunctionStr)

            # convert dates to a the series' period
            x = self.__splitPeriods(self.outputs.index)

            # handle periods
            floored = np.floor(x)
            dataPeriods = np.unique(floored)
                # check periods with data
            sizes = np.array([self.outputs[floored==g].dropna().shape[0] for g in dataPeriods])
            dataPeriods = dataPeriods[sizes!=0]

                # compute the required number of periods
            numberOfPeriods = np.int(np.round(dataPeriods.shape[0]*trainingPeriods))
            if numberOfPeriods<1:
                numberOfPeriods = 1
                # select the periods randomly
            trainingPeriods = np.random.permutation(dataPeriods)[:numberOfPeriods].tolist()

        # a json was passed
        if isinstance(trainingPeriods, (str)):
            trainingPeriods = json.loads(trainingPeriods)

        # a numpy array was passed
        if isinstance(trainingPeriods, np.ndarray):
            trainingPeriods.tolist()

        # store information
        trainingPeriods.sort()
        self.trainingPeriods = trainingPeriods

    def __synchronizeInputs(self, targets, extra, meteorology):
        '''
        Function that ensures continuous and matching indexes among targets and extra data frames
        '''

        # get start and end dates
        start = min(targets.index)
        if not isinstance(extra, type(None)) and not isinstance(extra, list):
            start = min((start, min(extra.index)))
        end = max(targets.index)
        if not isinstance(extra, type(None)) and not isinstance(extra, list):
            end = max((end, max(extra.index)))

        # increase end according to the leadtime
        end = self.fTimeJump(end, self.leadtime)

        # prepare a unified and continuous index
        dates = [start]
        i0=1
        while dates[-1]<end:
            dates.append(self.fTimeJump(start, i0))
            i0 += 1
        dates = pd.DatetimeIndex(dates)

        # re-define targets and extra
        targets = targets.reindex(index=dates)
        if not isinstance(extra, type(None)) and not isinstance(extra, list):
            extra = extra.reindex(index=dates)

        # handle meteorology
        if not isinstance(meteorology, type(None)):
            tmp = meteorology.droplevel(0, axis=0).unstack()
            tmp = tmp.reindex(index=dates)
            meteorology = tmp.stack(dropna=False)

        return targets, extra, meteorology 

    def __splitPeriods(self, datetimes, period=None):
        '''
        Function that returns an array where each period is equivalent to a unit
        period can be 'seconds', 'minutes', 'hours', 'days', 'weeks', 'months' and 'years'
        '''

        if period!=None:
            fJump = self.__PERIOD__[period]
        else:
            period = self.period
            fJump = self.fPeriodJump

        reference = self.reference
        #key = get_translation_in(self.timeStepUnit, 'en') From tethys!
        tmp = pd.date_range(start=min((fJump(reference, -2), fJump(datetimes[0], -2))), end=max((fJump(reference, 2), fJump(datetimes[0], 2))), freq='%u%s' % (self.timeStepSize, self.__frequencies__[key]))
        pointAfterReference = tmp[tmp>=reference][0]
        
        tmp = pd.date_range(start=min((fJump(self.reference, -2), fJump(datetimes[0], -2))), end=max((fJump(self.reference, 2), fJump(datetimes[-1], 2))), freq=self.__frequencies__[period])
        tmp = pd.DataFrame({'periods': np.arange(tmp.shape[0])}, index=tmp)
        periodAfterReference = tmp.index[tmp.index>=reference][0]

        offset = periodAfterReference - pointAfterReference
        tmp.index = tmp.index - offset

        #=======================================================================
        # if period == 'years' and reference >= dt.datetime(reference.year, 3, 1):
        #     offset = periodAfterReference - pointAfterReference
        #     tmp.index = tmp.index - offset
        # else:
        #     offset = afterReference - tmp.index[-2]
        #     tmp.index = tmp.index + offset
        #=======================================================================

        x = [d.timestamp() for d in tmp.index]
        y = tmp.values.ravel()
        ref = np.interp([reference.timestamp()], x, y)
        periods = np.interp([d.timestamp() for d in datetimes], x, y) - ref

        return np.round(periods, 6)

    def __intoPeriods(self, data):
        '''
        Function that orders data by columns, with each column corresponding to a separate period
        data should be a 1D data frame
        '''

        # convert dates to a the series' period
        x = self.__splitPeriods(data.index)

        # separate each period into columns
        idxs0 = np.linspace(0, 1, 1000)
        idxs = [i + idxs0 for i in np.unique(np.floor(x))]
        idxs = np.unique(np.hstack(idxs))
        idxs = idxs[:-1]
        tmp = np.floor(idxs)
        columns = []
        for y0 in np.unique(tmp):
            columns.append(np.interp(idxs[tmp==y0], x, self.targets.values.ravel(), left=np.nan, right=np.nan))
        columns = np.vstack(columns)

        # transform into a data frame
        columns = pd.DataFrame({i: columns[i, :] for i in range(columns.shape[0])}, index=idxs0[:-1])

        return columns

    def __fillMissing(self, columns, discard=0.2, periodWeight=15):
        '''
        Fills missing data in a period-ordered matrix
        '''

        # discard columns with too many gaps (>=20%)
        valid = np.mean(np.isnan(columns.values), axis=0) < discard
        columns = columns.iloc[:, valid]

        # fill missing values in valid columns
        means = np.nanmean(columns, axis=1)
            # 2d interpolation
        tmp = np.ma.masked_invalid(columns.values.T)
                # the distance from 1 period is equivalent to the distance between periodWeight time steps
        xx, yy = np.meshgrid(1 * np.arange(tmp.shape[1]), periodWeight * np.arange(tmp.shape[0]))
        x1 = xx[~tmp.mask]
        y1 = yy[~tmp.mask]
        z1 = tmp[~tmp.mask].ravel()
        columns = pd.DataFrame(interpolate.griddata((x1, y1), z1, (xx, yy), method='linear').T, index=columns.index, columns=columns.columns)
            # mean of the period
        tmp = np.where(np.isnan(np.sum(columns.values, axis=1)))[0]
        for i0 in tmp:
            columns.iloc[i0, np.isnan(columns.iloc[i0,:]).values] = means[i0]

        return columns

    def __splitSeasons(self, gamma=0.001, timeCoef=0.4, derivativeWeigth=.5):
        '''
        Function that identifies the seasons from the transformed outputs using actual values and their derivative
        '''

        # order by period
        #######CHECK THIS!!!!!!!!!!!!!!!!!!!!
        #=======================================================================
        # self.targets = self.targets.ffill()
        #=======================================================================
        
        values = self.__intoPeriods(self.targets)
        idxs = values.index.values

        # discard incomplete and fill missing data
        values = self.__fillMissing(values, discard=0.2, periodWeight=15)

        # compute derivatives
        try:
            derivatives = values.rolling(window=2).apply(lambda x: x.iloc[1]-x.iloc[0], raw=True)
        except Exception:
            derivatives = values.rolling(window=2).apply(lambda x: x[1]-x[0], raw=True)
        derivatives.iloc[0, :] = (derivatives.iloc[1, :].values+derivatives.iloc[1, :].values)/2

        # normalize and scale
        values = (values-np.min(values.values))/(np.max(values.values)-np.min(values.values))
        derivatives = derivativeWeigth * (derivatives-np.min(derivatives.values))/(np.max(derivatives.values)-np.min(derivatives.values))

        # join values and derivatives
        columns = pd.concat((values, derivatives), axis=1)

        # apply PCA to reduce the number of columns (min of 80% of the variance kept)
        pca = PCA(n_components=min((5, columns.shape[1])))
        pca.fit(columns.values.T)
        tmp = len(np.where(np.cumsum(pca.explained_variance_ratio_)<=0.8)[0]) + 1
        pcaData = pca.components_[:tmp,:]*np.transpose(np.tile(pca.explained_variance_ratio_[:tmp]/pca.explained_variance_ratio_[0],(columns.shape[0],1)))

        # apply clustering
        tmp = np.expand_dims(idxs, axis=1)
        toCluster = np.hstack((np.cos(tmp*2*np.pi)*timeCoef, np.sin(tmp*2*np.pi)*timeCoef, pcaData.T))
        kMeans = KMeans(n_clusters=self.seasons)
        kMeans.fit(toCluster)
        tmpSeasons = kMeans.predict(toCluster)

        # smooth clusters
        tmpIdxs = np.hstack((idxs, 1+idxs, 2+idxs))
        tmpClusters = np.hstack((tmpSeasons, tmpSeasons, tmpSeasons))
        tmpSmoothSeasons = np.zeros((self.seasons, len(idxs)))
        for i0 in range(self.seasons):
            centers = tmpIdxs[np.where(tmpClusters==i0)[0]]
            for c0 in centers:
                tmp = c0-tmpIdxs
                rbf = np.exp(-1/gamma*np.square(tmp));
                tmpSmoothSeasons[i0,:] = tmpSmoothSeasons[i0, :]+rbf[len(idxs):2*len(idxs)]
        tmp = 1/np.sum(tmpSmoothSeasons, axis=0)
        tmpSmoothSeasons = tmpSmoothSeasons*np.tile(tmp, (self.seasons, 1))
        tmpSmoothSeasons = np.hstack((tmpSmoothSeasons, tmpSmoothSeasons[:, :1]))
        tmpSmoothTimes = np.hstack((idxs, 1))

        # return arrays for interpolation
        tmpSmoothSeasons[tmpSmoothSeasons<0.001] = 0
        tmpSmoothSeasons[tmpSmoothSeasons>0.999] = 1
        return pd.DataFrame({i: tmpSmoothSeasons[i, :] for i in range(tmpSmoothSeasons.shape[0])}, index=tmpSmoothTimes)

    def __seasonWeights(self, datetimes, season):
        '''
        Returns a data frame with information about which patters to use and their seasonal weights
        '''

        # retrieved period fractions
        try:
            x = self.__splitPeriods(datetimes)
        except Exception as ex:
            raise(ex)
        period = x-np.floor(x)

        # compute weights
        weights = np.empty((x.shape[0], self.seasonDefinition.shape[1]))*np.nan
        for i0 in range(self.seasonDefinition.shape[1]):
            toInterpolate = self.seasonDefinition.iloc[:, i0]
            weights[:, i0] = np.interp(period, toInterpolate.index, toInterpolate)

        # guarantee that weights sum to 1
        weights[weights<0.001] = 0
        weights = weights / np.tile(np.sum(weights, axis=1), (weights.shape[1], 1)).T

        # prepare output
        toInterpolate = weights[:, season]!=0
        seasonWeights = pd.DataFrame({'Weights': weights[:, season], 'To interpolate': toInterpolate}, index=datetimes)

        return seasonWeights

    @classmethod
    def parseFunction(cls, parseFunctionString):
        '''
        Verifies the validity of a given input or target function
        '''

        # Remove non-recognized characters
        parseFunctionString = parseFunctionString.replace(' ','')
        allowed = [s0[0] for s0 in cls._CUSTOMFUNCTIONS_[2]]
        toKeep = []
        i0 = 0
        for s0 in allowed:
            while s0 in parseFunctionString:
                parseFunctionString = parseFunctionString.replace(s0, '#' + str(i0) + '#', 1)
                toKeep.append(s0)
                i0 += 1
        if len(re.findall('[a-zA-Z]', parseFunctionString))>0:
            raise Exception(str(_('Some keywords in the data parsing function are not allowed. Code halted. The allowed keywords are: ')) + str(allowed))
        parseFunctionString = re.sub('[a-zA-Z]','', parseFunctionString)

        for i0, s0 in enumerate(toKeep):
            parseFunctionString = parseFunctionString.replace('#' + str(i0) + '#', s0, 1)

        # Perform the substitutions
        for s0 in cls._CUSTOMFUNCTIONS_:
            for s1 in s0:
                parseFunctionString = parseFunctionString.replace(s1[0], s1[1])

        return 'lambda s, t, e, m, l: [' + parseFunctionString + ']'


    def __parseFunction(self, parseFunctionString):
        '''
        Function used to parse input and output functions
        '''

        return self.parseFunction(parseFunctionString)

        #=======================================================================
        # # Remove non-recognized characters
        # parseFunctionString = parseFunctionString.replace(' ','')
        # allowed = [s0[0] for s0 in self._CUSTOMFUNCTIONS_[2]]
        # toKeep = []
        # i0 = 0
        # for s0 in allowed:
        #     while s0 in parseFunctionString:
        #         parseFunctionString = parseFunctionString.replace(s0, '#' + str(i0) + '#', 1)
        #         toKeep.append(s0)
        #         i0 += 1
        # if len(re.findall('[a-zA-Z]', parseFunctionString))>0:
        #     raise Exception('Some keywords in the data parsing function are not allowed. Code halted. The allowed keywords are: ' + str(allowed))
        # parseFunctionString = re.sub('[a-zA-Z]','', parseFunctionString)
        #
        # for i0, s0 in enumerate(toKeep):
        #     parseFunctionString = parseFunctionString.replace('#' + str(i0) + '#', s0, 1)
        #
        # # Perform the substitutions
        # for s0 in self._CUSTOMFUNCTIONS_:
        #     for s1 in s0:
        #         parseFunctionString = parseFunctionString.replace(s1[0], s1[1])
        #
        # return 'lambda s, t, e, m, l: [' + parseFunctionString + ']'
        #=======================================================================

    def __transformData(self, functionStr, targets=None, extra=None, meteorology=None):
        '''
        Function that transforms the data using parsed function strings
        '''

        if isinstance(targets, type(None)):
            targets = self.targets

        if isinstance(extra, type(None)):
            extra = self.extra
            
        if isinstance(meteorology, type(None)):
            meteorology = self.meteorology

        if not isinstance(meteorology, list):
            meteorology = [meteorology]

        function = eval(functionStr)

        try:
            data = function(self, targets, extra, meteorology, self.leadtime)
        except Exception as ex:
            raise(ex)
        
        tmp = {}
        for i0, c0 in enumerate(data):
            if isinstance(c0, (pd.core.frame.DataFrame, pd.core.series.Series)):
                c0 = c0.values
            tmp['%u' % i0] = c0.ravel()
        dataframe = pd.DataFrame(tmp, index=targets.index)

        return dataframe

    def _fmFilter_(self, x, beta=0.79352):
        '''
        Applies a fading-memory filter to the array
        '''

        tmp = np.array(x)
        for i0 in range(1,x.shape[0]):
            if not np.isinf(tmp[i0]):
                tmp0 = tmp[i0-1]+(1-beta)*(tmp[i0]-tmp[i0-1])
                if not np.isnan(tmp0):
                    tmp[i0] = tmp0
            else:
                tmp[i0] = tmp0

        return tmp

    def _fLead_(self, lead, x, override=None):
        '''
        Displaces a vector according to the leadtime or an override integer
        '''

        # check for the override
        lead = int(lead)
        if override != None:
            lead = int(override)

        # main operation
        tmp = np.empty_like(x)*np.nan
        if lead==0:
            return x
        if lead>0:
            tmp[lead:]=x[:-lead]
        else:
            tmp[:lead]=x[-lead:]

        return tmp

    def _fLinTrend_(self, lead, x, steps=None, maxLeadtime=None):
        '''
        Estimates a linear trend
        '''
        
        # check for the steps
        if isinstance(steps, type(None)):
            steps = 1
        
        if isinstance(maxLeadtime, type(None)):
            maxLeadtime = steps*6
        
        # main operation
        if lead<=maxLeadtime:
            tmp = np.empty_like(x)*np.nan
            tmp[steps:] = (x[steps:]-x[0:-steps])/steps*lead
            tmp += x
        else:
            tmp = np.ones_like(x)
        
        return tmp

    def _fExtend_(self, x):
        '''
        Extends the vector in case of missing data
        '''
        
        x = x.ravel()
        
        mask = np.isnan(x)
        idx = np.where(~mask,np.arange(mask.shape[0]),0)
        np.maximum.accumulate(idx,axis=0, out=idx)
        x = x[idx]
        
        return x

    def _fShift_(self, lead, x, leadtime):
        '''
        Displaces a vector according to the leadtime of "this" forecast and the leadtime of another forecast, used as input
        '''
 
        tmp = self._fLead_(lead, x, override=lead-leadtime)

        return tmp

    def _fForecast_(self, lead, x):
        '''
        Returns the most recent forecasts associated with the series for the chosen time step and leadtime
        x index level 0 are observation dates
        '''
        
        leadtime = {str(self.timeStepUnit): int(self.timeStepSize)*lead}
        leadtime = pd.DateOffset(**leadtime)
        
        # dummy date for indexing
        tmp0 = x.index.get_level_values(0)[0] + leadtime
        valid_bool = np.isfinite(x.values).ravel()
        tmp1 = x.index.get_level_values(0)[0] + x.index.get_level_values(1)[valid_bool]
        valid = tmp1>=tmp0
        
        dates = x.index.get_level_values(0).unique()
        data = x.loc[valid_bool].loc[valid].unstack(-1).bfill(axis=1).iloc[:, 0]
        data = data.reindex(index=dates)
        
        return data.values 

    def _fFrcst_sum_(self, lead, x, sum_steps):
        '''
        Returns the most recent forecasts associated with the series for the chosen time step and leadtime
        x index level 0 are observation dates
        '''
        
        raise('To be fixed: see _fFrcst_filter_')
        
        leadtime = {str(self.timeStepUnit): int(self.timeStepSize)*lead}
        leadtime = pd.DateOffset(**leadtime)
        
        valid_bool = np.isfinite(x.values).ravel()
        valid_records = x.loc[valid_bool]
        
        sums = []
        for i0 in range(sum_steps):
            sums.append(pd.DateOffset(**{str(self.timeStepUnit): int(self.timeStepSize)*i0}))
        
        values = []
        for s0 in sums:
            tmp0 = valid_records.index.get_level_values(0)[0] + leadtime - s0
            tmp1 = valid_records.index.get_level_values(0)[0] + valid_records.index.get_level_values(1)
            valid = tmp1>=tmp0
            tmp = valid_records.loc[valid].unstack(-1).sort_index(axis=1, ascending=False).bfill(axis=1).iloc[:, 0]
            tmp.index = tmp.index + s0
            values.append(tmp)
        values = pd.concat(values, axis=1)
        values = values.sum(axis=1, skipna=False)
            
        dates = x.index.get_level_values(0).unique()
        data = values.reindex(index=dates)
        
        return data.values 

    def _fFrcst_filter_(self, lead, x, k=0.79, sum_steps=12):
        '''
        Returns the most recent forecasts associated with the series for the chosen time step and leadtime
        x index level 0 are observation dates
        '''
        
        leadtime = {str(self.timeStepUnit): int(self.timeStepSize)*lead}
        leadtime = pd.DateOffset(**leadtime)
        
        valid_bool = np.isfinite(x.values).ravel()
        valid_records = x.loc[valid_bool].copy()
        
        '''
        a = valid_records.unstack().sort_index(axis=1, ascending=False)
        a.iloc[:] = np.arange(365*129)
        a.iloc[:] = np.arange(365*129).reshape((365,129))
        valid_records = a.stack()
        
        values.columns = list(range(values.shape[1]))
        '''
        
        sums = []
        for i0 in range(sum_steps):
            sums.append(pd.DateOffset(**{str(self.timeStepUnit): int(self.timeStepSize)*i0}))
        
        values = []
        for s0 in sums[::-1]:
            tmp0 = valid_records.index.get_level_values(0)[0] - leadtime # reference production date
            tmp1 = valid_records.index.get_level_values(0)[0] - valid_records.index.get_level_values(1) # production date of the record
            valid = tmp1<=tmp0            
            tmp = fSortMetorologyColumns(valid_records.loc[valid].unstack(-1)).bfill(axis=1).iloc[:, 0]
            tmp.index = tmp.index + s0
            values.append(tmp)
        values = pd.concat(values, axis=1)
        
        gain = 1-k
        values = values.bfill(axis=1)
        filtered = values.iloc[:, [0]]
        for c0 in range(1, values.shape[1]):
            filtered.loc[:, filtered.columns[0]] += gain * (values.iloc[:, c0] - filtered.iloc[:, 0])
            
        dates = x.index.get_level_values(0).unique()
        data = filtered.reindex(index=dates)
        
        return data.values 

    def _fSum_(self, leadtime, x, override=None):
        '''
        Sums a vector up to the leadtime or an override integer
        '''

        # check for the override
        leadtime = int(leadtime) 
        if override != None:
            leadtime = int(override)

        # main cycle
        tmpX = x.values.ravel()
        if leadtime==0:
            return tmpX
        nanIdx = np.where(np.isnan(tmpX))[0]
        tmpX[nanIdx] = 0
        if leadtime>0:
            tmpX = np.cumsum(tmpX)
            tmp = np.empty_like(tmpX)*np.nan
            tmp[leadtime:]=tmpX[leadtime:]-tmpX[:-leadtime]
        elif leadtime<0:
            tmpX = np.cumsum(tmpX)
            tmp = np.empty_like(tmpX)*np.nan
            tmp[:-leadtime]=tmpX[:-leadtime]-tmpX[leadtime:]
        toErase = []
        if leadtime>0:
            for i0 in range(nanIdx.shape[0]):
                toErase.append(nanIdx[i0]+range(int(leadtime)+1))
        else:
            for i0 in range(nanIdx.shape[0]):
                toErase.append(nanIdx[i0]+range(0,int(leadtime)-1,-1))
        if len(toErase)>0:
            toErase = np.unique(np.hstack(toErase))
            toErase = toErase[toErase>=0]
            toErase = toErase[toErase<tmpX.shape[0]]
            tmp[toErase] = np.NaN

        return tmp

    def _cycle_(self, targets, period=None):
        '''
        Returns sine and cosine signals representing the periodic cycle associated with the targets
        the targets should come in the form of a pandas data frame with a datetime index
        '''

        # convert dates to a the series' period
        x = self.__splitPeriods(targets.index, period)

        return np.cos(x*2*np.pi), np.sin(x*2*np.pi)
