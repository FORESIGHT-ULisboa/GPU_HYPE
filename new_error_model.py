import pandas as pd
import numpy as np

class NewErrorModel:
    def __init__(self, targets=None, errorFunction='MAE', non_exceedance_threshold=0.5, logQopt=False):
        self.errorFunction = errorFunction
        self.logQopt = logQopt
        if logQopt and targets is not None:
            self.targets = np.log10(targets+1)
        else:
            self.targets = targets
        
        self.non_exceedance_threshold = non_exceedance_threshold

    def compute(self, simulations):
        '''Compute error between simulations and targets using specified error function
        simulations: matrix [records x population]
        Returns: DataFrame with multi-index and non-exceedance fractions'''
        
        # Create multi-index columns: (simulation_id, metric)
        # columns = []
        # for i in range(simulations.shape[1]):
        #     columns.extend([(i, 'observed'), (i, 'simulated'), (i, 'error')])
        
        # multi_index = pd.MultiIndex.from_tuples(columns)
        # df = pd.DataFrame(index=self.targets.index, columns=multi_index)
        
        nonExceedanceFraction = np.zeros(simulations.shape[1])
        errorMetric = np.zeros(simulations.shape[1])
        if self.logQopt:
            simulations = np.log10(simulations+1)
        df = pd.DataFrame(index=self.targets.index)
        for i in range(simulations.shape[1]):
            # Fill observed and simulated values
            dfi = pd.DataFrame(index=self.targets.index)
            dfi[(i, 'observed')] = self.targets.values.flatten()
            dfi[(i, 'simulated')] = simulations[:, i]
            
            # Calculate error based on selected function
            if self.errorFunction == 'MAE':
                dfi[(i, 'metric_error')] = np.abs(dfi[(i, 'simulated')] - dfi[(i, 'observed')])
                dfi[(i, 'error')] = np.abs(dfi[(i, 'simulated')] - dfi[(i, 'observed')])
            elif self.errorFunction == 'MSE':
                dfi[(i, 'metric_error')] = (dfi[(i, 'simulated')] - dfi[(i, 'observed')]) ** 2
                dfi[(i, 'error')] = np.abs(dfi[(i, 'simulated')] - dfi[(i, 'observed')])

            elif self.errorFunction == 'NSE':
                numerator = (dfi[(i, 'simulated')] - dfi[(i, 'observed')]) ** 2
                denominator = (dfi[(i, 'observed')] - np.mean(dfi[(i, 'observed')])) ** 2
                dfi[(i, 'numerator')] = numerator
                dfi[(i, 'denominator')] = denominator
                dfi[(i, 'metric_error')] = 1 - (np.sum(numerator) / np.sum(denominator)) # if np.sum(denominator) > 0 else np.nan
                dfi[(i, 'error')] = np.abs(dfi[(i, 'simulated')] - dfi[(i, 'observed')])
            else:
                raise ValueError(f"Unsupported error function: {self.errorFunction}")
            
            df = pd.concat([df, dfi], axis=1)
            df.columns = pd.MultiIndex.from_tuples(df.columns, names=['Model', 'Metric'])
            # Calculate non-exceedance: fraction of time steps where simulated > observed 
            # AND error > threshold
            high_error_mask = df[(i, 'error')] > self.non_exceedance_threshold
            simulated_above_observed = df[(i, 'simulated')] > df[(i, 'observed')]
            
            non_exceedance_count = np.sum(high_error_mask & simulated_above_observed)
            nonExceedanceFraction[i] = non_exceedance_count / np.sum(high_error_mask)
            if np.isnan(nonExceedanceFraction[i]):
                nonExceedanceFraction[i] = 0
            errorMetric[i] = df[(i, 'metric_error')].mean()
            if self.errorFunction == 'NSE':
                errorMetric[i] = -errorMetric[i] # Negate NSE to convert to a minimization problem
                #errorMetric[i] = 10**errorMetric[i] # To reverse the Log used by GPU
        #df.to_excel(r'./Test.xlsx')
        # Add non-exceedance as additional info (could be returned separately if needed)
        return errorMetric, nonExceedanceFraction


    def setTargets(self, targets):
        if isinstance(targets, pd.DataFrame):
            if self.logQopt:
                self.targets = np.log10(targets+1)
            else:
                self.targets = targets
        else:
            raise ValueError("Targets should be a pandas DataFrame.")
        
    def reshapeData(self, simulations):
        '''It was used for the openCl model, but here it is not necessary'''
        pass

    def prepareDump(self):
        '''It was used for the openCl model, but here it is not necessary'''
        pass

if __name__ == '__main__':
    # Example usage
    targets = pd.DataFrame(np.array([1, 2, 3]).T, index=pd.date_range(start='2020-01-01', periods=3, freq='D'))
    simulations = np.array([[1.1, 2.1, 2.9], [3.9, 5.2, 6.1]]).T
    
    
    error_model = NewErrorModel(targets, error_function='MAE', non_exceedance_threshold=0.5)
    error_model.set_targets(targets)
    error, non_exceedance, df = error_model.compute(simulations)
    
    print(f"Error: {error}")
    print(f"Non-exceedance rate: {non_exceedance}")
    print(f"Detailed error DataFrame:\n{df}")