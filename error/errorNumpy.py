'''
Created on 18/12/2018

@author: zepedro
'''

import numpy as np

class Error(object):
    
    def __init__(self, targets, errorFunction='MAE'):
        functions = {'MAE': self._mae,
                     'MSE': self._mse,
                     }
        
        self.targets = targets
        #=======================================================================
        # self.scaledTargets = None
        #=======================================================================
        self.shapedTargets = self.targets.values
        self.errorFunction = functions[errorFunction]
        
    def _reshapeTargets(self, simulations):
        self.shapedTargets = np.tile(self.targets, (1, simulations.shape[1]))
        
    def compute(self, simulations):
        if simulations.shape != self.shapedTargets.shape:
            self._reshapeTargets(simulations)
            
        error = self.errorFunction(simulations)
        nonExceedance = self._nonExceedance(simulations)
  
        return (error, nonExceedance)
  
    def _nonExceedance(self, simulations):
        return np.mean(simulations > self.shapedTargets, axis=0)
  
    def _mae(self, simulations):
        return np.mean(np.abs(self.shapedTargets-simulations), axis=0)
        
    def _mse(self, simulations):
        return np.mean(np.square(self.shapedTargets-simulations), axis=0)