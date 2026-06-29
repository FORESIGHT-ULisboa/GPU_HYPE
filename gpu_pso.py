# coding: utf-8
'''
Created on 21/09/2015

@author: Jose Pedro Matos
'''
import numpy as np
import time
from gpu import GPU
from error.errorOpenCL import Error
from regression.annOpenCL import ANN

class GPU_PSO(GPU):
    '''
    Particle Swarm Optimization implementation of GPU
    (other implementations are possible by changing the generate and select functions)
    '''
    def __init__(self, 
                 modelObject=ANN(), 
                 errorObject=Error(), 
                 variables = 41, 
                 inertia=0.5, 
                 c1=0.1, 
                 c2=0.1, 
                 c3=0.001, 
                 pBins=10, 
                 partial=1,
                 *args, 
                 **kwargs):
        '''
        Constructor
        '''
        
        # run the superclass' contructor
        super().__init__(modelObject, errorObject, variables, *args, **kwargs)
        
        # store required options
        self.opt['inertia'] = inertia   # inertia of the particle movement
        self.opt['c1'] = c1             # strength of the local attractors (bins)
        self.opt['c2'] = c2             # strength of the global attractors (toward the edges and center)
        self.opt['c3'] = c3             # strength of random changes
        self.opt['pBins'] = pBins       # number of local attractors
        self.opt['partial'] = partial   # fraction of the variables to be modified.
        
        # define required variables
        if self.opt['c3'] == 0:
            self.opt['c3'] = 1E-5
        self.partial = max(int(np.round(self.opt['variables'] * self.opt['partial'])),1)
        self.pBestIdxs = np.arange(self.opt['population'], dtype=np.int32)
        self.gBestIdxs = np.arange(self.opt['population'], dtype=np.int32)
        self.velocities = np.zeros((self.opt['population'],self.opt['variables']))
        tmp = np.hstack((-np.Inf, np.linspace(0,1, num=self.opt['pBins'])))
        self.pBins = np.vstack((tmp[0:-1],tmp[1:]))
    
    def __best(self, fit):
        '''
        Calculates global and local attractors to improve the optimization.
        Global attractors correspond the best parameter set in each of three regions of the solution space
        Local attractors - or particle attractors - correspond to the best parameteter set in each one of n regions of the solution space according to non-exceedance
        '''
        sortedIdxs = np.argsort(fit[:,0])
        
        # for global attractors the solution space is splitted in 3 regions
        nRegions = 3
        globalIdxsList = np.array_split(sortedIdxs, nRegions)
        binBorders = np.arange(0,nRegions+1)/nRegions
        binBorders[-1] = np.inf
        for i0 in range(nRegions):
            tmpIdxs = np.where(np.logical_and(fit[:,0]>=binBorders[i0], fit[:,0]<binBorders[i0+1]))[0]
            if len(tmpIdxs)!=0:
                if i0==0:
                    tmp = np.array((0.0, np.min(fit[tmpIdxs,1])))
                elif i0==1:
                    """ The second iteration doesn't appears to need the computation of the eclideanDist
                    because the minimum value is already known here, once is already the minimum value goten here.
                    Sugestion:
                        tmp = tmpIdxs[np.argmin(fit[tmpIdxs,1])]
                        self.gBestIdxs[globalIdxsList[i0]] = tmp
                        continue
                       """
                    tmp = fit[tmpIdxs[np.argmin(fit[tmpIdxs,1])],]
                else:
                    tmp = np.array((1.0, np.min(self.fitted[tmpIdxs,1])))
                dist = self.__euclideanDist(tmp, self.fitted[tmpIdxs,])
                self.gBestIdxs[globalIdxsList[i0]] = tmpIdxs[np.argmin(dist)]
        
        # for local attractors the solution space is split in n regions
        for i0 in range(self.opt['pBins']):
            tmpIdxs = np.where(np.logical_and(fit[:,0]>self.pBins[0, i0], fit[:,0]<=self.pBins[1, i0]))[0]
            if len(tmpIdxs)!=0:
                self.pBestIdxs[tmpIdxs] = tmpIdxs[np.argmin(fit[tmpIdxs,1])]
        tmpIdxs = np.where(fit[:,0] >= self.pBins[1, self.opt['pBins']-1])[0]
        if len(tmpIdxs)!=0:
            self.pBestIdxs[tmpIdxs] = tmpIdxs[np.argmin(fit[tmpIdxs,1])]
        
    def __euclideanDist(self, rRef, pList):
        '''
        Auxiliary function that calculates the Euclidean distance between solutions and
        reference points along in the solution space
        '''
        dist = np.zeros(pList.shape[0])

        for i0 in range(len(rRef)):
            dist += (pList[:,i0]-rRef[i0])**2
        return dist**0.5
    
    def _generate(self, fit, population):
        '''
        Generation method for the PSO implementation
        (used to create new parameter sets)
        
        returns the new parameter sets
        updates the self.jointVelocities variable (saved as an object variable because it is specific to the PSO implementation)
        '''
        # prepare best entries
        self.__best(fit)
        pBest = self.population[self.pBestIdxs,]
        gBest = self.population[self.gBestIdxs,]
           
        # calculate velocities
        tmpRnd1 = np.random.uniform(low=0.0, high=self.opt['c1'], size=(self.opt['population'],1)).repeat(self.opt['variables'],1)
        tmpRnd2 = np.random.uniform(low=0.0, high=self.opt['c2'], size=(self.opt['population'],1)).repeat(self.opt['variables'],1)
        tmpRnd3 = np.random.normal(loc=0.0, scale=self.opt['c3'], size=(self.opt['population'], self.opt['variables']))
        # limit changes to a number of variables (if required)
        if self.opt['partial']<1:
            idxs = np.arange(self.opt['variables'])
            partial = np.zeros_like(tmpRnd1)
            for i0 in range(self.opt['population']):
                partial[i0, np.random.permutation(idxs)[:self.partial]] = 1
            
            tmpRnd1 *= partial
            tmpRnd2 *= partial
            tmpRnd3 *= partial
        
        # update velocities for existing particles
        self.velocities *= self.opt['inertia']
        
        # update velocities for new particles
        candidateVelocities = (self.velocities + 
                               tmpRnd1 * (pBest-self.population) + 
                               tmpRnd2 * (gBest-self.population) +
                               tmpRnd3)
        
        # calculate positions
        candidates, toReflect = self._enforceBounds(self.population + candidateVelocities)
        candidateVelocities[toReflect] *= -0.5

        self.jointVelocities = np.vstack((self.velocities, candidateVelocities))
        
        return candidates
    
    def _select(self, fit, frontLvls, crowdDist, population):
        '''
        Selection method for the PSO implementation
        (used to choose which parameter sets to keep)
        
        returns the index of parameters sets to keep
        updates the self.velocities variable (saved as an object variable because it is specific to the PSO implementation)
        '''

        toKeepIdxs = []
        available = self.opt['population']-len(toKeepIdxs)
        refFront = 0
        while available > 0:
            frontIdxs = np.where(frontLvls==refFront)[0]
            if len(frontIdxs) <= available:
                toKeepIdxs.extend(frontIdxs)
            else:
                # Seems that the next line is doing nothing, because the fit_ has only its [1]
                # value changed, and the fit in the lexsort is the [0] value.
                # Organize the frontIdxs by crowding distance, using the non-exceedances as a tie-breaker 
                sortedPhenotype = frontIdxs[np.lexsort((fit[frontIdxs, 1], crowdDist[frontIdxs, 0]))[::-1]]
                    # (above) lexsorting with fit[frontIdxs, 0] and not fit[frontIdxs, 1] privileges high non-exceedances
                toKeepIdxs.extend(sortedPhenotype[:available])
            available = self.opt['population']-len(toKeepIdxs)
            refFront += 1
        
        toKeepIdxs = np.array(toKeepIdxs)
                         
        # update particle velocities
        self.velocities = self.jointVelocities[toKeepIdxs,]
        
        return toKeepIdxs