'''
Created on 4 jun 2015

@author: Jose Pedro Matos
'''

import numpy as np

def phenCrowdingNSGAII(*args, **kwargs):
    '''
    Phenotype crowding used in NSGAII
    '''
    
    if 'fronts' in kwargs:
        fronts = kwargs['fronts']
    else:
        fronts = list((range(0, len(args[0])),))
    
    distance = np.zeros_like(args[0])
    for m0 in args:
        if type(fronts) == type([]):
            for l0 in fronts:
                idx = np.array(l0)
                x = m0[idx]
                tmpSortIdx = np.argsort(x)
                tmpSort = x[tmpSortIdx]
                if (tmpSort[-1]-tmpSort[0]) != 0:
                    distance[idx[tmpSortIdx[1:-1]]] += (tmpSort[2:]-tmpSort[:-2])/(tmpSort[-1]-tmpSort[0])
                else:
                    distance[idx[tmpSortIdx[1:-1]]] = 0
                distance[idx[tmpSortIdx[0]]] = np.Inf
                distance[idx[tmpSortIdx[-1]]] = np.Inf
        else:
            for i0 in np.sort(np.unique(fronts)):
                idx = np.where(fronts==i0)[0]
                x = m0[idx]
                tmpSortIdx = np.argsort(x)
                tmpSort = x[tmpSortIdx]
                if (tmpSort[-1]-tmpSort[0]) != 0:
                    distance[idx[tmpSortIdx[1:-1]]] += (tmpSort[2:]-tmpSort[:-2])/(tmpSort[-1]-tmpSort[0])
                else:
                    distance[idx[tmpSortIdx[1:-1]]] = 0
                distance[idx[tmpSortIdx[0]]] = np.Inf
                distance[idx[tmpSortIdx[-1]]] = np.Inf
        
    return distance