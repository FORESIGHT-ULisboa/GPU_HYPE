'''Rui Marinheiro 21/5/24'''
import json
from gpu_pso import *
from gpu import *
from error.errorOpenCL import *
from regression.annOpenCL import *
import pandas as pd
import numpy as np
import matplotlib as plt
import pickle
import time 
from conceptual.HYPE import *
import performance
from pathlib import Path
import os
from new_error_model import NewErrorModel
#Debug purposes

'''This example file is the first experiment of joining HYPE and GPU the commentary
detail is higher in this doc because it will serve as example to further implementations
'''
if __name__ == '__main__':
    #plt.use('QtAgg')
    
    start = time.time()
    identifier = 'Casa '
    print('\nStarting...')

    #Set up
    population = 200
    epochs = 50
    #Each set of parameters inside the list is going to be tested sequentially! example:
    #loop_parameters = [['rrcs1', 'preccorr', 'kc'] ['wcfc', 'rrcs1', 'rrcs2', 'rrcs3'], ['rrcs1', 'rrcs2', 'rrcs3']]
    loop_parameters = [['wcfc', 'rrcs1', 'kc', 'ttmp', 'cmlt', 'preccorr'], ]
    #Hype options
    no_runs = 10
    random = True #init population
    normalization = True
    log = True #default log value, if log not set specifically for a parameter in HYPE class will use this value otherwise the value in the HYPE class
    #Note: parmeters intervals are defines in the HYPE class,
    sim_loc = Path(r'./Simulations/')
    os.makedirs(sim_loc, exist_ok=True)

    while loop_parameters != []:
        date = time.strftime('%Y-%m-%d_%Hh%M', time.localtime())
        parameters = loop_parameters.pop()
        print('Parameters:', parameters)
        #Data
        '''Contrairly to the ANN GPU implementation here the inputs for the model are written in
        the Tobs.txt and Pobs.txt (Temperature and Precipitation). So it ins't necessary to got them here
        althoug the observed discharge must be set to be used in the errorModel computations'''
        y = pd.read_csv(r'Qobs_Sweden6284.txt', sep=',', header=0, index_col='DATE', parse_dates=True, dayfirst=True)
        init_date = pd.to_datetime('2000-01-01')
        final_date = pd.to_datetime('2010-12-31')
        index2 = y.index[(y.index >= init_date) & (y.index <= final_date)]
        y = y.loc[index2]

        #Model Creation
        if 'rrcs2' in parameters:
            HYPEFolder = r'HYPEsweden\HypeFolder_w_rrcs2'
        else:
            HYPEFolder = r'HYPEsweden\HypeFolder'
        Model = HYPE(MultipleRuns=no_runs, records=index2, calibration_parameters=parameters, random=random, normalization=normalization, log=log,
                     HYPEfolder=HYPEFolder, outfile='results/0006284.txt')
        
        #Editing the simulation dates on the tmp folders
        Model.set_simulation_Dates(init_date, final_date)
        # + 2 only with METAHYPE
        variables = len(Model.parList[0]) #+ 2

        #Setting the error model
        #error_model = NewErrorModel(errorFunction='MAE', non_exceedance_threshold=1)
        error_model = Error(errorFunction='MAE')

        #GPU initialization
        ModelTest = GPU_PSO(modelObject=Model, errorObject=error_model, variables=variables, inertia=0.2, c1=0.3, c2=0.2, c3=0.001, pBins=10, partial=0.1, 
                            forcePositive=False, population=population, transformWeights=False, forceNonExceedance=0.01, displayEach=10)
        
        #naming the files
        if parameters != []:
            name = '_'.join(parameters)
        else:
            name = 'All'

        #Training
        print('Training...')
        os.makedirs(sim_loc / (identifier + date), exist_ok=True)
        os.makedirs(sim_loc / (identifier + date) / 'Calibration', exist_ok=True)
        save_fig = sim_loc / (identifier + date) / 'Calibration' / ('Training_DPS_TS' + '.pplot')
        results = ModelTest.fit(y, epochs=epochs, save=save_fig)

        #Remove the temporary files to save space in the computer
        ModelTest.modelObject.remove_tmpFiles()

        #Saving
        print('Saving the model...')
        os.makedirs(sim_loc / (identifier + date) / 'Model', exist_ok=True)
        ModelTest.save(sim_loc / (identifier + date) / 'Model' / ('GPU_HYPE_' + name + date + '.pkl'))
        finish = time.time()

        #Writting log file
        print('Writing log file...')
        path = sim_loc / (identifier + date) / 'calibration.log'
        with open(path, 'w') as f:
            f.write('Parameters: ' + str(parameters) + '\n')
            f.write('Training dates: ' + str(init_date) + ' to ' + str(final_date) + '\n')
            f.write('Population: ' + str(population) + '\n')
            f.write('Epochs: ' + str(epochs) + '\n')
            f.write('ModelTYPE: ' + Model.__class__.__name__ + '\n')
            f.write('HYPEoptions: ' + json.dumps(Model.opts) + '\n')
            f.write('Last registed performance: \n' + results[1][0])
            f.write(f'ErrorModel: {error_model.__class__.__name__}' + ' with ' + error_model.errorFunction + '\n')
            f.write(f'\nTotal time taken: {finish-start}s')

        m, s = divmod(finish-start, 60)
        h, m = divmod(m, 60)
        hours = f'{h} hours ' if h != 0 else ''
        minutes = f'{m} minutes ' if m != 0 else ''
        seconds = f'{int(s)} seconds' if s != 0 else ''
        if (minutes != '' or hours != '') and s!=0:
            seconds = 'and ' + seconds
        elif s==0 and hours!= '':
            minutes = 'and ' + minutes
        print(f'Training finished after ' + hours + minutes + seconds + '.')
