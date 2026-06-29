import numpy as np
import multiprocessing
import os
import shutil
from pathlib import Path
import tempfile
import pandas as pd 
import subprocess
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import time

def init(l, q):
    global lock, queue
    lock = l
    queue = q

def run_model(cls_instance, parameters):
    '''This function is responsible for running a model for a given set of parameters.
    It is used in the multiprocessing.Pool function to run the model in parallel.
    cls_instance: HYPE object --> Is used to access the attributes and methods of the HYPE object
    parameters: list --> List of parameters to be used in the model'''
    global lock, queue
    folder_index = None
    while folder_index is None:
        with lock:
            if not queue.empty():
                folder_index = queue.get()
        if folder_index is None: # No folder available, wait and retry
            time.sleep(0.1)  # Sleep for 100 ms before retrying

    # Identifies the current process and runs it in its folder
    folder_ = cls_instance.temp_dir
    folder = folder_ / f'Process{folder_index}'

    # Write parameters to a txt file
    cls_instance.write_partxt(folder / 'par.txt', parameters)

    # Execute the model executable
    try:
        subprocess.run([str(folder / 'HYPEwithoutPopup4All.exe')], cwd=folder, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except: 
        print(f'Executable not running as supposed in folder Process{folder_index}')
        return np.tile(np.array(9999), (cls_instance.records,1))
    
    # Read the output file
    
        #This with could appear redundant, but it is necessary to avoid the error of the file not being closed by pandas
        #and the file being locked by the process, from the documentation only the pyarrow engine can handle multiple processes
        #as is not installed on the package for now, this is the best solution
    try: #So para correr um modelo velho
        path =  folder / cls_instance.outputfile
    except Exception as e:
        path =  os.path.join(folder, 'timeCOUT.txt')
    try:
        outputvalues = cls_instance.read_outtxt(path)
    except Exception as e:
        print(f'Error reading output in folder Process{folder_index}')
        return np.tile(np.array(9999), (len(cls_instance.records),1))
    
    with lock:
            queue.put(folder_index)

    #This check is not always valuable, because the output is continous and obs can have missing values, which are cleared out!    
    if len(outputvalues) != len(cls_instance.records):
        print(f'Error, output length mismatch in folder Process{folder_index}')
        return np.tile(np.array(9999), (cls_instance.records,1))

    return outputvalues

# def lin_comp_dec(func):
#     def wrapper(*args, **kwargs):
#         weights = args[0].parList.copy()
#         m = weights[:, 0]*5 -2.5
#         b = weights[:, 1]*100-50
#         serie = pd.read_csv(r'C:\Users\ruism\RunHYPE\Independent-GPU\DefaultHYPEfolder\TimeCOUT.txt', sep='\t', skiprows=0, header=1, index_col='DATE', parse_dates=True)
#         serie = serie.values
#         results = m*serie + b
#         return results
#     return wrapper

class HYPE():
    '''Model class for HYPE. Made to run inside GPU'''

    def __init__(self, parList=[], MultipleRuns=10, records=0, calibration_parameters=[], random=True, normalization=True, log=False, 
                 HYPEfolder=r'HYPEFolder', outfile='timeCOUT.txt', **kwargs):

        self.parList = parList #List of values of the parameters to be optimized. Can appear more than one to the same parName
        self.parNames = [] #List of the names of the parameters to be optimized 
        self.MultipleRuns = MultipleRuns #Number of parallel runs of HYPE
        self.simulations = [] #Results of the runs
        self.DefaultFolderLoc = Path(HYPEfolder) #Default folder location
        self.records = records #Number of records expected from the output file
        self.calibration_parameters = calibration_parameters #Parameters to be optimized, if its empty all the parameters will be optimized
        self.outputfile = outfile #Output file to be used in the model
        self.pool = None

        #Auto proposed bound values
        # self.parametersBounds = {
        #     'wcfc': [0.0001, 1, False, 'multiply'], 
        #     'wcep': [0.0001, 1, True, 'multiply'],
        #     'wcwp': [0.0001, 1, True, 'multiply'],
        #     'rivvel': [0.0000001, 10000, True],
        #     'wcfc1': [0.0001, 1, True],
        #     'wcfc2': [0.0001, 1, True],
        #     'wcfc3': [0.0001, 1, True],
        #     'rrcs1': [0.0001, 5, False, 'multiply'],
        #     'preccorr':[-1, 10, False, 'substitute'],
        #     # 'rrcs1': [0.001, 10],
        #     'epotdist': [5, 20, False, 'substitute'],
        #     'rrcs2': [0.001, 5, True, 'multiply'],
        #     'rrcs3': [0.001, 100, False, 'multiply'],
        #     'mperc1': [0.001, 10, True, 'multiply'],
        #     'mperc2': [0.001, 10, True, 'multiply'],
        #     'mactrinf': [0.001, 100, True, ],
        #     'macrate': [0.001, 1, True],
        #     'srrate': [0.001, 1, False, 'substitute'],
        #     'jhtadd': [0.001, 5, True],
        #     'jhtscale': [40, 120],
        #     'rcgrwst': [0.001, 10],
        #     'trrcs': [0.001, 30], 
        #     'mactrsm': [0.001, 1], #Fraction
        #     'rcgrw': [0.001, 30], 
        #     'lp': [0.001, 30],
        #     'damp': [0.001, 1], #Fraction
        #     'tcalt': [0.001, 2],
        #     'tcelevadd': [0.001, 2],
        #     'pcaddg': [0.001, 10],
        #     'pcelevadd': [0.001, 15],
        #     'gldepi': [0.001, 15], 
        #     'qmean': [0.001, 500],
        #     'kc': [0.001, 25, False, 'multiply'],
        # }
        #From Gabinete set
        self.parametersBounds = {
            'wcfc': [0.0001, 1, True, 'multiply'], 
            #'wcfc': [0.0001, 1, False, 'multiply'],
            'rrcs1': [0.0001, 5, True, 'multiply'],
            'preccorr':[-1, 15, False, 'substitute'],
            'tempcorr': [-25, 25, False, 'substitute'],
            'epotdist': [5, 20, False, 'substitute'],
            'rrcs2': [0.0001, 1, True, 'multiply'],
            'mperc1': [0.0001, 300, True, 'multiply'],
            #'mperc1': [0.001, 10, False, 'multiply'],
            'mperc2': [0.0001, 300, True, 'multiply'],
            #'mperc2': [0.001, 10, False, 'multiply'],
            'srrate': [0.0001, 1, True, 'substitute'],
            #'srrate': [0.0001, 1, False, 'multiply'],
            'lp': [0, 25, False, 'substitute'],
            'kc': [0.001, 25, False, 'multiply'],

            #Lake related parameters

            # Outlet lake
            'gratp': [0.0001, 10, True, 'substitute'],
            'gratk': [0.001, 10, True, 'substitute'],

            # Internal lake
            'gldepi': [0.01, 100, True, 'substitute'],

            #Snow Routine parameters

            ## Snow melt
            'ttmp': [-3, 5, False, 'substitute'],
            'cmlt': [0.001, 100, True, 'multiply'],

            # if model 2 is used, the following parameters are added
            'snalbmin': [0.001, 1, True, 'multiply'],
            'snalbmax': [0.001, 1, True, 'multiply'],
            'cmrad': [0.001, 10, True, 'multiply'],

            ## Snow cover
            'fscdist0': [0.001, 1, True, 'multiply'],
            'fscdist1': [0.001, 10, True, 'multiply'],
            'fscdistmax': [0.001, 1, True, 'multiply'],
            'fsck1': [0.001, 2, True, 'substitute'],
            'fsckexp': [0.000001, 0.0001, True, 'substitute'],

            #Glacier related parameters

            ## Default glacier parameters
            'glacttmp': [-10, 5, False, 'substitute'],
            'glaccmlt': [0.001, 100, True, 'substitute'],

            ## Alternative glacier parameters (accounting with radiation)
            'glaccmrad': [0.001, 101, True, 'substitute'],
            'glaccmrefr': [0.001, 1, True, 'substitute'],
            'glacalb': [0.001, 1, True, 'substitute'],

            #Douro recession testing
            'rcgrw': [0.0001, 1, True, 'substitute'],
            'trrcs': [0.0001, 1, True, 'multiply'],
            'srrcs': [0.001, 1, True, 'multiply'],
            'mactrinf': [0.001, 5, True, 'multiply'],
            'macfrac': [0.001, 1, True, 'multiply'],
            'srbeta': [0.0001, 20, True, 'substitute'],
            'damp': [0.001, 1, True, 'substitute'],
            'rivvel': [0.01, 100, True, 'substitute'],
            'fscdist1': [0.001, 10, True, 'multiply'],
        }

        #Options for the model
        self.opts = {'init_random': random,
                     'Normalization':normalization,
                     'standartBounds': [0.001, 200, True],
                     'defaultLogValue': log,
                     'defaultMode': 'multiply'} 
        
        #Correction to the wcfc, wcep, wcwp problem
        self.soilFrac = None
        self._log_bounds()
        self._parameters_rewrite_mode()

        #Temporary folders to run HYPE in parallel
        self.init_tmpFiles() 
        self.read_partxt(self.DefaultFolderLoc  / 'par.txt')

    def set_simulation_Dates(self, init_date, final_date, b_init=None):

        if b_init is None:
           b_init = init_date

        info_path = self.DefaultFolderLoc / 'info.txt'
        with open(info_path, 'r') as file:
            lines = file.readlines()

        for folder in os.listdir(self.temp_dir):
            info_path = self.temp_dir / folder / 'info.txt'
            with open(info_path, 'w') as file:
                for line in lines:
                    if line.startswith('bdate'):
                        file.write(f'bdate\t{b_init.strftime("%Y-%m-%d")}\n')
                    elif line.startswith('cdate'):
                        file.write(f'cdate\t{init_date.strftime("%Y-%m-%d")}\n')
                    elif line.startswith('edate'):
                        file.write(f'edate\t{final_date.strftime("%Y-%m-%d")}\n')
                    else:
                        file.write(line)

    def remove_tmpFiles(self):
        shutil.rmtree(self.temp_dir)
        self.temp_dir = None

    def init_tmpFiles(self):
        self.temp_dir = Path(tempfile.TemporaryDirectory(prefix='GPU_HYPE', dir=None).name)
        print(f'Simulations running in {self.temp_dir.absolute()}')
        for i in range(self.MultipleRuns):
            run_folder = self.temp_dir / f'Process{i}'
            run_folder.mkdir(parents=True, exist_ok=True)
            for file in os.listdir(self.DefaultFolderLoc):
                src = self.DefaultFolderLoc / file
                if src.is_file():
                    shutil.copy2(src, run_folder)
                elif src.is_dir():
                    folder = run_folder / file
                    folder.mkdir(parents=True, exist_ok=True)

    def read_partxt(self , path):
        '''Reads the HYPE par.txt file and returns a list of parameters
        in it. The list is stored in the parList attribute of the object self.'''
        if self.calibration_parameters == []:
            with open(path, 'r') as f:
                lines = f.readlines()
                parList = []
                parNames = []
                for li in lines:
                    li = li[:-1].split('\t')
                    if li[0] != '!!':
                        if self.parametersBounds[li[0]][3] == 'multiply':
                            values = np.array(li[1:])
                            values = values.astype(float)
                            parNames.append([li[0], values])
                            if self.parametersBounds[li[0]][2]:
                                parList += [np.log10(1)]
                            else:
                                parList += [1]
                        else:
                            parNames.append([li[0], len(li[1:])])
                            parList += [float(par) for par in li[1:]]
                self.parNames = parNames
                self.parList =[(np.array(parList))]
        else:
            soil_parameters = {}
            with open(path, 'r') as f:
                lines = f.readlines()
                parList = []
                parNames = []
                for li in lines:
                    li = li[:-1].split('\t')
                    if li[0] in ['wcfc', 'wcep', 'wcwp']:
                        soil_parameters[li[0]] = np.array([float(par) for par in li[1:]])
                        if len(soil_parameters) == 3:
                            soil_parameters['soil'] = 1 - soil_parameters['wcep'] - soil_parameters['wcwp'] - soil_parameters['wcfc']
                            soil_occupied = soil_parameters['soil'] + soil_parameters['wcwp']
                            self.soilFrac = soil_occupied

                    if li[0] in self.calibration_parameters:
                        if self.parametersBounds[li[0]][3] == 'multiply':                            
                            values = np.array(li[1:])
                            values = values.astype(float)
                            if self.parametersBounds[li[0]][2]:
                                parList += [np.log10(1)]
                            else:
                                parList += [1]
                            parNames.append([li[0], values])
                            self.multiplyBounds(li[0], values)
                        else:
                            if self.parametersBounds[li[0]][2]:
                                parList += [np.log10(float(par)) for par in li[1:]]
                            else:
                                parList += [float(par) for par in li[1:]]
                            parNames.append([li[0], len(li[1:])])
            if len(soil_parameters) < 4:
                raise ValueError('Missing one of this soil parameters: wcfc, wcwp, and wcep')
            self.parNames = parNames    
            self.parList =[(np.array(parList))]

    def read_outtxt(self, path):
        '''Reads the out.txt file and returns a list of parameters in it.'''
        with open(path, 'r') as f:
            df = pd.read_csv(f, sep='\t', parse_dates=True, header=1, index_col=0)
            df = df.loc[self.records]
        return df[df.columns[0:]].values
    
    def write_partxt(self, path, parameters):
        '''Writes the parList attribute to the txt file, in order to save 
        the new parameters values and run HYPE.'''
        #Rewrite all the file
        #ADD HERE!!! the soilfraction verification
        if self.calibration_parameters == []:
            with open(path, 'w') as f:
                init = 0
                for par in self.parNames:
                    f.write(f'{par[0]}\t')
                    if self.parametersBounds[par[0]][3] == 'multiply':
                        values = parameters[init]*par[1]
                        line = '\t'.join([str(val) for val in values])
                        f.write(f'{line}\t')
                        f.write('\n')
                        init += 1
                    else:
                        for vi in range(init, init + par[1]-1):
                            f.write(f'{parameters[vi]}\t')
                        f.write(f'{parameters[init+par[1]-1]}')
                        f.write('\n')
                        init += par[1]
        #Rewrite only the parameters that are going to be optimized
        else:
            wcfc=False
            wcep=False
            with open(path, 'r+') as f:
                lines = f.readlines()
                modified_lines = []
                init = 0 #parList index
                p = 0 #parName index
                for i, line in enumerate(lines):
                    li = line.split('\t')
                    if li[0] not in self.calibration_parameters and li[0] != 'wcep':
                        modified_lines.append(line)
                    elif li[0] == 'wcep':
                            if wcfc:
                                modified_lines.append(li[0] + "\t" + '\t'.join([str(val) for val in wcepvals]) +'\n')
                            else:
                                wcepI = i
                    else:
                        if li[0] == 'wcfc':
                            if self.parametersBounds[li[0]][3] == 'multiply':
                                values = np.array(self.parNames[p][1])
                                line_val = values*parameters[init]
                                correction_val = line_val * (1- self.soilFrac)
                                modified_lines.append(li[0] + "\t" + '\t'.join([str(val) for val in correction_val]) +'\n')
                                init += 1
                            else: #substitute
                                values = np.array(parameters[init:init+len(li)-1])
                                correction_val = values * (1- self.soilFrac)
                                modified_lines.append(li[0] + "\t" + '\t'.join([str(val) for val in correction_val]) +'\n')
                                init += len(li)-1
                            wcepvals = 1 - self.soilFrac - correction_val
                            if wcep:
                                modified_lines[wcepI] = 'wcep'+ "\t" + '\t'.join([str(val) for val in wcepvals] +'\n')
                            wcfc=True
                        else:
                            if self.parametersBounds[li[0]][3] == 'multiply':
                                values = np.array(self.parNames[p][1])
                                line_val = values*parameters[init]
                                modified_lines.append(li[0] + "\t" + '\t'.join([str(val) for val in line_val]) +'\n')
                                init += 1
                            else:
                                modified_lines.append(li[0] + "\t" + '\t'.join([str(val) for val in parameters[init:init+len(li)-1]]) +'\n')
                                init += len(li)-1
                        p += 1

                f.seek(0)
                f.truncate()
                for line in modified_lines:
                     f.write(line)
                f.close()

    def setWeights(self, weights):
        '''Sets the weights of the parameters to be optimized.'''
        #Developer Notes: Is necessary to verify for the case where some of the parameters does not have boundaries,
        #maybe use a standart boundary for them
        self.parList = weights.copy()
        if self.opts['Normalization']:
            i = 0 
            for par in self.parNames:
                if self.parametersBounds[par[0]][3] == 'substitute':
                    tmp = self._denormPar(weights[:, i:i+par[1]], par[0])
                    if self.parametersBounds[par[0]][2]:
                        tmp = 10**(tmp)
                    self.parList[:, i:i+par[1]] = tmp
                    i += par[1]
                else: #multiply
                    tmp = self._denormPar(weights[:, i], par[0])
                    if self.parametersBounds[par[0]][2]:
                        tmp = 10**(tmp)
                    self.parList[:, i] = tmp
                    i += 1
        else:
            i = 0 
            for par in self.parNames:
                if self.parametersBounds[par[0]][3] == 'substitute':
                    tmp = weights[:, i:i+par[1]]
                    if self.parametersBounds[par[0]][2]:
                        tmp = 10**(tmp)
                    self.parList[:, i:i+par[1]] = tmp
                    i += par[1]
                else: #multiply
                    tmp = weights[:, i]
                    if self.parametersBounds[par[0]][2]:
                        tmp = 10**(tmp)
                    self.parList[:, i] = tmp
                    i += 1
            
    #@lin_comp_dec
    def compute(self):
        '''Computes the model with the parameters in the parList attribute.'''

        #To be compatible with older versions ()
        try:

            if self.pool == None:
                self.create_Pool()

        except AttributeError as e:
            self.pool = None
            self.create_Pool()

        # Temporarily detach the pool to avoid pickling issues
        temp_pool = self.pool
        self.pool = None

        partial_run_model = partial(run_model, self)

        results = temp_pool.map(partial_run_model, self.parList)

        # Reattach the pool
        self.pool = temp_pool

        #Arrange the results to fit GPU expected format
        results = np.array(results)
        results = results.reshape(results.shape[0], results.shape[1])
        results = results.T
        self.simulations = results

        return self.simulations
    
    def create_Pool(self):
        '''Creates a multiprocessing pool for the model.'''
        num_processes = min(len(self.parList), self.MultipleRuns)
        lock = multiprocessing.Lock()
        queue = multiprocessing.Queue()
        [queue.put(i) for i in range(self.MultipleRuns)]
        self.pool = multiprocessing.Pool(processes=num_processes,  initializer=init, initargs=(lock, queue))

    def delete_Pool(self):
        """
        Shutdown hook for the model’s multiprocessing pool, ensuring child workers stop accepting tasks, finish outstanding work, and release their resources. Closing prevents new tasks from being submitted, while joining blocks until all worker processes exit cleanly, avoiding zombie processes or incomplete state that could corrupt subsequent runs.
        """
        '''Clears the multiprocessing pool for the model.'''
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None
    
    def _normPar(self, v):
        '''Normalizes the parameters to be optimized.'''
        i = 0
        for par in self.parNames:
            if self.parametersBounds[par[0]][3] == 'substitute':
            #normalization not working right
                v[:, i:i+par[1]] = (v[:, i:i+par[1]] - self.parametersBounds[par[0]][0])/(self.parametersBounds[par[0]][1] - self.parametersBounds[par[0]][0])
                i += par[1]
            else: #multiply
                v[:, i] = (v[:, i] - self.parametersBounds[par[0]][0])/(self.parametersBounds[par[0]][1] - self.parametersBounds[par[0]][0])
                i += 1
        return v

    def _denormPar(self, v, par):
        '''Denormalizes the parameters to be optimized.'''
        r =v*(self.parametersBounds[par][1] - self.parametersBounds[par][0]) + self.parametersBounds[par][0]
        # if np.any(r > self.parametersBounds[par][1]) or np.any(r < self.parametersBounds[par][0]):
        #     raise ValueError("Error in the normalization process from HYPE")
        return r

    def prepareDump(self):
        '''Just to be compatible with GPU.save() method.'''
        pass
        
    def set_Data(self, path=None):
        '''This function was introduced to make this class compatible with the fit 
        method initialization on GPU.'''
        if path == None:
            path = self.DefaultFolderLoc
        pass

    def _log_bounds(self):
        for par in self.calibration_parameters:
            #Missing case where its equal to three but mode is defined instead of the log
            if par in self.parametersBounds.keys():
                if len(self.parametersBounds[par]) >= 3 and self.parametersBounds[par][2] == True and self.parametersBounds[par][3] == 'substitute':
                    self.parametersBounds[par][0:2] = [np.log10(self.parametersBounds[par][0]), np.log10(self.parametersBounds[par][1])]
                elif len(self.parametersBounds[par]) == 2 and self.opts['defaultLogValue'] and self.opts['defaultMode'] == 'substitute':
                    self.parametersBounds[par] = [np.log10(self.parametersBounds[par][0]), np.log10(self.parametersBounds[par][1])]
                    self.parametersBounds[par].append(True)
                elif len(self.parametersBounds[par]) == 2:
                    self.parametersBounds[par].append(False)
            else:
                self.parametersBounds[par] = self.opts['standartBounds']
                #Log?
                if self.opts['standartBounds'][2] and self.opts['defaultMode'] == 'substitute':
                    self.parametersBounds[par] = [np.log10(self.opts['standartBounds'][0]), np.log10(self.opts['standartBounds'][1]), True]
                elif self.opts['standartBounds'][2]:
                    self.parametersBounds[par] = [self.opts['standartBounds'][0], self.opts['standartBounds'][1], True]
                else:
                    self.parametersBounds[par] = [self.opts['standartBounds'][0], self.opts['standartBounds'][1], False]

    def _parameters_rewrite_mode(self):
        '''This function is responsible for verify if a rewrite mode was applied to the calibration parameters, if not it uses
        the default mode.'''
        for par in self.calibration_parameters:
                if len (self.parametersBounds[par]) == 4 and not self.parametersBounds[par][3] in ['substitute', 'multiply']:
                    self.parametersBounds[par][3] = self.opts['defaultMode']
                elif len(self.parametersBounds[par]) == 3:
                    self.parametersBounds[par].append(self.opts['defaultMode'])

    def multiplyBounds(self, par, values):
        #Mbounds must be defined after the read, since they rely on the parameters values to define the interval.
        #What if the interval were negative?

        interval = (self.parametersBounds[par][0]/min(values),  self.parametersBounds[par][1]/max(values))
        if self.parametersBounds[par][2]:
            interval = np.log10(interval)
        self.parametersBounds[par][0] = interval[0]
        self.parametersBounds[par][1] = interval[1]

    @staticmethod
    def plot_wbalance(folder_path, what_to_plot=['volumes', 'flows', 'main_downs'], entities=['mriver', 'soillayer1', 'soillayer2', 'soillayer3', 'lstream', 'ilake', 'olake', 'aquifer', 'riverplain'], observations=None, basins=None):
        """Function to plot multiple graphs from specific files in a folder.
        Function to plot multiple graphs from specific files in a folder.
        Plots will be arranged in a n x 1 layout.
        Args:
            folder_path (str): Path to the folder containing the result files.
            plot_options (list): List of file names (without extension) to plot.
        Returns:
            None: Displays the plots.
        """

        if 'volumes' in what_to_plot:
            #Volume files
            if len(entities) > 4:
                n_groups = len(entities) // 4 + (1 if len(entities) % 4 != 0 else 0)
                groups = [entities[i*4:(i+1)*4] for i in range(n_groups)]
            
            for group in groups:
                fig, axs = plt.subplots(len(group), 1, sharex=True)
                fig.set_size_inches([19.2 ,  9.75])
                fig.suptitle('Volumes [m^3]') 
                for i, entity in enumerate(group):
                    if entity == 'mriver':
                        file_name = '0012033.txt'
                    else:
                        file_name = f'WBs_{entity}.txt'
                    data = pd.read_csv(folder_path / file_name, sep='\t', index_col=0, parse_dates=True, skiprows=[1], dayfirst=False)

                    data = data[basins] if basins is not None and len(data.columns) > 1 else data
                    with plt.style.context('seaborn-darkgrid'):
                        axs[i].plot(data)
                        axs[i].set_title(f'{entity}')
                        axs[i].grid(True)
                plt.xlim(data.index.min(), data.index.max())
                plt.xlabel('Date')
                plt.ylim(bottom=0)
                plt.tight_layout()
                plt.legend()
        
        if 'flows' in what_to_plot:
            #Flow files

            if len(entities) > 6:
                if 'soillayer1' in entities:
                    entities.insert(1, 'satsurfaceflow')
                    entities.insert(2, 'rain_surfacerunoff')
                n_groups = len(entities) // 6 + (1 if len(entities) % 6 != 0 else 0)
                groups = [entities[i*6:(i+1)*6] for i in range(n_groups)]
            
            for group in groups:
                fig, axs = plt.subplots(len(group), 1, sharex=True)
                fig.set_size_inches([19.2 ,  9.75])
                fig.suptitle('Flows [m^3/s]') 
                for i, entity in enumerate(group):
                    if entity == 'mriver':
                        file_name = '0012033.txt'
                    elif 'satsurfaceflow' in entity:
                        file_name = f'WBf_satsurfaceflow_soillayer1_lstream.txt'
                    elif 'rain_surfacerunoff' == entity:
                        file_name = f'WBf_rain_surfacerunoff__lstream.txt'
                    elif 'soil' in entity:
                        file_name = f'WBf_soilrunoff_{entity}_lstream.txt'
                    else:
                        continue
                    data = pd.read_csv(folder_path / file_name, sep='\t', index_col=0, parse_dates=True, skiprows=[1], dayfirst=False)
                    data = data[basins] if basins is not None and len(data.columns) > 1 else data
                    with plt.style.context('seaborn-darkgrid'):
                        if entity != 'mriver':
                            data = data / 24 / 3600  # Convert from m3/day to m3/s
                        axs[i].plot(data, label=data.columns)
                        axs[i].set_title(f'{entity}')
                        axs[i].grid(True)
                        axs[i].legend(ncol=len(data.columns), frameon=False)
                        

                    if entity == 'mriver' and observations is not None:
                        obs_data = pd.read_csv(observations, sep='\t', index_col=0, parse_dates=True, dayfirst=False)
                        axs[i].plot(obs_data, label=f'Observed', linestyle='--')

                plt.xlim(data.index.min(), data.index.max())
                plt.xlabel('Date')
                plt.tight_layout()
                plt.ylim(bottom=0)

        if 'main_downs' in what_to_plot:
            #Main downstream flow file
            file_name = 'WBf_flow_olake_mriver_maindownstream.txt'
            data = pd.read_csv(folder_path / file_name, sep='\t', index_col=0, parse_dates=True, skiprows=[1], dayfirst=False)
            if len(data.columns) > 5:
                n_groups = len(data.columns) // 5 + (1 if len(data.columns) % 5 != 0 else 0)
                groups = [data.columns[i*5:(i+1)*5] for i in range(n_groups)]
            for group in groups:
                fig, axs = plt.subplots(len(group), 1, sharex=True)
                fig.set_size_inches([19.2 ,  9.75])
                for i, data_col in enumerate(group):
                    axs[i].plot(data.index, data[data_col]/24/3600, label=data_col)
                    axs[i].set_title(f'Main Downstream Flow - {data_col}')
                    axs[i].set_xlabel('Date')
                    axs[i].set_ylabel('Flow')
                    axs[i].set_xlim(data.index.min(), data.index.max())
                    axs[i].grid(True)
                    axs[i].legend()
                plt.xlim(data.index.min(), data.index.max())
                plt.xlabel('Date')
                plt.tight_layout()
                plt.ylim(bottom=0)
                plt.legend()
        plt.show(block=False)
        pass

    @staticmethod
    def single_run(folder_path, parameters, based_on_instance=None, non_exceedance=0.5):
        '''Runs a single instance of HYPE in a given folder with the given parameters.
        Args:
            folder_path (str): Path to the folder where HYPE is located.
            parameters (dict): Parameters as keys and a list of values.
            based_on_instance (GPU object, optional): Instance of GPU to base the run on. Defaults to None.
            non_exceedance (float, optional): Non-exceedance probability for the run. Defaults to 0.5.
        Returns:
            predictions (numpy.ndarray): Predictions from the HYPE run.

        '''
        # Read first to store the original par.txt in lines
        par_path = Path(folder_path) / 'par.txt'
        with open(par_path, 'r') as f:
            lines = f.readlines()

        #If based_on_instance is provided, write the parameters from the closest model
        if based_on_instance is not None:
            if not isinstance(based_on_instance.modelObject, HYPE):
                raise ValueError('based_on_instance must be an instance of GPU with a HYPE modelObject.')
            # Find the closest model index based on non-exceedance probability
            model_idx = np.argmin(np.abs(based_on_instance.fitted[:, 0] - non_exceedance))
            print('\nWARNING! macfrac remotion is only for testing purposes, please remove it afterwards\n')
            based_on_instance.modelObject.calibration_parameters.remove('macfrac') if 'macfrac' in based_on_instance.modelObject.calibration_parameters else None
            based_on_instance.modelObject.write_partxt(par_path, based_on_instance.modelObject.parList[model_idx])
               
        #Change the lines accordingly to the parameters provided
        new_lines = []
        if parameters is not None:
            for li in lines:
                li = li[:-1].split('\t')
                if li[0] in parameters.keys():
                    if len(parameters[li[0]]) != len(li[1:]):
                        raise ValueError(f'Parameter {li[0]} has {len(li[1:])} values in par.txt but {len(parameters[li[0]])} were provided.')
                    new_lines.append(li[0] + '\t' + '\t'.join([str(val) for val in parameters[li[0]]]) + '\n')
                else:
                    new_lines.append(li[0] + '\t' + '\t'.join(li[1:]) + '\n')

            with open(par_path, 'w') as f:
                f.writelines(new_lines)
        
        #Run HYPE
        subprocess.run([str(Path(folder_path) / 'HYPEwithoutPopup4All.exe')], cwd=folder_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        #Restablish the original par.txt
        with open(par_path, 'w') as f:
            f.writelines(lines) 


    def _get_par_distribution(self, population, fit, save_path=None, show_plots=False):
        '''
        Gets the distribution of the parameters to be optimized.
        Args:
            population (numpy.ndarray): Population of parameters, provided by GPU
            fit (numpy.ndarray): Fitness values of the population, provided by GPU (Error, non-exceedance)
        '''
        i=0
        for par in self.parNames:
            title = par[0]
            if isinstance(par[1], np.ndarray):
                if self.opts['Normalization']:
                    #The normalization also suffers from alterations in the self.parametersBounds
                    tmp = self._denormPar(population[:, i], par[0])
                    #You need to store the the log value at the time of
                    #the calibration so alterations in self.parmetersBound don't mess with old models results
                if self.parametersBounds[par[0]][2]:
                    tmp = 10**(tmp)
                tmp = tmp.reshape(-1, 1)
                tmp = tmp*par[1]
                i += 1
                plt.figure(figsize=(10,6))
                plt.title(f'Parameter distribution for {title}')
                if tmp.shape[1] > 1:
                    for j in range(tmp.shape[1]):
                        plt.plot(fit[:,0], tmp[:, j], 'o', alpha=0.5, label=f'Value {j+1}')
                    plt.legend()
            else:
                if self.opts['Normalization']:
                    tmp = self._denormPar(population[:, i:i+par[1]], par[0])
                if self.parametersBounds[par[0]][2]:
                    tmp = 10**(tmp)
                i += par[1]
                plt.figure(figsize=(10,6))
                plt.title(f'Parameter distribution for {title}')
                if par[1] > 1:
                    for j in range(par[1]):
                        plt.plot(fit[:,0], tmp[:, j], 'o', alpha=0.5, label=f'Value {j+1}')
                    plt.legend()
                else:
                    plt.plot(fit[:,0], tmp,  'o', alpha=0.5)
                
            plt.xlim(0, 1)
            plt.xlabel('Non-exceedance probability')
            plt.ylabel('Parameter value')
            plt.tight_layout()
            if show_plots:
                plt.show(block=False)
        
            if save_path is not None:
                plt.savefig(Path(save_path) / f'Parameter_distribution_{title}.png', dpi=300)

        print('Distributions plotted successfully.')

def predictHYPEdec(func):
        '''Predict decorator
            Made to compatibilize the predict method  with the HYPE class
            still not account with some details:
            1) if the folders where the model runs are not present it will raise an error'''
        def wrapper(*args, **kwargs):
            if isinstance(args[0].modelObject, HYPE):
                simulations = args[0].modelObject.compute()
                #Original form, does not work because the two '_' make it a private method
                #aggregated = args[0].__aggregateByBandWrapper(simulations)
                #Alternative form
                aggregated = args[0]._aggregateByBand(simulations, args[0].fitted, args[0].bandBounds, args[0].opt['minModels'], \
                                                      args[0].opt['forcePositive'])
                
                args[0].modelObject.delete_Pool()
                return aggregated
            else:
                print("Condition not met, running alternative action")
                # Alternative action can be anything you define here
                return func(*args, **kwargs)
        return wrapper    


class metaHYPE(HYPE):
    '''This class is a meta class for HYPE, to make a correction in the computation to address other parameter
    b which will be sum to the output of each model, this is supposed this will ensure that 
    we have covered all the non-exceedance interval, since the physically base of the model does not allow
    to cover all the probability ranges.'''
    def __init__(self, m=[-3,3], b=[-7.5,7.5], *args, **kwargs):
        self.bBounds = b
        self.mBounds = m
        super().__init__(*args, **kwargs)
    
    def compute(self):
        '''Computes the model with the parameters in the parList attribute.'''
        partial_run_model = partial(run_model, self, )
        num_processes = min(len(self.parList), self.MultipleRuns) 
        hype_parameters = self.parList[:, :-2]
        b = self.parList[:, -1]
        m = self.parList[:, -2]
        if self.opts['Normalization']:
            b = b*(self.bBounds[1] - self.bBounds[0]) + self.bBounds[0]
            m = m*(self.mBounds[1] - self.mBounds[0]) + self.mBounds[0]
        with multiprocessing.Pool(processes=num_processes, initializer=init, initargs=(lock)) as pool:
           results = pool.map(partial_run_model, hype_parameters)
           results = np.array(results)
           results = results.reshape(results.shape[0], results.shape[1])
           results = results.T
           results = results*m + b
           self.simulations = results
           return self.simulations

#Debug purposes!    
if __name__ == '__main__':

    # Minimal self-check against the bundled Türkheim example (run from the repo root).
    from pathlib import Path

    folder = Path('examples/set7_germany_tuerkheim')
    y = pd.read_csv('examples/Qobs.txt', sep='\t', header=0, index_col='Date', parse_dates=True)
    init_date = pd.to_datetime('1980-01-01')
    final_date = pd.to_datetime('1989-12-31')
    index2 = y.index[(y.index >= init_date) & (y.index <= final_date)]
    y = y.loc[index2]
    test = HYPE(MultipleRuns=4, records=index2, calibration_parameters=['wcfc', 'rrcs1'],
                HYPEfolder=str(folder), outfile='results/0050675.txt')
    test.set_simulation_Dates(init_date, final_date)
    test.remove_tmpFiles()
    print('HYPE wrapper initialised OK against', folder)

