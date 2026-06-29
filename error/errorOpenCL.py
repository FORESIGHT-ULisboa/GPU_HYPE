# coding: utf-8
'''
Created on 06/06/2015

@author: Jose Pedro Matos
'''

import warnings
import pkg_resources

import numpy as np
import pyopencl as cl

from openCL import OpenCL
#===============================================================================
# from pyopencl.cffi_cl import CompilerWarning
#===============================================================================

class Error(object):
    DEVICE_TYPE = {'ALL': cl.device_type.ALL, 'CPU': cl.device_type.CPU, 'GPU': cl.device_type.GPU} # @UndefinedVariable
    KERNELS = {'MAE': 'evalMAE.cl',
               'MSE': 'evalMSE.cl'}
    
    def __init__(self, stride=100, workGroup=(16, 16), platform=0, deviceType='ALL', verbose=0, errorFunction='MAE'):
        self.targets = None
        self.stride=stride
        self.errorFunction = errorFunction
        self.workGroup = workGroup
        self.platform = platform
        self.deviceType = deviceType
        
        self.sizes = dict()
        
        self.__startOpenCL__()
    
    def __startOpenCL__(self):
        self.openCL = OpenCL(self.workGroup, self.platform, self.DEVICE_TYPE[self.deviceType])
        with warnings.catch_warnings():
            #===================================================================
            # warnings.filterwarnings("ignore", category=CompilerWarning)
            #===================================================================
            kernelStr = pkg_resources.resource_string(__name__, self.KERNELS[self.errorFunction]) #@UndefinedVariable
            self.openCL.setKernel(kernelStr)
    
    def prepareDump(self):
        self.targets = None
        self.openCL = None
    
    def _increment(self, base, interval):
        tmp=base%interval
        if tmp==0:
            return (base, 0)
        else:
            return (base+interval-tmp, interval-tmp)
    
    def setTargets(self, targets):
        self.targets = targets.values.reshape(-1, order = 'C').astype(np.float32)
        
    def reshapeData(self, simulations):
        self.__startOpenCL__()
        self.sizes['originalObs'], self.sizes['originalPop'] = simulations.shape
        
        tmp0, tmp1 = self._increment(self.sizes['originalObs'], self.openCL.workGroup[0]*self.stride)
        tmpGroups = tmp0//self.stride
        self.stride -= int(np.floor(tmp1/tmpGroups))
        
        self.sizes['reshapedObs'], self.sizes['addObs'] = self._increment(self.sizes['originalObs'], self.openCL.workGroup[0]*self.stride)
        self.sizes['reshapedPop'], self.sizes['addPop'] = self._increment(self.sizes['originalPop'], self.openCL.workGroup[1])
        
        if self.openCL.verbose != 0:
            print('Vertical array adjustment: +%.1f%% (%u stride, %ux %u items)' % (self.sizes['addObs']/self.sizes['originalObs']*100, self.stride, self.sizes['reshapedObs']//self.stride//self.openCL.workGroup[0], self.openCL.workGroup[0]))
            print('Horizontal array adjustment: +%.1f%% (%ux %u items)' % (self.sizes['addPop']/self.sizes['originalPop']*100, self.sizes['reshapedPop']//self.openCL.workGroup[1], self.openCL.workGroup[1]))
            
    def compute(self, simulations):
        
        if self.openCL==None:
            self.__startOpenCL__()
        
        if 'reshapedObs' not in self.sizes.keys():
            self.reshapeData(simulations)
        
        simOpenCL = simulations.reshape(-1, order = 'F').astype(np.float32)
        
        globalSize = (int(np.int32(self.sizes['reshapedObs'])//self.stride), int(self.sizes['reshapedPop']))
        localSize = (int(self.openCL.workGroup[0]), int(self.openCL.workGroup[1]))
        
        mf = cl.mem_flags
        stride = np.int32(self.stride)
        length = np.int32(simulations.shape[0])
        lim0 = np.int32(np.ceil(self.sizes['originalObs']/self.stride))
        #=======================================================================
        # lim0 = np.int32(self.sizes['reshapedObs']/self.stride)
        #=======================================================================
        lim1 = np.int32(self.sizes['originalPop'])
        observedBuffer = cl.Buffer(self.openCL.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.targets)
        simulatedBuffer = cl.Buffer(self.openCL.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=simOpenCL)
        outErrorBuffer = cl.Buffer(self.openCL.ctx, mf.WRITE_ONLY, int(np.prod(globalSize)*np.int32(1).nbytes))
        outNonExceedanceBuffer = cl.Buffer(self.openCL.ctx, mf.WRITE_ONLY, int(np.prod(globalSize)*np.int32(1).nbytes))
        
        kernel = self.openCL.prg.eval

        kernel(self.openCL.queue, globalSize, localSize,
               stride, length, lim0, lim1,
               observedBuffer, simulatedBuffer,
               outErrorBuffer, outNonExceedanceBuffer)
        
        error = np.empty((np.prod(globalSize),)).astype(np.float32)
        cl.enqueue_copy(self.openCL.queue, error, outErrorBuffer)
        #=======================================================================
        # error = np.reshape(error, globalSize, order='F')[:int(self.sizes['reshapedObs']/self.stride),:int(self.sizes['originalPop'])]
        #=======================================================================
        error = np.reshape(error, globalSize, order='F')[:lim0,:lim1]
        errorMetric = np.sum(error,0)/self.sizes['originalObs']
        
        nonExceedance = np.empty((np.prod(globalSize),)).astype(np.int32)
        cl.enqueue_copy(self.openCL.queue, nonExceedance, outNonExceedanceBuffer)
        nonExceedance = np.reshape(nonExceedance, globalSize, order='F')[:lim0,:lim1]
        nonExceedanceFraction = (np.sum(nonExceedance,0))/self.sizes['originalObs']

        return (errorMetric, nonExceedanceFraction)
        