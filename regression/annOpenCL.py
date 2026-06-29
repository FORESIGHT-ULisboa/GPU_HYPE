'''
Created in June 2015

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

class Weights(object):
    def __init__(self, wHL=None, bHL=None, wOL=None, bOL=None):
        # The different dimensions of the weights represent the connections between the layers in the neural network.
        # wHL: Weight matrix connecting the input layer to the hidden layer.
        # bHL: Bias vector for the hidden layer.
        # wOL: Weight matrix connecting the hidden layer to the output layer.
        # bOL: Bias vector for the output layer.
        self.wHL = wHL
        self.bHL = bHL
        self.wOL = wOL
        self.bOL = bOL

class ANN(object):
    DEVICE_TYPE = {'ALL': cl.device_type.ALL, 'CPU': cl.device_type.CPU, 'GPU': cl.device_type.GPU} # @UndefinedVariable
    KERNELS = {'lin': 'annLinear.cl',
               'tan': 'annTansig.cl',
               'log': 'annLogsig.cl',
               'relu': 'annLeakyRelu.cl'
               }
    
    def __init__(self, nodes=8, workGroup=(16, 16), platform=0, deviceType='ALL', verbose=0, activationFunction='tan', lowerThreshold=None):
        self.data = None
        self.nodes = nodes
        self.activationFunction = activationFunction
        self.workGroup = workGroup
        self.platform = platform
        self.deviceType = deviceType
 
        self.__startOpenCL__()
        
        self.verbose = verbose
        self.lowerThreshold = lowerThreshold
        self.weights = None
    
    def __startOpenCL__(self):
        self.openCL = OpenCL(self.workGroup, self.platform, self.DEVICE_TYPE[self.deviceType])
        with warnings.catch_warnings():
            #===================================================================
            # warnings.filterwarnings("ignore", category=CompilerWarning)
            #===================================================================
            kernelStr = pkg_resources.resource_string(__name__, self.KERNELS[self.activationFunction]) #@UndefinedVariable
            self.openCL.setKernel(kernelStr)
    
    def prepareDump(self):
        self.data = None
        self.openCL = None
    
    def __str__(self):
        return 'ANN model\nNodes: %u' % (self.nodes) + \
            '\nOpenCL:\n ' + str(self.openCL.devList) + \
            '\nwHL:\n' + np.array_str(self.weights.wHL) + \
            '\nbHL:\n' + np.array_str(self.weights.bHL) + \
            '\nwOL:\n' + np.array_str(self.weights.wOL) + \
            '\nbOL:\n' + np.array_str(self.weights.bOL)
        
    def setData(self, data):
        self.data = data
        if isinstance(self.weights, type(None)):
            self.setWeights()
        
    def getWeightLen(self):
        return (self.data.shape[1]+2)*self.nodes+1
    
    def getWeightsToRegularize(self):
        tmp=np.zeros(self.getWeightLen(), dtype=np.bool)
        tmp[:self.data.shape[1]*self.nodes] = True
        tmp[-self.nodes-1:-1] = True
        return tmp
    
    def setWeights(self, weights=None):
        if weights is None:
            weights = np.random.normal(loc=0, scale=1, size=self.getWeightLen())
        
        if len(weights.shape)==1:
            weights = np.expand_dims(weights, axis=0)
        
        self.weightsOpenCL = np.reshape(weights, (-1,))
        
        tmp = self.data.shape[1]*self.nodes
        wHL = np.reshape(weights[:, :tmp], (-1, self.data.shape[1], self.nodes))
        bHL = np.reshape(weights[:, tmp:tmp+self.nodes], (-1, self.nodes))
        tmp += self.nodes
        wOL = np.reshape(weights[:, tmp:tmp+self.nodes].T, (self.nodes, -1))
        bOL = np.reshape(weights[:, -1], (-1, 1))
        self.weights = Weights(wHL, bHL, wOL, bOL)
        self.weightsOpenCL = weights
    
    def compute(self, X=[]):
        
        if self.openCL==None:
            self.__startOpenCL__()
        
        if len(X)==0:
            X = self.data
        else:
            pass
        
        originalLength = X.shape[0]
        originalWidth = self.weightsOpenCL.shape[0]
        
        remData = np.remainder(X.shape[0], self.openCL.workGroup[0])
        if remData != 0:
            X = np.vstack((X, np.zeros((self.openCL.workGroup[0]-remData, X.shape[1]))))
        else:
            remData=self.openCL.workGroup[0]
        
        remNetwork = np.remainder(self.weightsOpenCL.shape[0],self.openCL.workGroup[1])
        if remNetwork != 0:
            weights = np.vstack((self.weightsOpenCL, np.zeros((self.openCL.workGroup[1]-remNetwork, self.weightsOpenCL.shape[1]))))
        else:
            weights = self.weightsOpenCL
            remNetwork = self.openCL.workGroup[1]
        
        XOpenCL = X.reshape(-1, order = 'C').astype(np.float32)
        weightsOpenCL = weights.reshape(-1, order = 'C').astype(np.float32)
        
        mf = cl.mem_flags
        inputs = np.int32(X.shape[1])
        nodes = np.int32(self.nodes)
        dataSize = np.int32(X.shape[0])
        weightSize = np.int32(self.weightsOpenCL.shape[1])
        dataBuffer = cl.Buffer(self.openCL.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=XOpenCL)
        weightsBuffer = cl.Buffer(self.openCL.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=weightsOpenCL)
        outBuffer = cl.Buffer(self.openCL.ctx, mf.WRITE_ONLY, int(XOpenCL.nbytes/inputs*weights.shape[0]))
        
        kernel = self.openCL.prg.ann
        globalSize = (int(X.shape[0]), int(weights.shape[0]))
        localSize = (int(self.openCL.workGroup[0]), int(self.openCL.workGroup[1]))
            
        kernel(self.openCL.queue, globalSize, localSize, inputs, nodes, dataSize, weightSize, dataBuffer, outBuffer, weightsBuffer, cl.LocalMemory(self.weightsOpenCL[0,].nbytes*localSize[1]))
        
        phiOL = np.empty((np.prod(globalSize),)).astype(np.float32)
        cl.enqueue_copy(self.openCL.queue, phiOL, outBuffer)
        phiOL = np.reshape(phiOL, globalSize, order='F')[:originalLength,:originalWidth]
    
        if self.lowerThreshold != None:
            phiOL[phiOL<self.lowerThreshold] = self.lowerThreshold
        
        #phiOL contains the output of every one of the N modules to each input set.
        return phiOL