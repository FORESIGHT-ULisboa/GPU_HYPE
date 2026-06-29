'''
Created on 10 janv. 2019

@author: MAJO
'''

import pyopencl as cl


def showPlatforms():
    '''
    Shows available opencl platforms and devices
    '''
    for platform in cl.get_platforms():
        print("Platform name:", platform.name)
        print("Platform profile:", platform.profile)
        print("Platform vendor:", platform.vendor)
        print("Platform version:", platform.version)
        for device in platform.get_devices(device_type=cl.device_type.ALL): #@Undefinedvariable
            print("---------------------------------------------------------------")
            print("    Device name:", device.name)
            print("    Device type:", cl.device_type.to_string(device.type))  # @UndefinedVariable
            print("    Device memory: ", device.global_mem_size//1024//1024, 'MB')
            print("    Device max clock speed:", device.max_clock_frequency, 'MHz')
            print("    Device compute units:", device.max_compute_units)
            print("    Device max work items:", device.get_info(cl.device_info.MAX_WORK_ITEM_SIZES))  # @UndefinedVariable
            print("    Device local memory:", device.get_info(cl.device_info.LOCAL_MEM_SIZE)//1024, 'KB')  # @UndefinedVariable

class OpenCL(object):
    
    def __init__(self, workGroup, platform, typeOfProcessor, verbose=0):
        '''
        Constructor
        '''
        
        self.workGroup = workGroup
        self.platform = platform
        self.type = typeOfProcessor
        self.verbose = verbose

        platform = cl.get_platforms()[self.platform]
        self.devList = platform.get_devices(device_type=self.type)
        self.ctx = cl.Context(devices=self.devList)
        self.queue = cl.CommandQueue(self.ctx)
        
        if self.verbose>0:
            print("Platform name:", platform.name)
            print("Platform profile:", platform.profile)
            print("Platform vendor:", platform.vendor)
            print("Platform version:", platform.version)
            for device in self.devList:
                print("---------------------------------------------------------------")
                print("    Device name:", device.name)
                print("    Device type:", cl.device_type.to_string(device.type))  # @UndefinedVariable
                print("    Device memory: ", device.global_mem_size//1024//1024, 'MB')
                print("    Device max clock speed:", device.max_clock_frequency, 'MHz')
                print("    Device compute units:", device.max_compute_units)
                print("    Device max work items:", device.get_info(cl.device_info.MAX_WORK_ITEM_SIZES))  # @UndefinedVariable
                print("    Device local memory:", device.get_info(cl.device_info.LOCAL_MEM_SIZE)//1024, 'KB')  # @UndefinedVariable

    def setKernel(self, kernel):
        self.prg = cl.Program(self.ctx, kernel.decode('UTF-8')).build()