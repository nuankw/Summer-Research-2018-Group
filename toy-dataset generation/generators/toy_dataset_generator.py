# -*- coding: utf-8 -*-
# ipython console: %matplotlib auto
import numpy as np
import math
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import random
from enum import Enum

# VARIATION_TYPE['FREQUENCY_ONLY'].value == 1
# VARIATION_TYPE(1).name
# Out[43]: 'FREQUENCY_ONLY'
N_TYPE = 7

class SINGLE_VARIATION_TYPE(Enum):
    NEW_FUNCTION_ONLY = 0 # extra work needed
    FREQUENCY_ONLY = 1
    AMPLITUDE_ONLY = 2
    Y_CENTER_ONLY = 3
    T_ONLY = 4 # phase change
    RAND_VALUE_ONLY = 5
    ALL_ZERO = 6
    
    @classmethod
    def type_in_str(cls,number):
        return cls(number).name
    
    @classmethod
    def type_in_int(cls, name):
        return cls[name].value


class PatternGenerator:
    def __init__(self, n_periods ):
        self.n_periods = n_periods
        # numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
        self.t = np.linspace(0, 2*math.pi*n_periods, 20*n_periods)
        self.size = self.t.shape[0]
        self.x = np.array(range(self.size))
        self.scale = np.ones(self.size) * 0.5
        self.freq = np.ones(self.size) * 0.5
        self.func1 = np.sin # numpy.ufunc
        self.func2 = np.cos
        self.y_center = np.ones(self.size) * 0.5
        self.calculate_y()
        self.num_variations()
        self.allocate_variations()
        
        
    def num_variations(self):
        # determine the ratio of where anomaly happens (large and small redidual)
        assert self.size > 0
        if (self.size < 500):
            self.ratio_large = 0.05 # 10% of data
        elif (self.size < 1000):
            self.ratio_large = 0.035 # 7%
        else:
            self.ratio_large = 0.025 # 5%
        self.ratio_small = 2 * self.ratio_large
        
        self.n_large_anomaly = int(self.size * self.ratio_large)
        self.n_small_anomaly = int(self.size * self.ratio_small)
        self.n_anomaly = self.n_large_anomaly + self.n_small_anomaly

        anomaly_indexes = [random.randint(0, self.size-1) for _ in range(self.n_anomaly)]
        self.large_anomaly_indexes = anomaly_indexes[:self.n_large_anomaly]
        self.small_anomaly_indexes = anomaly_indexes[self.n_large_anomaly:]
        
        # allocate change (long and short span)
        self.ratio_change = self.ratio_large / 2
        self.n_change = int(self.size * self.ratio_change)
        self.n_large_change = int(self.n_change * 0.2) + 1
        self.n_small_change = self.n_change - self.n_large_change
        change_indexes = [random.randint(0, self.size-1) for _ in range(self.n_change)]
        self.large_change_indexes = change_indexes[:self.n_large_change]
        self.small_change_indexes = change_indexes[self.n_large_change:]
        
        # small change: short span (unlike anomaly which is just one point)
        small_portion_size = self.size // self.n_small_change
        small_change_startpts = np.array( np.array(range(0, self.n_small_change)) * small_portion_size, dtype = 'int32')
        small_change_period = np.array([ int(random.uniform(0.02,0.04) * self.size) for _ in range(self.n_small_change)])
        offset = np.array([ int(random.uniform(0,1) * small_portion_size) for _ in range(self.n_small_change)])
        self.small_change_spans = [ (small_change_startpts[i]+offset[i], small_change_startpts[i]+offset[i]+small_change_period[i] ) for i in range(self.n_small_change)]
        
        '''
        LARGE CHANGE: NUMERICAL CHOICE
        self.ratio_change = self.ratio_large / 2
        self.n_change = int(self.size * self.ratio_change)
        self.n_large_change = int(self.n_change * 0.2) + 1
        
        '''
        # large change
        large_portion_size = self.size // self.n_large_change
        large_change_startpts = np.array( np.array(range(0, self.n_large_change)) * large_portion_size / 4, dtype = 'int32')
        large_change_period = np.array([ int(random.uniform(0.5,0.6) * self.size) for _ in range(self.n_large_change)])
        offset = np.array([ int(random.uniform(0,1) * large_portion_size) for _ in range(self.n_large_change)])
        self.large_change_spans = [ (large_change_startpts[i]+offset[i], large_change_startpts[i]+offset[i]+large_change_period[i] ) for i in range(self.n_large_change)]
        # notice that the range, especially the end points is not guaranteed to fall within [0,size)
        
                
    def calculate_y(self):
        self.y = self.func1(self.t * self.freq) * self.scale + self.func1(self.t * self.freq) * self.scale + self.y_center
        
    def plot(self, formatStr):
        plt.plot(self.x, self.y, formatStr) 
        
    def add_anomaly(self, indexes, extent):
        # indexes: list of where to make new anomaly
        # extent: scalar, extent of outness
        a = random.uniform(0,1)
        b = random.uniform(1,2)
        self.y[indexes] = extent * np.array([random.uniform(a,b)*self.y[x] for x in indexes])
    
    def func_type(self, span, func):
        # TODO
        pass
    
    def frep_type(self, span):
        self.freq[span] = self.freq[span] * random.uniform(-2,2)
        self.calculate_y()
            
    def amp_type(self, span):
        self.scale[span] = self.scale[span] * random.uniform(-0.5,0.5)
        self.calculate_y()
            
    def y_center_type(self, span):
        self.y_center[span] = self.y_center[span] + random.uniform(-2,2)
        self.calculate_y()
        
    def phase_type(self, span):
        self.t[span] = self.t[span] + random.uniform(-np.pi,np.pi)
        self.calculate_y()
    
    def rand_type(self, span):
        # TODO
        pass
    
    
    def zero_type(self, span):
        self.y_center[span] = 0
        self.calculate_y()
    
    def generate_change_type(self, typeIndicator):
        switcher = {
            0: self.func_type,
            1: self.frep_type,
            2: self.amp_type,
            3: self.y_center_type,
            4: self.phase_type,
            5: self.rand_type,
            6: self.zero_type
        }
        return switcher[typeIndicator] # returns the functions

    def add_change(self, spanTuple, typeIndicator, func=None):
        # span: <--- a tuple of starting and ending points
        # typeIndictator: translated enum (int)
        # func: for func_type (typeIndicator == 0, NEW_FUNCTION_ONLY)
        span = np.array(range(spanTuple[0],spanTuple[1]))
        change_type = self.generate_change_type(typeIndicator)
        if (typeIndicator != 0):
            change_type(span)
        else:
            change_type(span,func)
        pass
    
    
    def allocate_variations(self):

        # seperate data into portions and apply change
        # best not to overlap
        # random choose for eadh self.small_change_spans
        # random choose for eadh self.large_change_spans
        population_small_change = [1,2,3,4,5,6,6,6,6,6,5,5,5, 5, 5, 4,4]
        self.small_change_indicators = random.sample(population_small_change,self.n_small_change)
        for i in range(len(self.small_change_indicators)):
            self.add_change(self.small_change_spans[i], self.small_change_indicators[i])
            
        population_large_change =  [1,2,3,4,5,6,6,1,6,6,5,5,4,4,2,2,2,6,6,1]
        self.large_change_indicators = random.sample(population_large_change,self.n_large_change)
        for i in range(len(self.large_change_indicators)):
            self.add_change(self.large_change_spans[i], self.large_change_indicators[i])
        #self.plot('r.-')
        
        # anomaly is randomly positioned
        #self.add_anomaly(self.large_anomaly_indexes, 3)
        #self.add_anomaly(self.small_anomaly_indexes, 1.5)
        #self.plot('bo-')

for i in range(95,100):
    temp = PatternGenerator(30)
    plt.figure()
    temp.plot('bo-')
    plt.savefig(str(i)+'.png')
    plt.close()
    np.save(str(i)+'.npy', temp.y)
#p1 = PatternGenerator(30)
#p2 = PatternGenerator(50)
#p3 = PatternGenerator(50)

# TODO 
# p2 = 
# combine them

# variation ONE: sudden change of frequency (anomaly)


# https://stackoverflow.com/questions/13628725/how-to-generate-random-numbers-that-are-different
# anomaly_index_in_t = random.sample(range(size), n_anomaly)
