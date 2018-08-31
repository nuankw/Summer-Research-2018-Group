import timesynth as ts
import numpy as np
import math
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import random


'''

STEPS:

Goal:
    Two lenthy time series, one wih change and one without

A. Implement the following functionality
   [what this piece of python code is all about]
        1. be able to generate long, flat time series
           with small, constant variations (pseudoperiodic maybe)
        2. be able to generate short, highly varied time series
        3. be able to bump up/down y values (list of tuples)
        4. be able to add ramp-ups and ramp-downs
        5. be able to mix several time series

B. Create testing data for research (the main.py)

'''




class Pattern_generator:

    def __init__(self, series_length, period, amplitude, std, ftype = np.sin, signal_type = ts.signals.Sinusoidal):
        # initialize time steps
        self.time_sampler = ts.TimeSampler(stop_time=series_length)
        self.irregular_time_samples = time_sampler.sample_irregular_time(num_points=series_length,keep_percentage=100)
        # initialize time series pattern
        self.signal_generator = signal_type(amplitude=amplitude, frequency=1/period, ftype=ftype)
        self.white_noise_generator = ts.noise.GaussianNoise(std=std)
        self.timeseries = ts.TimeSeries(signal_generator=self.signal_generator, noise_generator=self.white_noise_generator)
        # generate data
        self.samples, self.signals, self.errors = self.timeseries.sample(irregular_time_samples)

    def getSamples(self): # less anomaly
        return self.signals

    def getSignals(self):  # more like real-life data
        return self.samples

    # ASSUMING that





# class Pattern_mixer:
#
#     def __init__(self,  ):
#         pass
