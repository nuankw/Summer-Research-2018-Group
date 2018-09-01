import timesynth as ts
import numpy as np
import math
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import random

#%matplotlib inline

'''
Goal:
    Two lenthy time series, one wih change and one without

STEPS:

A. Implement the following functionality
   [what this piece of python code is all about]
        1. be able to generate long, flat time series
           with small, constant variations (pseudoperiodic maybe)
        2. be able to generate short, highly varied time series
        3. be able to bump up/down y values given positions and magnitudes
        4. be able to add ramp-ups and ramp-downs [left for later]
           note to self: basiclly times the beginning and ending of bumping
                         with a sequence of incresing numbers
        5. be able to mix several time series

B. Create testing data for research (the main.py)

'''

class Pattern_generator:

    def __init__(self, stop_time, period, amplitude, std, ftype = np.sin, signal_type = ts.signals.Sinusoidal):
        # initialize time steps
        self.stop_time = stop_time
        self.series_length = self.stop_time
        self.time_sampler = ts.TimeSampler(stop_time=self.stop_time)
        self.irregular_time_samples = self.time_sampler.sample_irregular_time(num_points=self.series_length,keep_percentage=100)

        # initialize time series pattern
        self.period = period
        self.amplitude = amplitude
        self.std = std
        self.ftype = ftype
        self.signal_type = signal_type
        self.signal_generator = self.signal_type(amplitude=self.amplitude, frequency=1/self.period, ftype=self.ftype)
        self.white_noise_generator = ts.noise.GaussianNoise(std=self.std)
        self.timeseries = ts.TimeSeries(signal_generator=self.signal_generator, noise_generator=self.white_noise_generator)
        # generate data
        self.samples, self.signals, self.errors = self.timeseries.sample(self.irregular_time_samples)

    def get_smooth_values(self): # less anomaly
        return self.signals

    def get_bumpy_values(self):  # more like real-life data
        return self.samples

    def set_smooth_values(self, new_signals): # less anomaly
        self.signals = new_signals

    def set_bumpy_values(self, new_samples):  # more like real-life data
        self.samples = new_samples

    def plot(self, plot_the_smoother_line = True):
        if (plot_the_smoother_line):
            y = self.get_smooth_values()
        else:
            y = self.get_bumpy_values()
        f = plt.figure()
        plt.plot(np.arange(self.series_length), y)

    def bump_to_above_zero(self):
        smooth_value_min = min(self.get_smooth_values())
        if (smooth_value_min < 0):
            self.set_smooth_values(self.get_smooth_values() - smooth_value_min)
        bumpy_value_min = min(self.get_bumpy_values())
        if (bumpy_value_min < 0):
            self.set_bumpy_values(self.get_bumpy_values() - bumpy_value_min)



    def bump(self, position_tuple_list, magnitude_list, bump_the_smoother_line = True, ramp_it = False):
        # position_tuple_list take a list of (starting, ending points)
        assert (len(position_tuple_list) == len(magnitude_list))
        if (bump_the_smoother_line):
            orig_data = self.get_smooth_values()
            setFunc = self.set_smooth_values
        else:
            orig_data = self.get_bumpy_values()
            setFunc = self.set_bumpy_values


        for i in range(len(position_tuple_list)):
            assert(len(position_tuple_list[i]) == 2)
            orig_data[position_tuple_list[i][0]: position_tuple_list[i][1]] += magnitude_list[i]

        setFunc(orig_data)

        if (ramp_it):
            self.ramp(position_tuple_list, magnitude_list)


''' comment this line for test
# simple test
PG1 = Pattern_generator(series_length=10000, period=24, amplitude=1, std=0.1, ftype = np.sin, signal_type = ts.signals.Sinusoidal)
ptl = [(2000,4000)]
ml = [3]
PG1.set_smooth_values(bump(PG1, position_tuple_list = ptl, magnitude_list = ml, bump_the_smoother_line = True))
PG1.plot()
# '''

def pattern_mixer(pattern_generator_1, pattern_generator_2,  position_tuple_list, mix_the_smoother_lines = True ):
    if (mix_the_smoother_lines):
        orig_data = pattern_generator_1.get_smooth_values()
        repeat_data = pattern_generator_2.get_smooth_values()
    else:
        orig_data = pattern_generator_1.get_bumpy_values()
        repeat_data = pattern_generator_2.get_bumpy_values()

    for i in range(len(position_tuple_list)):
        orig_data[position_tuple_list[i][0]:position_tuple_list[i][1]] += repeat_data

    return orig_data

def get_pulse_list(num, length_mean, length_std, amplitude, verbose=True):
    len_ndarray = np.random.normal(loc=length_mean, scale=length_std, size=num)
    len_ndarray = len_ndarray.astype(int)
    if verbose:
        print("len_ndarray: ",len_ndarray)
    pattern_list = []
    for i in range(n):
        len = len_ndarray[i]
        scale = 0.8 # for amplitude, according to observations
        pattern = Pattern_generator(stop_time=len+1, period=(len+1)*2, amplitude=amplitude*scale, std=std, ftype = np.sin, signal_type = ts.signals.Sinusoidal)
        pattern.bump_to_above_zero()
        pattern_list.append(pattern)
        if verbose:
            pattern_list[i].plot(False)
    return pattern_list
