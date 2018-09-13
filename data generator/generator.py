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
        self.series_length = self.stop_time # this is correct! since starting point range from zero to stop_time  e.g. 0__1__2
        # when plotting, stop_time will not be shown.
        self.time_sampler = ts.TimeSampler(stop_time=self.stop_time)
        #self.time_samples = self.time_sampler.sample_irregular_time(num_points=self.stop_time + 1,keep_percentage=100) # + 1 here to fix the bug
        self.time_samples = self.time_sampler.sample_regular_time(num_points=self.stop_time+1) # + 1 here to fix the bug

        # initialize time series pattern
        self.period = period
        self.amplitude = amplitude
        self.std = std
        self.ftype = ftype
        self.signal_type = signal_type
        if signal_type == ts.signals.PseudoPeriodic:
            self.signal_generator = self.signal_type(amplitude=self.amplitude, frequency=1/self.period, ftype=self.ftype, ampSD=0, freqSD=24,)
        else:
            self.signal_generator = self.signal_type(amplitude=self.amplitude, frequency=1/self.period, ftype=self.ftype)
        self.white_noise_generator = None

        self.timeseries = ts.TimeSeries(signal_generator=self.signal_generator, noise_generator=self.white_noise_generator)
        # generate data
        self.samples, self.signals, self.errors = self.timeseries.sample(self.time_samples)


    def get_smooth_values(self): # less anomaly
        return self.signals

    def get_bumpy_values(self):  # more like real-life data
        return self.samples

    def set_smooth_values(self, new_signals): # less anomaly
        self.signals = new_signals

    def set_bumpy_values(self, new_samples):  # more like real-life data
        self.samples = new_samples

    def plot(self,plot_the_smoother_line = True, plot_start = None, plot_till=None, filename = None):
        if (plot_the_smoother_line):
            y = self.get_smooth_values()
        else:
            y = self.get_bumpy_values()
        f = plt.figure()
        if (plot_till==None):
            if (plot_start == None):
                plt.plot(np.arange(self.series_length+1), y) # to fix the same bug <-- no longer bug
            else:
                plt.plot(np.arange(self.series_length+1-plot_start), y[plot_start:])
        else:
            if (plot_start == None):
                plt.plot(np.arange(plot_till), y[:plot_till])
            else:
                plt.plot(np.arange(plot_till-plot_start), y[plot_start:plot_till])

        if (filename != None):
            plt.savefig(filename+".jpg")

    def move_to_above_zero(self):
        smooth_value_min = min(self.get_smooth_values())
        if (smooth_value_min < 0):
            self.set_smooth_values(self.get_smooth_values() - smooth_value_min)
        bumpy_value_min = min(self.get_bumpy_values())
        if (bumpy_value_min < 0):
            self.set_bumpy_values(self.get_bumpy_values() - bumpy_value_min)

    def move_all(self, bump_extent):
        self.set_smooth_values(self.get_smooth_values() + bump_extent)
        self.set_bumpy_values(self.get_bumpy_values() + bump_extent)

    def move(self, position_tuple_list, magnitude_list, bump_the_smoother_line = True, ramp_it = False):
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

    def multiply(self, position_tuple_list, magnitude_list, bump_the_smoother_line = True, ramp_it = False):
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
            orig_data[position_tuple_list[i][0]: position_tuple_list[i][1]] *= magnitude_list[i] # the only difference from move

        setFunc(orig_data)

        if (ramp_it):
            self.ramp(position_tuple_list, magnitude_list)

# TODO
    def ramp(self, position_tuple_list, magnitude_list):
        pass


def pattern_mixer(pattern_generator_1, pattern_generator_2,  position_tuple_list, mix_the_smoother_lines = True ):
    #### CAUTION: ONLY DO MIXER AT THE END OF PATTERN GENERATION ####
    #### attributes for each pattern_generator would NOT change except y values (smooth and bumpy) ####
    #### BETTER CALL pattern_mixer twice each time, let mix_the_smoother_lines = True, and False ####
    if (mix_the_smoother_lines):
        orig_data = pattern_generator_1.get_smooth_values()
        repeat_data = pattern_generator_2.get_smooth_values()[:-1]
        # use [:-1] to resolve the problem, will look deeper into it [solved] 0--1--2, length = 2, num_values = 3
        set_values = pattern_generator_1.set_smooth_values
    else:
        orig_data = pattern_generator_1.get_bumpy_values()
        repeat_data = pattern_generator_2.get_bumpy_values()[:-1]
        set_values = pattern_generator_1.set_bumpy_values

    pattern2_length = pattern_generator_2.series_length

    for i in range(len(position_tuple_list)):
        # now can handle situations that position_tuple_list is larger than pattern_generator_2's span
        span = position_tuple_list[i][1] - position_tuple_list[i][0]
        assert(span >= pattern2_length)
        diff = span - pattern2_length
        if (diff == 0):
            varied = 0
        else:
            varied = np.random.randint(diff)
        real_starting_point = position_tuple_list[i][0] + varied
        real_ending_point = real_starting_point + pattern2_length
        #print('\norig_data:\n', orig_data[real_starting_point:real_ending_point])
        #print('will be changed into:\n', repeat_data)
        orig_data[real_starting_point:real_ending_point] += repeat_data

    set_values(orig_data)
    return orig_data

def get_pulse_list(num, length_mean, length_std, amplitude, verbose=True, plot_the_smoother_line=False, fileName_start_with = None,):
    len_ndarray = np.random.normal(loc=length_mean, scale=length_std, size=num)
    len_ndarray = len_ndarray.astype(int)
    if verbose:
        print("len_ndarray: ",len_ndarray)
    pattern_list = []
    for i in range(num):
        len = len_ndarray[i]
        pattern = Pattern_generator(stop_time=len, period=len, amplitude=amplitude, std=length_std, ftype = np.sin, signal_type = ts.signals.Sinusoidal)
        pattern.move_to_above_zero()
        pattern_list.append(pattern)
        if verbose:
            pattern_list[i].plot(plot_the_smoother_line)
            if fileName_start_with != None:
                plt.savefig(fileName_start_with+"_"+str(i+1)+".jpg")
    return pattern_list

def day_list_2_end_start_tuple_list(day_list, morning_start_at = 0, night_end_at = 23 ):
    # note that morning_start_at and night_end_at can be greater than 24 or less then 0,
    # however need to take care of index bound via choise of days in the day_list
    start_end_tuple_list = []
    for i in day_list:
        start = i * 24 + morning_start_at
        end = i * 24 + night_end_at
        start_end_tuple_list.append((start,end))
    return start_end_tuple_list
