# %%

# %%
%matplotlib inline
import timesynth as ts
import numpy as np
import matplotlib.pyplot as plt

def sinusoid_pattern(length, ftype, period, amplitude, std):
    time_sampler = ts.TimeSampler(stop_time=length)
    irregular_time_samples = time_sampler.sample_irregular_time(num_points=192,keep_percentage=100)
    sinusoid = ts.signals.Sinusoidal(amplitude=amplitude, frequency=1/period, ftype=ftype)
    white_noise = ts.noise.GaussianNoise(std=std)
    timeseries = ts.TimeSeries(sinusoid, noise_generator=white_noise)
    samples, signals, errors = timeseries.sample(irregular_time_samples)
    return signals # 整洁
    #return samples # 更乱

def add_periodic_change(time_series, period, change_point_list, magnitude):
    # ASSUMPTION: the begining and ending point will not become points of change, and changes happen at the begining of each period
    num_change = len(change_point_list)
    assert(num_change < time_series.size / period)
    for i in change_point_list:
        #print(time_series[int((i+0.25)*period):int((i+1.25)*period)])
        time_series[i*period:(i+1)*period] += magnitude
        #print(time_series[int((i+0.25)*period):int((i+1.25)*period)])
    return time_series

def add_trend(time_series,trend_func):
    pass
f = plt.figure()
y1 = sinusoid_pattern(length=192,ftype=np.sin,period=48, amplitude=0.1, std=0.1)
plt.plot(np.arange(192), y1)
plt.plot(np.arange(192), np.abs(y1))

y2 = sinusoid_pattern(length=192,ftype=np.cos,period=7, amplitude=0.02, std=0)
# plt.plot(np.arange(192), y2)

y3 = y1 + y2
#y3 = np.abs(y3)
#y3 = np.abs(y1)
# plt.plot(np.arange(192), y3)

# print(y3)

y4 = add_periodic_change(time_series=y3, period=96, change_point_list=[1], magnitude=3)
y_min = np.min(y4)
y4 -= y_min
#plt.plot(np.arange(20), y4[:20], 'b-')
plt.plot(np.arange(192), y4, 'b')
