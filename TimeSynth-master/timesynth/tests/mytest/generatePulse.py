# %%
'''
git clone https://github.com/TimeSynth/TimeSynth.git
For Nuan's Windows system:
run anaconda prompt as administrator
cd C:\Users\Nuan Wen\Desktop\Summer\Research\TimeSynth-master
python setup.py install
'''
# %%
%matplotlib inline
import timesynth as ts
import matplotlib
import numpy as np
matplotlib.use('agg')
import matplotlib.pyplot as plt

def sinusoid_pattern(length, ftype, period, amplitude, std):
    time_sampler = ts.TimeSampler(stop_time=length)
    irregular_time_samples = time_sampler.sample_irregular_time(num_points=192,keep_percentage=100)
    sinusoid = ts.signals.Sinusoidal(amplitude=amplitude, frequency=1/period, ftype=ftype)
    white_noise = ts.noise.GaussianNoise(std=std)
    timeseries = ts.TimeSeries(sinusoid, noise_generator=white_noise)
    samples, signals, errors = timeseries.sample(irregular_time_samples)
    return samples

def add_periodic_change(time_series, times_of_change, magnitude):
    assert(times_of_change > 0)
    change_length = time_series.size // (times_of_change+1)
    for i in range(1, times_of_change, 2):
        print(time_series[i*change_length:(i+1)*change_length])
        time_series[i*change_length:(i+1)*change_length] += magnitude
        print(time_series[i*change_length:(i+1)*change_length])
    return np.copy(time_series)

def add_trend(time_series,trend_func):
    pass
print(list(range(0,3,2)))

# %% plot
f = plt.figure()
y1 = sinusoid_pattern(length=192,ftype=np.sin,period=24, amplitude=1, std=0.3)
plt.plot(np.arange(192), y1)

y2 = sinusoid_pattern(length=192,ftype=np.cos,period=6, amplitude=0.2, std=0.1)
plt.plot(np.arange(192), y2)

y3 = y1 + y2
plt.plot(np.arange(192), y3)

y4 = add_periodic_change(time_series=y3, times_of_change=1, magnitude = 3)
plt.plot(np.arange(192), y4)
