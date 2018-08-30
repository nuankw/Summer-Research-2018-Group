'''
git clone https://github.com/TimeSynth/TimeSynth.git
For Nuan's Windows system:
run anaconda prompt as administrator
cd C:\Users\Nuan Wen\Desktop\Summer\Research\TimeSynth-master
python setup.py install
'''
# %%
import timesynth as ts
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
# %% Initializing TimeSampler
time_sampler = ts.TimeSampler(stop_time=200)
print(type(time_sampler))
print(time_sampler.__dict__)

# %% Sampling irregular time samples
irregular_time_samples = time_sampler.sample_irregular_time(num_points=500,keep_percentage=50)
print(type(irregular_time_samples))
print(irregular_time_samples.size)

# %% Initializing Sinusoidal signal
sinusoid = ts.signals.Sinusoidal(frequency=0.25)
print(type(sinusoid))
print(sinusoid.__dict__)

# %% Initializing Gaussian noise
white_noise = ts.noise.GaussianNoise(std=0.3)
print(type(white_noise))
print(white_noise.__dict__)

# %% Initializing TimeSeries class with the signal and noise objects
timeseries = ts.TimeSeries(sinusoid, noise_generator=white_noise)
print(type(timeseries))
print(timeseries.__dict__)

# %% Sampling using the irregular time samples
samples, signals, errors = timeseries.sample(irregular_time_samples)
print(type(samples))
print(type(signals))
print(type(errors))
print(samples)
print(signals)
print(errors)
