from generator import Pattern_generator, pattern_mixer, get_pulse_list, day_list_2_end_start_tuple_list
import timesynth as ts
import numpy as np
import math
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import random

random.seed(2333)
np.random.seed(2333)
'''   BASE   '''
# %matplotlib inline
window_size = 192
period_length = 24
N = 12500
n_periods_per_window = window_size // period_length
time_series_length = window_size * N
amp = 0.05
std = 0
# BASE_PATTERN = Pattern_generator(stop_time=time_series_length, period=24, amplitude=amp/2, std=std, ftype = np.sin, signal_type = ts.signals.Sinusoidal)
BASE_PATTERN = Pattern_generator(stop_time=time_series_length, period=24, amplitude=amp, std=std, ftype = np.sin, signal_type = ts.signals.PseudoPeriodic)
BASE_PATTERN.move_to_above_zero()
BASE_PATTERN.plot(plot_till=24,filename='./dataset/no change/img/base_pattern_24')
BASE_PATTERN.plot(plot_till=24*7,filename='./dataset/no change/img/base_pattern_24_7')
BASE_PATTERN.plot(plot_till=24*28,filename='./dataset/no change/img/base_pattern_24_28')

'''
设计 hours_in_day, days_in_week, days_in_month 三种起伏，每种各n个类似 ==> 改；one for each, since there is no noise within those patterns any more
'''

# n = 3 # num_choice_for_each_pattern
n = 1 # num_choice_for_each_pattern
# hours_in_day pattern
hd_amp = 0.1
hd_len_mean = 13
hd_len_std = 0
#hd_pattern_list = get_pulse_list(num=n, length_mean=hd_len_mean, length_std=hd_len_std, amplitude=amp, verbose=False, plot_the_smoother_line=True)
hd_pattern_list = get_pulse_list(num=n, length_mean=hd_len_mean, length_std=hd_len_std, amplitude=hd_amp, verbose=True, plot_the_smoother_line=True, fileName_start_with = "dataset/no change/img/hours_in_day_Pattern")

# ===> hours_in_day 起伏的对应位置
# hour_in_day: 7-22

hd_position_list = []
for i in range(time_series_length // 24):
    hd_position_list.append((i*24+7,i*24+22))

for i in range(n):
    print(type(hd_position_list))
    pattern_mixer(pattern_generator_1 = BASE_PATTERN, pattern_generator_2 = hd_pattern_list[i],  position_tuple_list = hd_position_list, mix_the_smoother_lines = True)


# days_in_week pattern
# weekday bump up
# days_in_week: Mon-Fri, fixed

dw_stop_time = 5*24
dw_period = 24
dw_amp = 0.5
dw_std = 0
dw_bump_extent = 0
dw_pattern_list = []
for i in range(n):
    dw_pattern_list.append(Pattern_generator(stop_time=dw_stop_time, period=dw_period, amplitude=dw_amp, std=dw_std, ftype = np.sin, signal_type = ts.signals.Sinusoidal))
    dw_pattern_list[i].move_to_above_zero()
    dw_pattern_list[i].move_all(bump_extent = dw_bump_extent)
    dw_pattern_list[i].plot(filename="dataset/no change/img/days_in_week_Pattern_"+str(i+1))


dw_position_list = []
for i in range(time_series_length // 24 // 7):
    dw_position_list.append((i*24*7+0*24, i*24*7+5*24))

for i in range(n):
    print(type(dw_position_list))
    pattern_mixer(pattern_generator_1 = BASE_PATTERN, pattern_generator_2 = dw_pattern_list[i],  position_tuple_list = dw_position_list, mix_the_smoother_lines = True)

# days_in_month pattern
dm_stop_time = 2*24
dm_period = 24
dm_amp = 1
dm_std = 0
dm_bump_extent = 0
dm_pattern_list = []
for i in range(n):
    dm_pattern_list.append(Pattern_generator(stop_time=dm_stop_time, period=dm_period, amplitude=dm_amp, std=dm_std, ftype = np.sin, signal_type = ts.signals.Sinusoidal))
    dm_pattern_list[i].move_to_above_zero()
    dm_pattern_list[i].move_all(bump_extent = dm_bump_extent)
    dm_pattern_list[i].plot(filename="dataset/no change/img/days_in_month_Pattern_"+str(i+1))

dm_position_list = []
for i in range(time_series_length // 24 // 28):
    dm_position_list.append((i*24*28+26*24, i*24*28+28*24))

for i in range(n):
    print(type(dw_position_list))
    pattern_mixer(pattern_generator_1 = BASE_PATTERN, pattern_generator_2 = dm_pattern_list[i],  position_tuple_list = dm_position_list, mix_the_smoother_lines = True)

BASE_PATTERN.plot(plot_the_smoother_line = True, plot_start =0, plot_till=24*30, filename='./dataset/no change/img/mixed_pattern_24_30')
BASE_PATTERN.plot(plot_the_smoother_line = True, plot_till=24*7, filename='./dataset/no change/img/mixed_pattern_24_7')
BASE_PATTERN.plot(plot_the_smoother_line = True, plot_till=24, filename='./dataset/no change/img/mixed_pattern_24')
BASE_PATTERN.plot(plot_the_smoother_line = True, filename='./dataset/no change/img/mixed_pattern_all')
smooth_ys = BASE_PATTERN.get_smooth_values()
np.save('dataset/no change/Dataset_3_normal.npy', smooth_ys)


'''

Variation 1: with anomaly

'''

num_days = time_series_length // 24
num_random_holidays = int(15 / 365 * num_days)
population = list(range(time_series_length))
random_holidays = random.sample(population=list(range(num_days)),k = num_random_holidays)
holidays_peak = random_holidays[0::2]
holidays_dent = random_holidays[1::2]

holidays_peak_position_list = day_list_2_end_start_tuple_list(holidays_peak, morning_start_at = 0, night_end_at = 24 )
holidays_peak_magni_list = (np.ones(len(holidays_peak))*4).tolist()
BASE_PATTERN.move(position_tuple_list=holidays_peak_position_list, magnitude_list=holidays_peak_magni_list, bump_the_smoother_line = True, ramp_it = False)

plot_spot = holidays_peak_position_list[np.random.randint(len(holidays_peak_position_list))][0]
BASE_PATTERN.plot(plot_the_smoother_line = True, plot_start = plot_spot - 24 * 10, plot_till = plot_spot + 24 * 20, filename='./dataset/with change/img/anomaly_peak_24_30')
BASE_PATTERN.plot(plot_the_smoother_line = True, plot_start = plot_spot - 24 * 3,  plot_till = plot_spot + 24 * 4, filename='./dataset/with change/img/anomaly_peak_24_7')
BASE_PATTERN.plot(plot_the_smoother_line = True, filename='./dataset/with change/img/anomaly_peak_all')


holidays_dent_position_list = day_list_2_end_start_tuple_list(holidays_dent, morning_start_at = 0, night_end_at = 24 )
holidays_dent_magni_list = (np.random.uniform(low=0.0, high=0.2, size=len(holidays_dent))).tolist()
BASE_PATTERN.multiply(position_tuple_list=holidays_dent_position_list, magnitude_list=holidays_dent_magni_list, bump_the_smoother_line = True, ramp_it = False)

plot_spot = holidays_dent_position_list[np.random.randint(len(holidays_dent_position_list))][0]
BASE_PATTERN.plot(plot_the_smoother_line = True, plot_start = plot_spot - 24 * 10, plot_till = plot_spot + 24 * 20, filename='./dataset/with change/img/anomaly_dent_24_30')
BASE_PATTERN.plot(plot_the_smoother_line = True, plot_start = plot_spot - 24 * 3,  plot_till = plot_spot + 24 * 4, filename='./dataset/with change/img/anomaly_dent_24_7')
BASE_PATTERN.plot(plot_the_smoother_line = True, filename='./dataset/with change/img/anomaly_all')

np.save("dataset/with change/anomaly_peak_start_end_timesteps.npy", np.array(holidays_peak_position_list))
np.save("dataset/with change/anomaly_dent_start_end_timesteps.npy", np.array(holidays_dent_position_list))

smooth_ys = BASE_PATTERN.get_smooth_values()
np.save('dataset/with change/Dataset_3_anomaly.npy', smooth_ys)


'''

Variation 2: with change

'''
num_changes = int(0.2 * num_days)
average_distance = time_series_length // num_changes # roughly about one change per month

# three types of change:
# a few days
# more than one week
# more than half month

population = list(range(num_days))
if (num_changes % 2 != 0):
    num_changes += 1
print("num_changes: ", num_changes)
list_chosen_x = random.sample(population=population, k = num_changes)
list_chosen_x.sort()
starting_points_list = (np.array(list_chosen_x[0::2]) * 24).tolist()
ending_points_list = (np.array(list_chosen_x[1::2]) * 24).tolist()
difference_ndarray = np.array(ending_points_list) - np.array(starting_points_list)
difference_list = difference_ndarray.tolist()

# preview
print("\nstarting_points_list[:10]:\n", starting_points_list[:10])
print("\nending_points_list[:10]:\n", ending_points_list[:10])
print("\ndifference_list[:10]:\n", difference_list[:10])

print("\nnumber of (start, end) pairs: ", len(difference_list))
print("average length:", np.mean(difference_ndarray)) # around 20
print("minimum length:", np.min(difference_ndarray)) # amost always 1
print("maximum length:", np.max(difference_ndarray)) # usually 120+, rarely over 200
print("ratio: ", sum(difference_list)/time_series_length) # around 0.5

# starting_points_list = starting_points_list[:10]
# ending_points_list = ending_points_list[:10]
# difference_list = difference_list[:10]

assert(len(starting_points_list) == len(ending_points_list) == len(difference_list))

# three types of change:
# 1. a few days' change in one week (7*24, 14 * 24)
# 2. more than one week's change in a month (14*24, 28*24)
# 3. more than half month's change in more than a month (28*24, )


position_short_changes = []
short_threshold = 7*24
short_change_duration = 4 * 24
position_med_changes = []
med_threshold = 14*24
med_change_duration = 5 * 24
position_long_changes = []
long_threshold = 28*24
long_change_duration = 15 * 24
for i in range(len(difference_list)):
    init_start_pts, init_end_pts = starting_points_list[i], ending_points_list[i]
    init_pos_distance = init_end_pts - init_start_pts
    assert(init_pos_distance == difference_list[i])
    if (difference_list[i] > long_threshold):
        assert(init_pos_distance > short_change_duration)
        real_start_pts = np.random.randint(low = init_start_pts, high = init_start_pts + init_pos_distance - long_change_duration)
        real_end_pts = real_start_pts + long_change_duration
        assert(real_end_pts <= init_end_pts)
        this_pos_tuple = (real_start_pts,real_end_pts)
        position_long_changes.append(this_pos_tuple)

    elif (difference_list[i] > med_threshold):
        assert(init_pos_distance > short_change_duration)
        real_start_pts = np.random.randint(low = init_start_pts, high = init_start_pts + init_pos_distance - med_change_duration)
        real_end_pts = real_start_pts + med_change_duration
        assert(real_end_pts <= init_end_pts)
        this_pos_tuple = (real_start_pts,real_end_pts)
        position_med_changes.append(this_pos_tuple)

    elif (difference_list[i] > short_threshold):
        assert(init_pos_distance > short_change_duration)
        real_start_pts = np.random.randint(low = init_start_pts, high = init_start_pts + init_pos_distance - short_change_duration)
        real_end_pts = real_start_pts + short_change_duration
        assert(real_end_pts <= init_end_pts)
        this_pos_tuple = (real_start_pts,real_end_pts)
        position_short_changes.append(this_pos_tuple)

print("\nposition_short_changes[:20]:\n", position_short_changes[:20])
print("\nposition_med_changes[:20]:\n", position_med_changes[:20])
print("\nposition_long_changes[:20]:\n", position_long_changes[:20])

print()
print(len(position_short_changes))
print(len(position_med_changes))
print(len(position_long_changes))
print(len(position_short_changes) + len(position_med_changes) + len(position_long_changes))
print()

num_short_changes = len(position_short_changes)
num_med_changes = len(position_med_changes)
num_long_changes = len(position_long_changes)

print(np.mean(BASE_PATTERN.get_smooth_values()))
print(np.median(BASE_PATTERN.get_smooth_values()))
print(np.percentile(BASE_PATTERN.get_smooth_values(), 92.5))
print(np.max(BASE_PATTERN.get_smooth_values()))
print(np.min(BASE_PATTERN.get_smooth_values()))

bump_extent = np.percentile(BASE_PATTERN.get_smooth_values(), 92.5)

BASE_PATTERN.move(position_tuple_list=position_short_changes, magnitude_list=np.ones(num_short_changes)*bump_extent, bump_the_smoother_line = True, ramp_it = False)
BASE_PATTERN.move(position_tuple_list=position_med_changes, magnitude_list=np.ones(num_med_changes)*bump_extent, bump_the_smoother_line = True, ramp_it = False)
BASE_PATTERN.move(position_tuple_list=position_long_changes, magnitude_list=np.ones(num_long_changes)*bump_extent, bump_the_smoother_line = True, ramp_it = False)


plot_spot = position_short_changes[np.random.randint(len(position_short_changes))][0]
BASE_PATTERN.plot(plot_the_smoother_line = True, plot_start = plot_spot - 24 * 5, plot_till = plot_spot + 24 * 9, filename='./dataset/with change/img/change_short_2weeks')

plot_spot = position_med_changes[np.random.randint(len(position_med_changes))][0]
BASE_PATTERN.plot(plot_the_smoother_line = True, plot_start = plot_spot - 24 * 10,  plot_till = plot_spot + 24 * 20, filename='./dataset/with change/img/change_med_1month')

plot_spot = position_long_changes[np.random.randint(len(position_long_changes))][0]
BASE_PATTERN.plot(plot_the_smoother_line = True, plot_start = plot_spot - 24 * 20,  plot_till = plot_spot + 24 * 40, filename='./dataset/with change/img/change_long_2months')


BASE_PATTERN.plot(plot_the_smoother_line = True, filename='./dataset/with change/img/change_anomaly_all')

position_short_changes = np.array(position_short_changes)
np.save("dataset/with change/change_short_start_end_timesteps.npy",position_short_changes)
position_med_changes = np.array(position_med_changes)
np.save("dataset/with change/change_med_start_end_timesteps.npy",position_med_changes)
position_long_changes = np.array(position_long_changes)
np.save("dataset/with change/change_long_start_end_timesteps.npy",position_long_changes)

smooth_ys = BASE_PATTERN.get_smooth_values()
np.save('dataset/with change/Dataset_3_change_anomaly.npy', smooth_ys)
