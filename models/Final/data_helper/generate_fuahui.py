import csv
import numpy as np

def norm(data):
	aver = np.true_divide(np.sum(data), data.shape[0]*data.shape[1])
	stddev = np.std(data)
	data_norm = np.true_divide((data-aver), stddev)
	return (data_norm)

def gen_hour (num_days):
    hour_temp = np.concatenate([np.arange(1,25).reshape([1,24]) for i in range(num_days)], axis=0)
    return hour_temp

def gen_dat(num_days, start):
    day_temp = np.concatenate([np.arange(1,8) for i in range(35)],axis=0)[start-1:start-1+num_days]
    day_temp = np.concatenate([day_temp.reshape([num_days, 1]) for i in range(24)], axis=1)
    return day_temp

num_days_list = np.array([31,30,31,31,30,31,30,31])
hours_all = gen_hour(np.sum(num_days_list))
hours_all = norm(hours_all)
days_all = gen_dat(np.sum(num_days_list), 1)
days_all = norm(days_all)

day_may = np.zeros([31, 24])
day_jun = np.zeros([30, 24])
day_jul = np.zeros([31, 24])
day_aug = np.zeros([31, 24])
day_sep = np.zeros([30, 24])
day_oct = np.zeros([31, 24])
day_nov = np.zeros([30, 24])
day_dec = np.zeros([31, 24])

hour_may = np.zeros([31, 24])
hour_jun = np.zeros([30, 24])
hour_jul = np.zeros([31, 24])
hour_aug = np.zeros([31, 24])
hour_sep = np.zeros([30, 24])
hour_oct = np.zeros([31, 24])
hour_nov = np.zeros([30, 24])
hour_dec = np.zeros([31, 24])

day_list = [day_may, day_jun, day_jul, day_aug, day_sep, day_oct, day_nov, day_dec]
hour_list = [hour_may, hour_jun, hour_jul, hour_aug, hour_sep, hour_oct, hour_nov, hour_dec]

for i, day in enumerate(day_list):
    if i==0:
        day_list[i] = days_all[0:num_days_list[i]]
        hour_list[i] = hours_all[0:num_days_list[i]]
    else:
        day_list[i] = days_all[np.sum(num_days_list[:i]):np.sum(num_days_list[:i+1])]
        hour_list[i] = hours_all[np.sum(num_days_list[:i]):np.sum(num_days_list[:i+1])]



print (hour_list[0].shape)


with open("../data/huawei/hour.csv","w") as f:
	writer = csv.writer(f, delimiter=',')
	writer.writerows(hours_all.tolist())
	print (hours_all.shape)
	print (type(hours_all))

with open("../data/huawei/day.csv","w") as f:
	writer = csv.writer(f, delimiter=',',)
	writer.writerows(days_all.tolist())
