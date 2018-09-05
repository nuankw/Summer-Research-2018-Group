import csv
#import tensorflow as tf
import numpy as np
import glob, os

def num_2_onehot(index, length):
	onehot = [0 for i in range(length)]
	onehot[index]=1
	return onehot

def norm(data):
	aver = np.true_divide(np.sum(data), data.shape[0]*data.shape[1])
	stddev = np.std(data)
	data_norm = np.true_divide((data-aver), stddev)
	return (data_norm)

def read_data(filename):
	with open(filename) as f:
		reader = csv.reader(f,delimiter=',')
		data = np.array(list(reader))
		start_time = data[1, 0]
		end_time = data[-1, 0]
		data = data[1: , 1:]
		print ("data.shape: ",data.shape)
	return data, start_time, end_time

def read_covar(filename):
	with open(filename) as f:
		reader = csv.reader(f,delimiter=',')
		data = np.array(list(reader), dtype=np.float32)
	return data

def read(data_file_list):
	#read covariates
	hour_of_the_day = read_covar("../data/huawei/hour.csv")
	day_of_the_week = read_covar("../data/huawei/day.csv")

	print (hour_of_the_day.shape)
	print (day_of_the_week.shape)
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

	num_days_list = np.array([31,30,31,31,30,31,30,31])
	day_list = [day_may, day_jun, day_jul, day_aug, day_sep, day_oct, day_nov, day_dec]
	hour_list = [hour_may, hour_jun, hour_jul, hour_aug, hour_sep, hour_oct, hour_nov, hour_dec]

	for i, day in enumerate(day_list):
	    if i==0:
	        day_list[i] = day_of_the_week[0:num_days_list[i]]
	        hour_list[i] = hour_of_the_day[0:num_days_list[i]]
	    else:
	        day_list[i] = day_of_the_week[np.sum(num_days_list[:i]):np.sum(num_days_list[:i+1])]
	        hour_list[i] = hour_of_the_day[np.sum(num_days_list[:i]):np.sum(num_days_list[:i+1])]
	#read data
	data_cov_tuple_list = []
	for data_file in data_file_list:
		data, start_time, end_time = read_data(data_file)
		num_series = data.shape[1]
		start_month = int(start_time[-14:-12])
		start_day = int(start_time[-11:-9])
		end_month = int(end_time[-14:-12])
		end_day = int(end_time[-11:-9])

		start_hour = int(start_time[-8:-6])
		end_hour = int(end_time[-8:-6])
		day_cov, hour_cov = choose_cov(num_series, hour_list, day_list, start_month, start_day, end_month, end_day, start_hour, end_hour)
		data_cov_tuple_list.append((data, day_cov,hour_cov ))
	return data_cov_tuple_list


def choose_cov(num_series, hour_list, day_list, start_month, start_day, end_month, end_day, start_hour, end_hour):
	month_list = np.arange(start_month-5, end_month+1-5).tolist()
	day_cov_list = []
	hour_cov_list = []
	for month in month_list:
		if month == month_list[0]:
			day_cov_list.append(day_list[month][start_day-1:])
			hour_cov_list.append(hour_list[month][start_day-1:])
		elif month == month_list[-1]:
			day_cov_list.append(day_list[month][:end_day])
			hour_cov_list.append(hour_list[month][:end_day])
		else:
			day_cov_list.append(day_list[month][:])
			hour_cov_list.append(hour_list[month][:])
	day_cov = np.concatenate(day_cov_list, axis=0)
	day_cov = day_cov.flatten() #shape = [69*24,]
	day_cov_length = day_cov.shape[0]
	day_cov = day_cov[start_hour-1:day_cov_length-24+end_hour]
	day_cov = np.concatenate([np.expand_dims(day_cov, axis=1) for i in range(num_series)], axis=1)
	print (day_cov.shape)
	hour_cov = np.concatenate(hour_cov_list, axis=0)
	hour_cov = hour_cov.flatten()#shape = [69*24,]
	hour_cov_length = hour_cov.shape[0]
	hour_cov = hour_cov[start_hour-1:hour_cov_length-24+end_hour]
	hour_cov = np.concatenate([np.expand_dims(hour_cov, axis=1) for i in range(num_series)], axis=1)
	print (hour_cov.shape)
	return day_cov, hour_cov



def prepare():
	num_covar = 2
	encode_length = 168
	decode_length = 24
	num_series=10 #
	batch_size=64
	embedding_output_size= 5 ####
	hidden_unit=10 #40
	num_layer=3
	window_size = encode_length+decode_length

	param = np.array([num_covar,
	encode_length,
	decode_length ,
	num_series, #
	embedding_output_size, #5
	hidden_unit, #40
	num_layer,
	window_size])

	#read data

	shift_train_data = [] #data + 3 covariates for each window
	shift_train_onehot = [] #onehot vectors indicating which serie
	v = []	# vi for each window
	shift_train_label = [] #label for each window

	#file_list = ["./data/processed/0201_"+str(i)+".csv" for i in range(100)]
	file_list = []
	os.chdir("../data/huawei/processed/")
	for file in glob.glob("*.csv"):
		file_list.append("../data/huawei/processed/"+file)
		#print ("./data/processed/"+file)
	os.chdir("../../../data_helper/")
	print (len(file_list))
	data_cov_tuple_list = read(file_list)
	for data_cov_tuple in data_cov_tuple_list:
		data, day_of_the_week, hour_of_the_day = data_cov_tuple
		num_window = (data.shape[0]-encode_length)//decode_length -1 # for each time serie (without padding)
		#num_window = 40
		#print ("num_window: ", num_window)
		for serie in range(num_series):	 # loop through all time series
			#print (serie)
			for i in range(num_window):
			#for i in range(1060,1060+num_window):  # loop through all windows in one time serie
				#computing Vi for each window
				vi = np.sum(np.array(data[i*24:i*24+window_size,serie], dtype = np.float64), axis = 0)  #data[i:i+window_size,serie] is a single window
				'''
				if vi ==0:
					break
				'''
				vi = np.true_divide(vi, window_size)+1
				#vi = vi.reshape([1, num_series]) #shape=[num_series, 1]
				temp_shift_train_data = np.true_divide(np.array(data[i*24:i*24+window_size,serie], dtype = np.float64),vi ).reshape([1,window_size])
				#print ("temp_shift_train_data.shape: ",temp_shift_train_data.shape)
				temp_hour = hour_of_the_day[i*24:i*24+window_size,serie].reshape([1,window_size])
				temp_day = day_of_the_week[i*24:i*24+window_size,serie].reshape([1,window_size])
				temp_train = np.concatenate([temp_shift_train_data,temp_hour,temp_day]).transpose() #shape = [ window_size,4]
				#print ("temp_train.shape: ", temp_train.shape)
				shift_train_data.append(temp_train) #
				shift_train_onehot.append(num_2_onehot(serie, num_series))
				v.append(vi)
				#No scaling for label
				shift_train_label.append((data[i*24+1:i*24+window_size+1,serie]))

	#shape: [, window_size,3]
	shift_train_data = np.array(shift_train_data)
	#shape: [, num_series]
	shift_train_onehot = np.array(shift_train_onehot)
	#shape:[, 1]
	v = np.expand_dims(np.array(v), axis=1)
	#shape: [, window_size]
	shift_train_label = np.array(shift_train_label)

	print ("shift_train_data.shape:", shift_train_data.shape)
	print ("shift_train_onehot.shape: ",shift_train_onehot.shape)
	print ("v.shape: ",v.shape)
	print ("shift_train_label.shape:", shift_train_label.shape)
	##### permutation ####
	num_window_all = shift_train_data.shape[0]
	indexs = np.arange(num_window_all)
	indexs_pred = np.random.choice(num_window_all, num_window_all//10 )
	print ("indexs_pred.shape: ", indexs_pred.shape) #[64,]

	for ele_to_remove in indexs_pred:
		for j in range(ele_to_remove-5,ele_to_remove+6): #从ele_to_remove-5 至 ele_to_remove+6 都与 ele_to_remove有重叠
			try: indexs.remove(j)
			except:	pass

	print ("indexs.length: ", len(indexs))
	np.random.shuffle(indexs)
	np.random.shuffle(indexs_pred)
	num_row = (len(indexs))//batch_size
	num_row_pred = len(indexs_pred)//batch_size #suppose to be 704//64 = 11
	indexs_list=[]
	indexs_pred_list=[]
	for i in range(num_row):
		indexs_list.append(indexs[i*batch_size:(i+1)*batch_size])
		#if i==(num_row-1): indexs_list.append(indexs[(i+1)*batch_size:])
		if i==(num_row-1): pass # keep batch_size = 64
	indexs_list = np.array(indexs_list)

	for i in range(num_row_pred):
		indexs_pred_list.append(indexs_pred[i*batch_size:(i+1)*batch_size])
		#if i==(num_row_pred-1): indexs_pred_list.append(indexs_pred[(i+1)*batch_size:])
	indexs_pred_list = np.array(indexs_pred_list)
	return (shift_train_data,
			shift_train_onehot,
			v,
			shift_train_label,
			param,
			indexs_list,
			indexs_pred_list)


(shift_train_data,
shift_train_onehot,
v,
shift_train_label,
param,
indexs_list,
indexs_pred_list)=prepare()

np.save("../data/huawei/shift_train_data.npy",shift_train_data)
np.save("../data/huawei/shift_train_onehot.npy",shift_train_onehot)
np.save("../data/huawei/v.npy",v)
np.save("../data/huawei/shift_train_label.npy",shift_train_label)
np.save("../data/huawei/param.npy", param)
np.save("../data/huawei/indexs_list.npy", indexs_list)
np.save("../data/huawei/indexs_pred_list.npy", indexs_pred_list)
