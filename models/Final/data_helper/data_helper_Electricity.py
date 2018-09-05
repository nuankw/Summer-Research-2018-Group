'''
Chen Song, Nuan Wen, Yunkai Zhang 08/23/2018
packs all data with covariants into windows of size 192,
'''

from __future__ import division
import csv
import numpy as np

def read_data(filename):
	with open(filename) as f:
		reader = csv.reader(f,delimiter=',')
		data = np.array(list(reader))
	return data[1: , 1:]

def read_covar(filename):
	with open(filename) as f:
		reader = csv.reader(f,delimiter=',')
		data = np.array(list(reader), dtype=np.float32)
	return data

#create a list of zeros except there is a one at index
#example: (2, 3) --> [0, 0, 1]
def num_2_onehot(index, length):
	onehot = [0 for i in range(length)]
	onehot[index] = 1
	return onehot

def norm(data):
	aver = np.average(data)
	stddev = np.std(data)
	if stddev == 0:
		return np.zeros(data.shape)
	data_norm = (data-aver)/stddev
	return data_norm

def prepare():
	num_covar = 3
	encode_length = 168
	decode_length = 24
	num_series = 370
	batch_size = 64
	embedding_output_size = 20
	hidden_unit = 40
	num_layer = 3
	window_size = encode_length+decode_length
	param = (num_covar, encode_length, decode_length, num_series,
	embedding_output_size, hidden_unit, num_layer, window_size)

	#n, dim = 192, 1  # number of samples, dimension
	#model = "rbf"  # "l1", "rbf", "linear", "normal", "ar"


	age = norm(read_covar("../data/Electricity/age.csv")[0])
	data = read_data('../data/Electricity/electricity_hourly.csv')
	day_of_the_week = norm(read_covar("../data/Electricity/day_of_the_week.csv")[0])
	hour_of_the_day = norm(read_covar("../data/Electricity/hour_of_the_day.csv")[0])

	shift_train_data = [] #data + 3 covariates for each window
	shift_train_onehot = [] #onehot vectors indicating series number
	v = []	# vi for each window
	shift_train_label = [] #label for each window
	#shift_train_pvalue = [] #change points for each window

	num_window = (data.shape[0]-encode_length)//decode_length -1 # for each time series (without padding)
	start_train = 0
	for series in range(num_series):	 # loop through all time series
		for i in range(start_train, start_train+num_window):  # loop through all windows in one time series
			#computing Vi for each window, skip 24 time steps each time,
			#gather window by series, and let v be average of the window plus one
			#change_points = np.zeros(192)
			current_window = np.array(data[i*24:i*24+window_size, series], dtype = np.float64)
			#algo = rpt.Window(width=48, model=model).fit(current_window[:-24])
			sigma = np.std(current_window)
			#my_bkps = algo.predict(pen=2*np.log(n)*dim*sigma**2)
			#for j in my_bkps:
			#	change_points[j-1] = 1
			vi = np.average(current_window)+1
			shift_train_data.append(np.stack([current_window/vi, age[i*24:i*24+window_size],
			hour_of_the_day[i*24:i*24+window_size], day_of_the_week[i*24:i*24+window_size]], axis = 1)) #shape=[window_size,4]
			shift_train_onehot.append(num_2_onehot(series, num_series))
			v.append(vi)
			shift_train_label.append((data[i*24+1:i*24+window_size+1, series]))#No scaling for label
			#shift_train_pvalue.append(change_points)

	#shape: [num_window*num_series(1453*370), window_size,4]
	shift_train_data = np.array(shift_train_data)
	#shape: [num_window*num_series(1453*370), num_series]
	shift_train_onehot = np.array(shift_train_onehot)
	#shape:[num_window*num_series(1453*370) , 1]
	v = np.array(v).reshape([(num_window)*num_series , 1])
	#shape: [num_window*num_series(1453*370), window_size]
	shift_train_label = np.array(shift_train_label)

	#shift_train_pvalue = np.array(shift_train_pvalue)
	#print ("shift_train_pvalue.shape: ", shift_train_pvalue.shape)

	##### permutation ####
	indexs = np.arange(num_window*num_series)
	num_pred = 64
	#take the middle and last point for each series (off by one each time to prevent out
	#of bound error) and make the total size divisible by the batch size
	indexs_pred = [indexs[i*(num_window//2-1)] for i in range(num_series*2)]
	indexs_pred = indexs_pred[0:704]

	#remove all training samples that could overlap with the test samples
	for ele_to_remove in indexs_pred:
		for j in range(ele_to_remove-5,ele_to_remove+6):
			try: indexs.remove(j)
			except:	pass

	### shuffle ###
	np.random.shuffle(indexs)
	np.random.shuffle(indexs_pred)
	num_row = (len(indexs))//batch_size
	num_row_pred = len(indexs_pred)//batch_size #should be 704//64 = 11
	indexs_list=[]
	indexs_pred_list=[]
	for i in range(num_row):
		indexs_list.append(indexs[i*batch_size:(i+1)*batch_size])
		if i==(num_row-1): pass
	for i in range(num_row_pred):
		indexs_pred_list.append(indexs_pred[i*batch_size:(i+1)*batch_size])
		
	np.save("../data/Electricity/elect_pre_train_data", shift_train_data)
	np.save("../data/Electricity/elect_train_onehot", shift_train_onehot)
	np.save("../data/Electricity/elect_train_v", v)
	np.save("../data/Electricity/elect_train_label", shift_train_label)
	np.save("../data/Electricity/elect_train_param", param)
	np.save("../data/Electricity/elect_train_index", indexs_list)
	np.save("../data/Electricity/elect_train_pred_index", indexs_pred_list)
	#np.save("./data/elect_train_p_value", shift_train_pvalue)
	return

'''for testing purposes'''
prepare()
