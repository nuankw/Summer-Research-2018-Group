import csv
#import tensorflow as tf 
import numpy as np


def read_data(filename):
	
	with open(filename) as f:
		reader = csv.reader(f,delimiter=',')
		data = np.array(list(reader))
		data = data[1: , 1:]
		#print ("data.shape: ",data.shape)
	return data

def read_covar(filename):
	with open(filename) as f:
		reader = csv.reader(f,delimiter=',')
		data = np.array(list(reader), dtype=np.float32)
		data=data.transpose()
		#print (filename, ".shape: ",data.shape)
		#print (filename,"[:,0]: ",(data[:,0]))
	return data

def num_2_onehot(index, length):
	onehot = [0 for i in range(length)]
	onehot[index]=1
	return onehot

def norm(data): 
	aver = np.true_divide(np.sum(data), data.shape[0]*data.shape[1])
	stddev = np.std(data)
	data_norm = np.true_divide((data-aver), stddev)
	return (data_norm)

def prepare():
	num_covar = 3
	encode_length = 168
	decode_length = 24
	num_series=370 #370
	batch_size=64
	embedding_output_size=20 #20
	hidden_unit=40 #40
	num_layer=3
	window_size = encode_length+decode_length



	param = (num_covar,
	encode_length,
	decode_length ,
	num_series, #370
	
	embedding_output_size, #20
	hidden_unit, #40
	num_layer,
	window_size)
	
	#read data
	data=read_data('./data/electricity_hourly.csv')
	#read covariates
	age = read_covar("./data/age.csv")
	day_of_the_week = read_covar("./data/day_of_the_week.csv")
	hour_of_the_day = read_covar("./data/hour_of_the_day.csv")
	print("shapes:")
	print (data.shape)
	print (age.shape)
	print (day_of_the_week.shape)
	print (hour_of_the_day.shape)
	
	


	shift_train_data = [] #data + 3 covariates for each window
	shift_train_onehot = [] #onehot vectors indicating which serie
	v = []	# vi for each window
	shift_train_label = [] #label for each window
	
	
	num_window = (data.shape[0]-encode_length)//decode_length -1 # for each time serie (without padding)
	num_window = 40
	print ("num_window: ", num_window)
	for serie in range(num_series):	 # loop through all time series
		#print (serie)
		### Need to fix
		#for i in range(num_window):
		for i in range(1060,1060+num_window):  # loop through all windows in one time serie
		
			#computing Vi for each window 
			
			vi = np.sum(np.array(data[i*24:i*24+window_size,serie], dtype = np.float64), axis = 0)  #data[i:i+window_size,serie] is a single window
			vi = np.true_divide(vi, window_size)+1
			#vi = vi.reshape([1, num_series]) #shape=[num_series, 1]
			temp_shift_train_data = np.true_divide(np.array(data[i*24:i*24+window_size,serie], dtype = np.float64),vi ).reshape([1,window_size])
			#print ("temp_shift_train_data.shape: ",temp_shift_train_data.shape)
			
			temp_age = age[i*24:i*24+window_size,serie].reshape([1,window_size])
			temp_age = norm(temp_age)
			temp_hour = hour_of_the_day[i*24:i*24+window_size,serie].reshape([1,window_size])
			temp_hour = norm(temp_hour)
			temp_day = day_of_the_week[i*24:i*24+window_size,serie].reshape([1,window_size])
			temp_day = norm(temp_day)
			temp_train = np.concatenate([temp_shift_train_data, temp_age,temp_hour,temp_day]).transpose() #shape = [ window_size,4]

			#print ("temp_train.shape: ", temp_train.shape) 
			shift_train_data.append(temp_train) #
			shift_train_onehot.append(num_2_onehot(serie, num_series))
			v.append(vi)
			#No scaling for label
			shift_train_label.append((data[i*24+1:i*24+window_size+1,serie]))

	
	#shape: [num_window*num_series(1453*370), window_size,4]
	shift_train_data = np.array(shift_train_data)
	#shape: [num_window*num_series(1453*370), num_series]
	shift_train_onehot = np.array(shift_train_onehot)
	#shape:[num_window*num_series(1453*370) , 1]
	v = np.array(v).reshape([(num_window)*num_series , 1]).reshape([(num_window)*num_series, 1])
	#shape: [num_window*num_series(1453*370), window_size]
	shift_train_label = np.array(shift_train_label)
	
	print ("shift_train_data.shape:", shift_train_data.shape)
	print ("shift_train_onehot.shape: ",shift_train_onehot.shape)
	print ("v.shape: ",v.shape)
	print ("shift_train_label.shape:", shift_train_label.shape)

	##### permutation ####
	indexs = np.arange(num_window*num_series)
	#np.random.shuffle(indexs)
	#num_pred = (num_window*num_series)//10 # number of windows for prediction
	num_pred = 64
	
	#indexs_pred = indexs[num_window*num_series-num_pred:]
	#indexs_pred =indexs[(num_window*num_series)//2:(num_window*num_series)//2+64 ]
	indexs_pred = [indexs[i*(num_window//2-1)] for i in range(num_series*2)]
	indexs_pred = indexs_pred[0:704] # devisible by 64 (704 = 64*11) 
	#print (indexs_pred)

	for ele_to_remove in indexs_pred:
		for j in range(ele_to_remove-5,ele_to_remove+6): #从ele_to_remove-5 至 ele_to_remove+6 都与 ele_to_remove有重叠
			try: indexs.remove(j)
			except:	pass
	#print (indexs)

	### shuffle ###
	np.random.shuffle(indexs)
	np.random.shuffle(indexs_pred)
	#print ("indexs_pred after shuffle: ", indexs_pred)


	num_row = (len(indexs))//batch_size
	num_row_pred = len(indexs_pred)//batch_size #suppose to be 704//64 = 11
	print ("num_row_pred: ", num_row_pred)

	indexs_list=[]
	indexs_pred_list=[]
	for i in range(num_row):
		indexs_list.append(indexs[i*batch_size:(i+1)*batch_size])
		if i==(num_row-1): indexs_list.append(indexs[(i+1)*batch_size:])

	for i in range(num_row_pred):
		indexs_pred_list.append(indexs_pred[i*batch_size:(i+1)*batch_size])
		#if i==(num_row_pred-1): indexs_pred_list.append(indexs_pred[(i+1)*batch_size:])

	#print ("num_row: ",num_row)
	#print (len(indexs_list))
	#print (indexs_list[-1].shape)

	#indexs_list = [indexs_pred for i in range (100)]

	return (shift_train_data,
			shift_train_onehot, 
			v,
			shift_train_label, 
			param, 
			indexs_list, 
			indexs_pred_list)
'''
(shift_train_data,
shift_train_onehot, 
v,
shift_train_label, 
param, 
indexs_list, 
indexs_pred_list)=prepare()

print (len(indexs_pred_list))
print (len(indexs_pred_list[0]))
'''






