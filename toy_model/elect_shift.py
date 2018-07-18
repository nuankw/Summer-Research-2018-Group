from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
from os import path

from tensorflow.contrib.timeseries.python.timeseries import estimators as ts_estimators
from tensorflow.contrib.timeseries.python.timeseries import model as ts_model
from tensorflow.contrib.timeseries.python.timeseries import  NumpyReader

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from math import sqrt
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras import optimizers
from keras import initializers
from numpy import concatenate
from keras import backend as K

def timeseries_to_supervised(df, lag=1):
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df

def ND_metrics(y_true, y_pred):
    SS_res =  K.sum(K.abs( y_true - y_pred ))
    SS_tot = K.sum(K.abs( y_true ) )
    return SS_res/SS_tot

def RMSE(y_true, y_pred):
    return 1

elect_data_frame = pd.read_csv("electricity_hourly.csv")
del elect_data_frame['time']
elect_data_frame = timeseries_to_supervised(elect_data_frame, 1)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled = scaler.fit_transform(elect_data_frame.values)
elect_data_frame = pd.DataFrame(scaled)

num_series = int(elect_data_frame.shape[1]/2)
train_values = elect_data_frame.head(21000).values
train_X = train_values[:,:num_series]
train_y = train_values[:,num_series:]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
valid_values = elect_data_frame.iloc[21000:28000, :].values
valid_X = valid_values[:,:num_series]
valid_y = valid_values[:,num_series:]

valid_X = valid_X.reshape((valid_X.shape[0], 1, valid_X.shape[1]))
test_values = elect_data_frame.tail(7000).values
test_X = test_values[:,:num_series]
test_y = test_values[:,num_series:]
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print('train_X.shape:')
print(train_X.shape)
print('valid_y.shape:')
print(valid_y.shape)

# design network
model = Sequential()
model.add(LSTM(40, input_shape=(train_X.shape[1], train_X.shape[2]), dropout=0.2,
kernel_initializer = 'truncated_normal', return_sequences = True))
model.add(LSTM(30, dropout=0.2, kernel_initializer = 'truncated_normal', return_sequences = True))
model.add(LSTM(20, dropout=0.2, kernel_initializer = 'truncated_normal'))
model.add(Dense(370)) #final layer must match the total number of time series
adam = optimizers.Adam(lr=0.001)
model.compile(loss='mse', optimizer=adam, metrics = ['mse', ND_metrics])
# fit network
history = model.fit(train_X, train_y, epochs=100, batch_size=64, validation_data=(valid_X, valid_y), verbose=2, shuffle=False)

# make a prediction
yhat = model.predict(test_X)
'''test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE'''
rmse = sqrt(mean_squared_error(yhat, test_y))
print('Test RMSE: %.3f' % rmse)
yhat_sample = np.zeros(700)
test_y_sample = np.zeros(700)
for i in range(700):
    yhat_sample[i] = yhat[i*10][0]
    test_y_sample[i] = test_y[i*10][0]
print('yhat.shape:')
print(yhat.shape)
print('test_y.shape:')
print(test_y.shape)
plt.plot(yhat[0:500][0], label = 'yhat_1')
plt.plot(test_y[0:500][0], label = 'test_y_1')
#plt.plot(yhat[0:500][1], label = 'yhat_2')
#plt.plot(test_y[0:500][1], label = 'test_y_2')
plt.legend()
plt.show()
