import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
#####



class DeepAR (object):
	def __init__(self,
				 input_x_all, #shape:[num_window*num_series(1453*370),  window_size,4]
				 input_y_all, #shape: [num_window*num_series(1453*370), window_size]
				 input_x_onehot_all, #shape: [num_window*num_series(1453*370), num_series]
				 num_covar,
				 encode_length,
				 decode_length,
				 num_series, #370
				 embedding_output_size, #20
				 hidden_unit, #40
				 num_layer,
				 window_size,
				 v_all, #shape: [num_window*num_series(1453*370) , 1]
				 index_list,
				 indexs_pred,
				):

		self.input_x_all = input_x_all
		self.input_y_all = input_y_all
		self.input_x_onehot_all = input_x_onehot_all
		self.v_all = v_all

		self.window_size = window_size

		self.num_series = num_series
		self.encode_length = encode_length
		self.decode_length = decode_length
		self.index_list=index_list
		self.indexs_pred = indexs_pred

		self.input_x = tf.placeholder(
			tf.float32, [None,window_size, num_covar+1], name = 'input_x') # [None, 192, 4]
		self.input_x_onehot = tf.placeholder(
			tf.float32, [None,], name= 'input_onehot')   # [None, 370]
		self.input_y = tf.placeholder(
			tf.float32, [None,window_size], name = 'input_y') # [None, 192]
		self.v = tf.placeholder(
			tf.float32, [None, 1], name = 'input_v') # [None, 1]
		self.batch_size = tf.placeholder(
			tf.int32, [], name='input_batch')
		self.keep_prob = tf.placeholder(
			tf.float32, name='keep_prob')
		batch_size = self.batch_size
		self.forget_gate_mask = tf.placeholder(
			tf.float32, [None, window_size, hidden_unit], name = "forget_gate_mask")

		self.forget_gate_mask_one = tf.fill(tf.shape(self.forget_gate_mask), 1.0) # mask with all elements=1.0
		'''
		embedding_shape = [num_series, embedding_output_size]
		self.W = tf.Variable(tf.truncated_normal(embedding_shape, stddev=0.1),name = 'W' )
		self.b = tf.Variable(tf.constant(0.1, shape = [embedding_output_size]), name='b')
		self.embedded_input = tf.nn.xw_plus_b(self.input_x_onehot, self.W, self.b, name='embedded_input') # [None, 20]
		self.embedded_input = tf.nn.relu(self.embedded_input)
		embedded_input_expand = tf.expand_dims(self.embedded_input, axis = 1) # [None, 1, 20]
		self.embedded_input_all = tf.concat([embedded_input_expand for i in range(window_size)], axis=1) # [None, 192, 20]
		'''
		#concat embedded_input_all and input_x
		self.lstm_input = self.input_x # [None, 192, 1]
		#self.cnn_input = self.input_x[:, :encode_length, 0] #shape = [batch_size, encode_length], pure data
		self.cnn_input = self.input_x[:, :window_size, 0] #shape = [batch_size, encode_length], pure data
		self.cnn_input = tf.expand_dims(self.cnn_input, axis=2)
		self.cnn_input = tf.expand_dims(self.cnn_input, axis=3) #[batch_size, encode_length, width=1, input_channel=1]
		print ("cnn_input.shape: ", self.cnn_input.shape)

		### Conv ###
		def CNN(self, filter_size_list, dialations_size, input_channel_list, output_channel_list, keep_prob, cnn_input, is_pred, ):
			#cnn_input.shape = [batch_size, height, width, input_channel]
			#num_filter is for every kernal
			cnn_input_list = [] # input for each layer
			cnn_input_list.append(cnn_input) # take raw value as input for the first layer
			for i, filter_size in enumerate(filter_size_list):
				with tf.variable_scope("CNN-"+str(filter_size), reuse=is_pred):
					filter_shape = [filter_size, 1, input_channel_list[i], output_channel_list[i]]
					# parameter for conv
					if not is_pred:
						# Weights
						W = tf.get_variable(name='W', shape=filter_shape, dtype=tf.float32,initializer=tf.random_normal_initializer(mean=0, stddev=0.1))
						b = tf.get_variable(name='b', shape=[output_channel_list[i]], dtype = tf.float32, initializer = tf.constant_initializer(value=0.1))
						# inputs
						batch_input_conv = cnn_input_list[i]
					else:
						# Weights
						W = tf.get_variable(name='W')
						b = tf.get_variable(name='b')
						# inputs
						batch_input_conv = cnn_input_list[i]
					#print ("batch_input_conv_pad.shape: ", batch_input_conv_pad.shape)
					conv = tf.nn.conv2d(
						batch_input_conv,  ### shape = [batch_size, encode_length]
						W,
						strides=[1,1,1,1],
						padding = 'VALID', #不进行padding，多余部分丢弃
						name='conv',
						dilations = [1,dialations_size[i],1,1])
					h = tf.nn.relu(tf.nn.bias_add(conv,b), name = 'relu') # [batch_size, encode_length, width=1, ouput_channel]
					if i!= (len(filter_size_list)-1):
						h = tf.nn.dropout(h, keep_prob)
					cnn_input_list.append(h)
					#print ("h.shape: ", h.shape)
			return tf.squeeze(h, axis=2) #[batch_size, input_length - 23, ouput_channel=40]
		filter_size_list = [4, 2, 3]
		dialations_size = [1, 4, 8]
		input_channel_list = [1,80,60]
		output_channel_list = [80,60,40]

		with tf.variable_scope("RNN"):
			cnn_output = CNN(self, filter_size_list, dialations_size, input_channel_list, output_channel_list, self.keep_prob, cnn_input=self.cnn_input, is_pred=False,)
		print ("cnn_output.shape: ", cnn_output.shape)
		cnn_output_padding = tf.ones(shape = [self.batch_size, 23, hidden_unit], dtype= tf.float32) #padding for first 23 cnn_output
		print ("cnn_output_padding.shape: ", cnn_output_padding.shape) # [batch_size, length = 192, hidden_unit=40]
		self.cnn_output = tf.concat([cnn_output_padding,cnn_output], axis=1) #[batch_size, 192, 40]
		print ("self.cnn_output.shape: ", self.cnn_output.shape)
		#calculate cnn output for predction part
		cnn_output_average = tf.reduce_mean(self.cnn_output, axis=1) #[batch_size, num_filter*4=40]
		self.cnn_output_pad = tf.concat([tf.expand_dims(cnn_output_average, axis=1) for i in range(decode_length)], axis=1) # [batch_size, decode_length, num_filter*4=40]
		print ("cnn_output_pad.shape: ", self.cnn_output_pad.shape)
		self.cnn_output_pred = tf.concat([self.cnn_output[:, :encode_length, :],self.cnn_output_pad], axis=1)
		self.lstm_input_with_cnn = tf.concat([self.cnn_output, self.lstm_input], axis = 2) #[batch_size, 192, 40+1]
		print ("lstm_input_with_cnn.shape: ", self.lstm_input_with_cnn.shape)
		#self.embedded_input = tf.nn.relu(self.embedded_input)
		def lstm_cell(hidden_unit):
			cell = tf.contrib.rnn.LSTMCell(num_units = hidden_unit, forget_bias = 1.0,state_is_tuple=True)
			#cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.7)
			return cell

		def lstm_cell_drop_out(hidden_unit, keep_prob):
			cell = tf.contrib.rnn.LSTMCell(num_units = hidden_unit, forget_bias = 1.0,state_is_tuple=True)
			cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
			return cell

		#堆叠多层lstm，返回state=(c,h)
		#cell = lstm_cell(hidden_unit)
		#cell = lstm_cell_drop_out(hidden_unit, self.keep_prob)

		cell_list_without_drop = [lstm_cell(hidden_unit)]
		cell_list_with_drop = [lstm_cell_drop_out(hidden_unit, self.keep_prob) for i in range(num_layer-1)]
		cell_list = cell_list_without_drop + cell_list_with_drop
		cell = tf.nn.rnn_cell.MultiRNNCell(cell_list, state_is_tuple=True)

		self.cell = cell
		self.initial_state = cell.zero_state(self.batch_size, dtype = tf.float32)

		#output的list
		outputs = []
		state = self.initial_state
		with tf.variable_scope("RNN"):
			for time_step in range(self.window_size):
				if time_step > 0: tf.get_variable_scope().reuse_variables()

				#cell_output:[batch_size, hidden_unit], state=(c,h)
				(cell_output, state) = cell(self.lstm_input_with_cnn[:,time_step,:], state,)  #feed forget_gate_mask according to time_step
				cell_output = tf.nn.relu(cell_output)
				outputs.append(cell_output) #outputs's length=192, each element's shape: [None, 40]


		with tf.variable_scope("Miu"):
			miu = []
			W = tf.get_variable(name='W',
								shape=[hidden_unit, 1],
								dtype=tf.float32,
								initializer=tf.random_normal_initializer(mean=0, stddev=0.1))
			b = tf.get_variable(name='b',
								shape=[1],
								dtype = tf.float32,
								initializer = tf.constant_initializer(value=0.1))
			for output in outputs:
				miu_temp = (tf.nn.xw_plus_b(output, W, b, name='miu')) #[None, 1]

				miu_temp = tf.multiply(miu_temp, self.v) #还原scale
				miu.append(miu_temp)


			#concate miu in all timesteps
			self.miu_concat = tf.concat(miu, axis=1) # [None, window_size]
			#print ("miu_concat.shape: ", self.miu_concat.shape)

		with tf.variable_scope("Sigma"):
			sigma = []
			#W = tf.Variable(tf.truncated_normal([hidden_unit, 1]), name='W')
			#b = tf.Variable(tf.constant(0.1, shape = [1]), name='b')
			W = tf.get_variable(name='W',
								shape=[hidden_unit, 1],
								dtype=tf.float32,
								initializer=tf.random_normal_initializer(mean=0, stddev=0.1))
			b = tf.get_variable(name='b',
								shape=[1],
								dtype = tf.float32,
								initializer = tf.constant_initializer(value=0.1))
			for output in outputs:
				sigma_temp = tf.nn.softplus(tf.nn.xw_plus_b(output, W, b), name='sigma')
				sigma_temp = tf.multiply(sigma_temp, tf.sqrt(self.v))
				sigma.append(sigma_temp)
			#concate sigma in all timesteps
			self.sigma_concat = tf.concat(sigma, axis = 1)

		def log_likelihood(mu, sigma, z):
			dist = tf.contrib.distributions.NormalWithSoftplusScale(loc=mu, scale=sigma)
			prob_log = dist.log_prob(value = z)
			neg_log_likelihood = -1.0 * tf.reduce_mean(prob_log)
			return prob_log, neg_log_likelihood

		with tf.variable_scope("loglikelihood"):
			#loglikelihood of all timesteps
			#print ("input_y.shape: ", self.input_y.shape)

			self.prob_log, self.neg_log_likelihood = log_likelihood(mu=self.miu_concat, sigma=self.sigma_concat, z=self.input_y )

			#self.minus_likelihood = tf.subtract(0.0 , self.log_likelihood_sum)


		def RMSE(mu, label):
			mu_pred = mu[:, self.encode_length:self.window_size] #[None, 24]
			#print ("mu_pred.shape: ", mu_pred.shape)
			label_pred = label[:, self.encode_length:self.window_size]
			RMSE = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(label_pred, mu_pred))))
			norm = tf.reduce_mean(tf.abs(label_pred))
			RMSE_norm = RMSE/norm
			return RMSE_norm

		def ND(mu, label):
			mu_pred = mu[:, self.encode_length:self.window_size] #[None, 24]
			label_pred = label[:, self.encode_length:self.window_size]
			sum_subtract = tf.reduce_sum(tf.abs(tf.subtract(mu_pred,label_pred)))
			sum_label = tf.reduce_sum(label_pred)
			ND_value = sum_subtract/sum_label
			return ND_value



		self.RMSE_train = RMSE(self.miu_concat, self.input_y)
		self.ND_train = ND(self.miu_concat, self.input_y)


##########################.   Prediction.     ############################
		# update input for prediction since CNN_output is different in prediction
		self.lstm_input_with_cnn = tf.concat([self.cnn_output_pred, self.lstm_input], axis = 2) #[batch_size, 192, 40+8]
		with tf.variable_scope("Miu", reuse=True):
			W_pred = tf.get_variable("W") # [hidden_unit, 1]
			b_pred = tf.get_variable("b") # [1]

		def miu_pred(output_pred, v_pred, W_pred, b_pred): #output_pred.shape: [1, hidden_unit]
			miu_temp = tf.nn.xw_plus_b(output_pred, W_pred, b_pred, name='miu')

			#sigma_temp = tf.multiply(sigma_temp, tf.sqrt(v_pred)) # shape: []
			# here sigma has not be scaled back
			return miu_temp

		def decode_input(decode_input_value,time_step, keep_prob):
			#update lstm_input
			'''
			decode_input_data = tf.concat([self.lstm_input_with_cnn[:,time_step,0:40],
											decode_input_value,
											self.lstm_input_with_cnn[:,time_step,41:]], axis=1) #前40为mask，第41为需要替换的decode_input_value, 第42-64为covariates
			'''
			filter_size_list = [4, 2, 3]
			dialations_size = [1, 4, 8]
			input_channel_list = [1,80,60]
			output_channel_list = [80,60,40]
			cnn_input = tf.expand_dims(tf.expand_dims(self.lstm_input_with_cnn[:,time_step-24:time_step, 40], axis=2), axis=3) #[batch_size, 24, 1, 1]
			cnn_output_latest = CNN(self, filter_size_list, dialations_size, input_channel_list, output_channel_list, self.keep_prob, cnn_input=cnn_input, is_pred=True, ) #[batch_size, length=1, hidden_unit=40]
			cnn_output_latest = tf.squeeze(cnn_output_latest, axis=1) #[batch_size, hidden_unit=40]
			#print ("&&&&&")
			decode_input_data = tf.concat([cnn_output_latest,
											decode_input_value,
											], axis=1)
			return decode_input_data


		miu_pred_list = []
		state_pred = cell.zero_state(self.batch_size, dtype = tf.float32)
		outputs_pred=[] # shape for each element: [2,batch_size, hidden_unit], total length=192
		self.hidden_states_all = []
		with tf.variable_scope("RNN", reuse = True):
			for time_step_encode in range(self.encode_length):
				(cell_output, state_pred) = cell(self.lstm_input_with_cnn[:,time_step_encode,:], state_pred, ) #
				self.hidden_states_all.append(tf.expand_dims(state_pred, axis = -1)) #[3,2,64,40,1] or [2,64,40,1]
				cell_output = tf.nn.relu(cell_output)
				outputs_pred.append(cell_output) #outputs's length=168, each element's shape: [None, hidden_unit]
				miu_pred_list.append(tf.multiply(miu_pred(cell_output, self.v, W_pred, b_pred),self.v)) ##update list of miu
			decode_input_value = miu_pred(cell_output, self.v, W_pred, b_pred) # output => not scaled back miu => next input value
			decode_input_data = decode_input(decode_input_value, self.encode_length, self.keep_prob) #更新decode第一个input的数值
			for time_step_decode in range(self.encode_length, self.window_size):
				(cell_output, state_pred) = cell(decode_input_data, state_pred,  )
				self.hidden_states_all.append(tf.expand_dims(state_pred, axis=-1)) #[3,2,64,40,1] or [2,64,40,1]
				cell_output = tf.nn.relu(cell_output)
				outputs_pred.append(cell_output)
				decode_input_value = miu_pred(cell_output, self.v, W_pred, b_pred)
				miu_pred_list.append(tf.multiply(decode_input_value,self.v)) #update list of miu
				if time_step_decode < (self.window_size-1):
					decode_input_data = decode_input(decode_input_value, time_step_decode+1, self.keep_prob) #更新time_step_decode+1 的 input的数值
			self.miu_concat_pred = tf.concat(miu_pred_list , axis=1) # shape=[None, 192]
			#print (self.hidden_states_all[0].shape)
			self.hidden_states_all = tf.concat(self.hidden_states_all, axis = -1)  #[3,2,64,40,192] or [2,64,40,192]
			print ("hidden_states_all.shape: ", self.hidden_states_all.shape)

		sigma_pred_list = []
		with tf.variable_scope("Sigma",reuse=True):
			W_pred_s = tf.get_variable("W")
			b_pred_s = tf.get_variable("b")
			for output in outputs_pred:
				#sigma_temp_pred = tf.nn.softplus(tf.nn.xw_plus_b(output, W_pred_s, b_pred_s), name='sigma')
				sigma_temp_pred = tf.nn.softplus(tf.nn.xw_plus_b(output, W_pred_s, b_pred_s))
				sigma_temp_pred = tf.multiply(sigma_temp_pred, tf.sqrt(self.v))
				sigma_pred_list.append(sigma_temp_pred)
			#concate sigma in all timesteps
			self.sigma_concat_pred = tf.concat(sigma_pred_list, axis = 1)
			#print (type(self.sigma_concat_pred))
			#print ("sigma_concat_pred.shape: ", self.sigma_concat_pred.shape)

		###### prediction evaluation
		self.prob_log_pred, self.neg_log_likelihood_pred = log_likelihood(self.miu_concat_pred, self.sigma_concat_pred, self.input_y)
		self.RMSE_pred = RMSE(self.miu_concat_pred, self.input_y)
		self.ND_pred = ND(self.miu_concat_pred, self.input_y)







	def train(self):

		def plot(label, prediction,train,num_plot, step):
			x = np.arange(192)
			f = plt.figure()
			base = num_plot*100+10

			for i in range(num_plot):
				label_temp = label[i].reshape([self.window_size,])
				#print ("label_temp.shape: ", label_temp.shape)
				pred_temp = prediction[i].reshape([self.window_size,])
				train_temp = train[i].reshape([self.window_size])

				plt.subplot(base+i+1)
				plt.plot(x,label_temp, color='b')
				plt.plot(x,pred_temp, color='r')
				plt.plot(x,train_temp,color='g')
				plt.axvline(168, color='k', linestyle = "dashed")
				#参考线

			#plt.pause(5)
			f.savefig(str(step)+'.png')
			plt.close()


		#plot(self.input_y_all[self.indexs_pred] ,self.input_y_all[self.indexs_pred],self.input_y_all[self.indexs_pred], 5 )

		self.global_step = tf.Variable(0, name='global_step', trainable = False)

		self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001,
    									beta1=0.9,
    									beta2=0.999,
    									epsilon=1e-08,
    									use_locking=False,
    									name='Adam')

		#self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001,name='GradientDescent')
		self.grad = self.optimizer.compute_gradients(self.neg_log_likelihood)
		self.train_op = self.optimizer.apply_gradients(
				self.grad, global_step = self.global_step)

		#### Add variables to collection for loading Model in future ####

		tf.add_to_collection("train_op",self.train_op)
		tf.add_to_collection("label" ,self.input_y)
		## Train
		tf.add_to_collection("miu_train", self.miu_concat)
		tf.add_to_collection("sigma_train", self.sigma_concat)
		tf.add_to_collection("RMSE_train", self.RMSE_train)
		tf.add_to_collection("ND_train", self.ND_train)
		## Predict
		tf.add_to_collection("miu_pred", self.miu_concat_pred)
		tf.add_to_collection("sigma_pred", self.sigma_concat_pred)
		tf.add_to_collection("RMSE_pred", self.RMSE_pred)
		tf.add_to_collection("ND_pred", self.ND_pred)
		tf.add_to_collection("hidden_states_all", self.hidden_states_all)
		tf.add_to_collection("CNN_output", self.cnn_output)
		#tf.add_to_collection("cell",self.cell)




		def train_step(x_batch,onehot_batch, y_batch,v_batch, batch_size,forget_gate_mask,keep_prob=0.7):
			#x_batch = np.concatenate((forget_gate_mask, x_batch), axis = 1) #[batch_size, 40+24]

			feed_dict = {
			self.input_x: x_batch,
			self.input_x_onehot: onehot_batch,
			self.input_y: y_batch,
			self.v :v_batch,
			self.batch_size: batch_size,
			self.forget_gate_mask: forget_gate_mask,
			self.keep_prob: keep_prob,
			}

			_, step, neg_log_likelihood, prob_log, RMSE_train, ND_train, miu_train = sess.run([self.train_op,
														 self.global_step,
														 self.neg_log_likelihood,
														 self.prob_log,
														 self.RMSE_train,
														 self.ND_train,
														 self.miu_concat,

														 ],feed_dict = feed_dict)
			#print ("likelihood: ", likelihood)

			return (step, neg_log_likelihood, prob_log,RMSE_train, ND_train, miu_train)

		######

		def pred_step(x_batch, onehot_batch, y_batch, v_batch, batch_size,forget_gate_mask, keep_prob=1.0):
			#x_batch = np.concatenate((forget_gate_mask, x_batch), axis = 1) #[batch_size, 40+24]

			feed_dict ={
			self.input_x: x_batch,
			self.input_x_onehot: onehot_batch,
			self.input_y: y_batch,
			self.v :v_batch,
			self.batch_size: batch_size,
			self.forget_gate_mask: forget_gate_mask,
			self.keep_prob: keep_prob,
			}
			neg_log_likelihood_pred, prob_log_pred,RMSE_pred, ND_pred, miu_pred = sess.run([
			 			 										self.neg_log_likelihood_pred,
			 			 										self.prob_log_pred,
			 			 										self.RMSE_pred,
			 			 										self.ND_pred,
			 			 										self.miu_concat_pred
			 			 										],
			 			 										feed_dict = feed_dict)
			return (neg_log_likelihood_pred, RMSE_pred, ND_pred, miu_pred)





		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
		with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
			init = tf.global_variables_initializer()
			sess.run(init)
			#创建saver对象
			saver = tf.train.Saver()
			forget_gate_mask_sample = np.full((64, 192, 40), 1, dtype = np.float32)

			print ("###########     TIME     ##########")
			print (datetime.datetime.now())
			print ("###########  TRAIN START ############")

			for j in range(20): #epochs
				for index in self.index_list:
					#index = index.reshape([self.batch_size, 1])
					input_x_batch = self.input_x_all[index]
					input_y_batch = self.input_y_all[index]
					input_onehot_batch = self.input_x_onehot_all
					input_v_batch = self.v_all[index]
					batch_size = len(index)
					forget_gate_mask_batch = forget_gate_mask_sample #[batch_size, 40]

					step, neg_log_likelihood, prob_log,RMSE_train, ND_train, miu_train = train_step(input_x_batch,
																	input_onehot_batch,
																	input_y_batch,
																	input_v_batch,
																	batch_size,
																	forget_gate_mask_batch,
																	keep_prob = 0.7,
																	)
					#print (forget_gate_mask_train) #should be all 1 with shape = [batch_size, hidden_units]
					#print ("forget_gate_mask_train.shape: ", forget_gate_mask_train.shape)
					if step%500==0:
						print ("prediction")
							### prediction ###
						neg_log_likelihood_pred_sum=0.0
						RMSE_pred_sum=0.0
						ND_pred_sum = 0.0

						for index_pred in self.indexs_pred:
							# prepare data for prediction
							input_x_batch_pred = self.input_x_all[index_pred]
							input_y_batch_pred = self.input_y_all[index_pred]
							input_onehot_batch_pred = self.input_x_onehot_all
							input_v_batch_pred = self.v_all[index_pred]
							batch_size_pred = len(index_pred)
							forget_gate_mask_batch_pred =  forget_gate_mask_sample

							neg_log_likelihood_pred, RMSE_pred, ND_pred, miu_pred = pred_step(input_x_batch_pred,
																			 input_onehot_batch_pred,
																			 input_y_batch_pred,
																			 input_v_batch_pred,
																			 batch_size_pred,
																			 forget_gate_mask_batch_pred,
																			 keep_prob = 1.0)
							neg_log_likelihood_pred_sum += neg_log_likelihood_pred
							RMSE_pred_sum += RMSE_pred
							ND_pred_sum += ND_pred


						neg_log_likelihood_pred_avg = neg_log_likelihood_pred_sum/(len(self.indexs_pred))
						RMSE_pred_avg = RMSE_pred_sum/(len(self.indexs_pred))
						ND_pred_avg = ND_pred_sum/(len(self.indexs_pred))


						print ("step:", step)
						print ("neg_log_likelihood_train: ", neg_log_likelihood)
						print ("neg_log_likelihood_prediction: ", neg_log_likelihood_pred_avg)
						print ("RMSE_train: ", RMSE_train)
						print ("RMSE_prediction: ", RMSE_pred_avg)
						print ("ND_train: ", ND_train)
						print ("ND_pred: ", ND_pred_avg)

						print ("input_y_batch_pred.shape",input_y_batch_pred.shape)
						#print (type(input_y_batch_pred[0,0]))
						#print (input_y_batch_pred)
						print ("miu_pred.shape",miu_pred.shape)
						#print (miu_pred)
						saver.save(sess, '../../checkpoint/checkpoint_CNN_Fake/DeepAR_model',global_step = step)


					if step%2000 == 0:
						index_pred = self.indexs_pred[0]
						input_x_batch_pred = self.input_x_all[index_pred]
						input_y_batch_pred = self.input_y_all[index_pred]
						input_onehot_batch_pred = self.input_x_onehot_all
						input_v_batch_pred = self.v_all[index_pred]
						batch_size_pred = len(index_pred)
						forget_gate_mask_batch_pred =  forget_gate_mask_sample

						neg_log_likelihood_pred, RMSE_pred, ND_pred, miu_pred = pred_step(input_x_batch_pred,
																			 input_onehot_batch_pred,
																			 input_y_batch_pred,
																			 input_v_batch_pred,
																			 batch_size_pred,
																			 forget_gate_mask_sample,
																			 keep_prob = 1.0,
																			 )

						#plot(input_y_batch_pred, miu_pred, 5)
						plot(input_y_batch_pred, miu_pred, miu_pred, 8, step)
						print ("Update 1.png")

				print ("###########  TIME  ##########")
				print (datetime.datetime.now())
				print ("###########  ", j, "Finish  #########")





########################################################################################################################
########################################################################################################################

#Electricity
'''
shift_train_data = np.load("../../data/Electricity/elect_pre_train_data.npy")
shift_train_onehot = np.load("../../data/Electricity/elect_train_onehot.npy")
v_all = np.load("../../data/Electricity/elect_train_v.npy")
shift_train_label = np.load("../../data/Electricity/elect_train_label.npy")
param = np.load("../../data/Electricity/elect_train_param.npy")
index_list = np.load("../../data/Electricity/elect_train_index.npy")
indexs_pred_list = np.load("../../data/Electricity/elect_train_pred_index.npy")

#Huawei
shift_train_data = np.load("../../data/huawei/shift_train_data.npy")
shift_train_onehot = np.load("../../data/huawei/shift_train_onehot.npy")
v_all = np.load("../../data/huawei/v.npy")
shift_train_label = np.load("../../data/huawei/shift_train_label.npy")
param = np.load("../../data/huawei/param.npy")
index_list = np.load("../../data/huawei/indexs_list.npy")
indexs_pred_list = np.load("../../data/huawei/indexs_pred_list.npy")

'''
#Fake data
shift_train_data = np.load("../../data/fake_data/elect_pre_train_data.npy")
shift_train_onehot = np.load("../../data/fake_data/elect_train_onehot.npy")
v_all = np.load("../../data/fake_data/elect_train_v.npy")
shift_train_label = np.load("../../data/fake_data/elect_train_label.npy")
param = np.load("../../data/fake_data/elect_train_param.npy")
index_list = np.load("../../data/fake_data/elect_train_index.npy")
indexs_pred_list = np.load("../../data/fake_data/elect_train_pred_index.npy")

print ("shift_train_data.shape: ", shift_train_data.shape )
print ("shift_train_onehot.shape: ", shift_train_onehot.shape )
print ("v_all.shape: ", v_all.shape )
print ("shift_train_label.shape: ", shift_train_label.shape )
print ("param.shape: ", param.shape )
print ("index_list.shape: ", index_list.shape )
print ("indexs_pred_list.shape: ", indexs_pred_list.shape )


(num_covar,
encode_length,
decode_length ,
num_series, #370
embedding_output_size, #20
hidden_unit, #40
num_layer,
window_size,
)=param.tolist()
print (num_covar,
encode_length,
decode_length ,
num_series, #370
embedding_output_size, #20
hidden_unit, #40
num_layer,
window_size)
print ("************************************************")
Model = DeepAR(
				input_x_all=shift_train_data, #shape:[num_window*num_series(1453*370), window_size,4]
				 input_y_all=shift_train_label, #shape: [num_window*num_series(1453*370),window_size]
				 input_x_onehot_all = shift_train_onehot,#shape: [num_window*num_series(1453*370), num_series]
				 num_covar= num_covar,
				 encode_length=encode_length,
				 decode_length=decode_length,
				 num_series=num_series, #370
				 embedding_output_size=embedding_output_size, #20
				 hidden_unit=hidden_unit, #40
				 num_layer=num_layer,
				 window_size = window_size,
				 v_all = v_all, #shape: [num_window*num_series(1453*370) , 1]
				 index_list = index_list,
				 indexs_pred = indexs_pred_list,
				)
Model.train()
