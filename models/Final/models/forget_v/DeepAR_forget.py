import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
#####

dataset = []
checkpoint_dir = []


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
				 #mean_all,
				):

		self.input_x_all = input_x_all
		self.input_y_all = input_y_all
		self.input_x_onehot_all = input_x_onehot_all
		self.num_covar = num_covar
		self.encode_length = encode_length
		self.decode_length = decode_length
		self.num_series = num_series
		self.embedding_output_size = embedding_output_size
		self.hidden_unit = hidden_unit
		self.num_layer = num_layer
		self.window_size = window_size
		self.v_all = v_all
		#self.mean_all = mean_all
		self.index_list=index_list
		self.indexs_pred = indexs_pred

		self.input_x = tf.placeholder(
			tf.float32, [None,window_size, num_covar+1], name = 'input_x') # [None, 192, 4]
		self.input_x_onehot = tf.placeholder(
			tf.float32, [None, num_series], name= 'input_onehot')   # [None, 370]
		self.input_y = tf.placeholder(
			tf.float32, [None,window_size], name = 'input_y') # [None, 192]
		self.v = tf.placeholder(
			tf.float32, [None, 1], name = 'input_v') # [None, 1]
		#self.mean = tf.placeholder(
		#	tf.float32, [None, 1], name = 'input_mean') # [None, 1]
		self.batch_size = tf.placeholder(
			tf.int32, [], name='input_batch')
		self.keep_prob = tf.placeholder(
			tf.float32, name='keep_prob')
		self.forget_gate_mask = tf.placeholder(
			tf.float32, [None, window_size, hidden_unit], name = "forget_gate_mask")
		self.forget_gate_mask_one = tf.fill(tf.shape(self.forget_gate_mask), 1.0) # mask with all elements=1.0

		embedding_shape = [num_series, embedding_output_size]
		self.W = tf.Variable(tf.truncated_normal(embedding_shape, stddev=0.1),name = 'W' )
		self.b = tf.Variable(tf.constant(0.1, shape = [embedding_output_size]), name='b')
		self.embedded_input = tf.nn.xw_plus_b(self.input_x_onehot, self.W, self.b, name='embedded_input') # [None, 20]
		self.embedded_input = tf.nn.relu(self.embedded_input)
		embedded_input_expand = tf.expand_dims(self.embedded_input, axis = 1) # [None, 1, 20]
		self.embedded_input_all = tf.concat([embedded_input_expand for i in range(window_size)], axis=1) # [None, 192, 20]
		#concat embedded_input_all and input_x
		self.lstm_input = tf.concat([self.input_x, self.embedded_input_all], axis=2) # [None, 192, 24]
		self.lstm_input_with_mask = tf.concat([self.forget_gate_mask, self.lstm_input], axis = 2) #[batch_size, 192, 40+8]

		### ??? ###
		#self.embedded_input = tf.nn.relu(self.embedded_input)

		def lstm_cell(hidden_unit):
			cell = tf.contrib.rnn.LSTMCell(num_units = hidden_unit, forget_bias = 1.0,state_is_tuple=True)
			#cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.7)
			return cell

		def lstm_cell_drop_out(hidden_unit, keep_prob):
			cell = tf.contrib.rnn.LSTMCell(num_units = hidden_unit, forget_bias = 1.0,state_is_tuple=True)
			cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
			return cell

		# Multi-layer LSTM
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
				(cell_output, state) = cell(self.lstm_input_with_mask[:,time_step,:], state,)  #feed forget_gate_mask according to time_step
				cell_output = tf.nn.relu(cell_output)
				outputs.append(cell_output) #outputs's length=192, each element's shape: [None, 40]


		with tf.variable_scope("Miu"):
			miu = []
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
			self.prob_log, self.neg_log_likelihood = log_likelihood(mu=self.miu_concat, sigma=self.sigma_concat, z=self.input_y )

		def RMSE(mu, label):
			mu_pred = mu[:, self.encode_length:self.window_size] #[None, 24]
			#print ("mu_pred.shape: ", mu_pred.shape)
			label_pred = label[:, self.encode_length:self.window_size]
			RMSE = tf.square(tf.subtract(label_pred, mu_pred))
			norm = tf.abs(label_pred)
			norm = tf.reduce_mean(norm)
			norm_pred = tf.abs(mu_pred)
			norm_pred = tf.reduce_mean(norm_pred)
			RMSE = tf.sqrt(tf.reduce_mean(RMSE))
			if norm!=0:
				RMSE_norm = RMSE/norm
				return RMSE_norm
			else:
				if RMSE==0:
					return 0
				else:
					RMSE_norm = RMSE/((norm_pred+norm)/2.0)
					return RMSE_norm

		def ND(mu, label):
			mu_pred = mu[:, self.encode_length:self.window_size] #[None, 24]
			label_pred = label[:, self.encode_length:self.window_size]
			sum_subtract = tf.abs(tf.subtract(mu_pred,label_pred))
			sum_subtract = tf.reduce_sum(sum_subtract)
			sum_label = tf.reduce_sum(np.abs(label_pred))
			sum_pred = tf.reduce_sum(np.abs(mu_pred))
			if sum_label!=0:
				ND_value = sum_subtract/sum_label
				return ND_value
			else:
				if sum_subtract==0:
					return 0
				else:
					return sum_subtract/((sum_label+sum_pred)/2.0)

		self.RMSE_train = RMSE(self.miu_concat, self.input_y)
		self.ND_train = ND(self.miu_concat, self.input_y)


##########################.   Prediction.     ############################
		with tf.variable_scope("Miu", reuse=True):
			W_pred = tf.get_variable("W") # [hidden_unit, 1]
			b_pred = tf.get_variable("b") # [1]

		def miu_pred(output_pred, v_pred, W_pred, b_pred): #output_pred.shape: [1, hidden_unit]
			miu_temp = tf.nn.xw_plus_b(output_pred, W_pred, b_pred, name='miu')
			# here sigma has not be scaled back
			return miu_temp

		def decode_input(self, decode_input_value,time_step):
			#update lstm_input
			decode_input_data = tf.concat([self.lstm_input_with_mask[:,time_step,0:self.hidden_unit],
											decode_input_value,
											self.lstm_input_with_mask[:,time_step,self.hidden_unit+1:]], axis=1) #前40为mask，第41为需要替换的decode_input_value, 第42-64为covariates
			return decode_input_data

		miu_pred_list = []
		state_pred = cell.zero_state(self.batch_size, dtype = tf.float32)
		outputs_pred=[] # shape for each element: [2,batch_size, hidden_unit], total length=192
		self.hidden_states_all = []
		with tf.variable_scope("RNN", reuse = True):
			for time_step_encode in range(self.encode_length):
				#cell_output:[batch_size, hidden_unit], state=(c,h)
				(cell_output, state_pred) = cell(self.lstm_input_with_mask[:,time_step_encode,:], state_pred, ) #
				self.hidden_states_all.append(tf.expand_dims(state_pred, axis = 3))
				cell_output = tf.nn.relu(cell_output)
				outputs_pred.append(cell_output) #outputs's length=168, each element's shape: [None, hidden_unit]
				miu_pred_list.append((tf.multiply(miu_pred(cell_output, self.v, W_pred, b_pred),self.v))) ##update list of miu

			decode_input_value = miu_pred(cell_output, self.v, W_pred, b_pred) # output => not scaled back miu => next input value
			decode_input_data = decode_input(self, decode_input_value, self.encode_length) #更新decode第一个input的数值
			for time_step_decode in range(self.encode_length, self.window_size):
				(cell_output, state_pred) = cell(decode_input_data, state_pred,  )
				self.hidden_states_all.append(tf.expand_dims(state_pred, axis=3))
				cell_output = tf.nn.relu(cell_output)
				outputs_pred.append(cell_output)
				decode_input_value = miu_pred(cell_output, self.v, W_pred, b_pred)
				miu_pred_list.append((tf.multiply(decode_input_value,self.v))) #update list of miu
				if time_step_decode < (self.window_size-1):
					decode_input_data = decode_input(self, decode_input_value,time_step_decode+1) #更新time_step_decode+1 的 input的数值
			self.miu_concat_pred = tf.concat(miu_pred_list , axis=1) # shape=[None, 192]
			self.hidden_states_all = tf.concat(self.hidden_states_all, axis = 3)
			print ("hidden_states_all.shape: ", self.hidden_states_all.shape)



		sigma_pred_list = []
		with tf.variable_scope("Sigma",reuse=True):
			W_pred_s = tf.get_variable("W")
			b_pred_s = tf.get_variable("b")
			for output in outputs_pred:
				sigma_temp_pred = tf.nn.softplus(tf.nn.xw_plus_b(output, W_pred_s, b_pred_s))
				sigma_temp_pred = tf.multiply(sigma_temp_pred, tf.sqrt(self.v))
				sigma_pred_list.append(sigma_temp_pred)
			#concate sigma in all timesteps
			self.sigma_concat_pred = tf.concat(sigma_pred_list, axis = 1)

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
		#tf.add_to_collection("cell",self.cell)




		def train_step(x_batch,onehot_batch, y_batch,v_batch, batch_size,forget_gate_mask,keep_prob=0.7):
			#x_batch = np.concatenate((forget_gate_mask, x_batch), axis = 1) #[batch_size, 40+24]

			feed_dict = {
			self.input_x: x_batch,
			self.input_x_onehot: onehot_batch,
			self.input_y: y_batch,
			self.v :v_batch,
			#self.mean: mean_batch,
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

		def pred_step(x_batch, onehot_batch, y_batch, v_batch,  batch_size,forget_gate_mask, keep_prob=1.0):
			#x_batch = np.concatenate((forget_gate_mask, x_batch), axis = 1) #[batch_size, 40+24]

			feed_dict ={
			self.input_x: x_batch,
			self.input_x_onehot: onehot_batch,
			self.input_y: y_batch,
			self.v :v_batch,
			#self.mean: mean_batch,
			self.batch_size: batch_size,
			self.forget_gate_mask: forget_gate_mask,
			self.keep_prob: keep_prob,
			}
			neg_log_likelihood_pred, prob_log_pred,RMSE_pred, ND_pred, miu_pred, sigma_pred = sess.run([
			 			 										self.neg_log_likelihood_pred,
			 			 										self.prob_log_pred,
			 			 										self.RMSE_pred,
			 			 										self.ND_pred,
			 			 										self.miu_concat_pred,
																self.sigma_concat_pred,
			 			 										],
			 			 										feed_dict = feed_dict)
			#Recompute RMSE and ND with compromise
			#RMSE_pred = RMSE_compromise(mu=miu_pred, label=y_batch, std=sigma_pred)
			#ND_pred = ND_compromise(mu=miu_pred, label=y_batch, std=sigma_pred)
			return (neg_log_likelihood_pred, RMSE_pred, ND_pred, miu_pred)





		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
		with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
			init = tf.global_variables_initializer()
			sess.run(init)
			#创建saver对象
			saver = tf.train.Saver()
			forget_gate_mask_sample = np.full((64, 192, self.hidden_unit), 1, dtype = np.float32)

			print ("###########     TIME     ##########")
			print (datetime.datetime.now())
			print ("###########  TRAIN START ############")

			for j in range(20): #epochs
				for index in self.index_list:
					#index = index.reshape([self.batch_size, 1])
					input_x_batch = self.input_x_all[index]
					input_y_batch = self.input_y_all[index]
					input_onehot_batch = self.input_x_onehot_all[index]
					input_v_batch = self.v_all[index]
					#input_mean_batch = self.mean_all[index]
					batch_size = len(index)
					forget_gate_mask_batch = forget_gate_mask_sample #[batch_size, 40]


					step, neg_log_likelihood, prob_log,RMSE_train, ND_train, miu_train = train_step(input_x_batch,
																	input_onehot_batch,
																	input_y_batch,
																	input_v_batch,
																	#input_mean_batch,
																	batch_size,
																	forget_gate_mask_batch,
																	keep_prob = 0.7,
																	)

					#print (forget_gate_mask_train) #should be all 1 with shape = [batch_size, hidden_units]
					#print ("forget_gate_mask_train.shape: ", forget_gate_mask_train.shape)
					if step%2000==0:
						print ("prediction")
							### prediction ###
						neg_log_likelihood_pred_sum=0.0
						RMSE_pred_sum=0.0
						ND_pred_sum = 0.0

						for index_pred in self.indexs_pred:
							# prepare data for prediction
							input_x_batch_pred = self.input_x_all[index_pred]
							input_y_batch_pred = self.input_y_all[index_pred]
							input_onehot_batch_pred = self.input_x_onehot_all[index_pred]
							input_v_batch_pred = self.v_all[index_pred]
							#input_mean_batch_pred = self.mean_all[index_pred]
							batch_size_pred = len(index_pred)
							forget_gate_mask_batch_pred =  forget_gate_mask_sample

							neg_log_likelihood_pred, RMSE_pred, ND_pred, miu_pred = pred_step(input_x_batch_pred,
																			 input_onehot_batch_pred,
																			 input_y_batch_pred,
																			 input_v_batch_pred,
																			 #input_mean_batch_pred,
																			 batch_size_pred,
																			 forget_gate_mask_batch_pred,
																			 keep_prob = 1.0,
																			 )
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
						print ("miu_pred.shape",miu_pred.shape)
						plot(input_y_batch_pred, miu_pred, miu_pred, 8, step)
						print ("Update 1.png")
						saver.save(sess, '../../checkpoint/checkpoint_forget_Fuahui_v/DeepAR_model',global_step = step)
						#print (miu_pred)


					'''
					if step%2000 == 0:
						index_pred = self.indexs_pred[0]
						input_x_batch_pred = self.input_x_all[index_pred]
						input_y_batch_pred = self.input_y_all[index_pred]
						input_onehot_batch_pred = self.input_x_onehot_all[index_pred]
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
					'''
				print ("###########  TIME  ##########")
				print (datetime.datetime.now())
				print ("###########  ", j, "Finish  #########")





########################################################################################################################
########################################################################################################################
'''
#Electricity
shift_train_data = np.load("../../data/Electricity/elect_pre_train_data.npy")
shift_train_onehot = np.load("../../data/Electricity/elect_train_onehot.npy")
v_all = np.load("../../data/Electricity/elect_train_v.npy")
shift_train_label = np.load("../../data/Electricity/elect_train_label.npy")
param = np.load("../../data/Electricity/elect_train_param.npy")
index_list = np.load("../../data/Electricity/elect_train_index.npy")

indexs_pred_list = np.load("../../data/Electricity/elect_train_pred_index.npy")
'''
#Huawei
shift_train_data = np.load("../../data/huawei/shift_train_data.npy")
shift_train_onehot = np.load("../../data/huawei/shift_train_onehot.npy")
v_all = np.load("../../data/huawei/v.npy")
shift_train_label = np.load("../../data/huawei/shift_train_label.npy")
param = np.load("../../data/huawei/param.npy")
index_list = np.load("../../data/huawei/indexs_list.npy")
indexs_pred_list = np.load("../../data/huawei/indexs_pred_list.npy")
#mean_all = np.load("../../data/huawei/mean.npy")


print ("shift_train_data.shape: ", shift_train_data.shape )
print ("shift_train_onehot.shape: ", shift_train_onehot.shape )
print ("v_all.shape: ", v_all.shape )
print ("shift_train_label.shape: ", shift_train_label.shape )
print ("param.shape: ", param.shape )
print ("index_list.shape: ", index_list.shape )
print ("indexs_pred_list.shape: ", indexs_pred_list.shape )
#print ("mean_all.shape: ", mean_all.shape )

(num_covar,
encode_length,
decode_length ,
num_series, #370
embedding_output_size, #20
hidden_unit, #40
num_layer,
window_size,
)=param.tolist()

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

				 #mean_all = mean_all,
				)
Model.train()
