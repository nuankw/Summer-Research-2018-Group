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
                 hidden_unit, #10
                 num_layer,
                 window_size,
                 v_all, #shape: [num_window*num_series(1453*370) , 1]
                 index_list,
                 indexs_pred,
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
        self.index_list=index_list
        self.indexs_pred = indexs_pred

        self.input_x = tf.placeholder(
            tf.float32, [None, encode_length, num_covar+1], name = 'input_x') # [None, 168, 3]
        self.input_x_onehot = tf.placeholder(
            tf.float32, [None, num_series], name= 'input_onehot')   # [None, 370]
        self.input_y = tf.placeholder(
            tf.float32, [None,encode_length], name = 'input_y') # [None, 168]
        self.v = tf.placeholder(
            tf.float32, [None, 1], name = 'input_v') # [None, 1]
        self.batch_size = tf.placeholder(
            tf.int32, [], name='input_batch')
        self.keep_prob = tf.placeholder(
            tf.float32, name='keep_prob')
        self.forget_gate_mask = tf.placeholder(
            tf.float32, [None, encode_length, hidden_unit], name = "forget_gate_mask")
        self.forget_gate_mask_one = tf.fill(tf.shape(self.forget_gate_mask), 1.0) # mask with all elements=1.0
        batch_size = self.batch_size

        embedding_shape = [num_series, embedding_output_size]
        self.W = tf.Variable(tf.truncated_normal(embedding_shape, stddev=0.1),name = 'W' )
        self.b = tf.Variable(tf.constant(0.1, shape = [embedding_output_size]), name='b')
        self.embedded_input = tf.nn.xw_plus_b(self.input_x_onehot, self.W, self.b, name='embedded_input') # [None, 20]
        self.embedded_input = tf.nn.relu(self.embedded_input)
        embedded_input_expand = tf.expand_dims(self.embedded_input, axis = 1) # [None, 1, 20]
        self.embedded_input_all = tf.concat([embedded_input_expand for i in range(encode_length)], axis=1) # [None, 192, 20]

        #concat embedded_input_all and input_x
        self.lstm_input = tf.concat([self.input_x, self.embedded_input_all], axis=2) # [None, 192, 5+2+1(embedded one-hot)+(data)+(covariates)]
        self.cnn_input = self.input_x[:, :, 0] #shape = [batch_size, encode_length], pure data
        self.cnn_input = tf.expand_dims(self.cnn_input, axis=2)
        self.cnn_input = tf.expand_dims(self.cnn_input, axis=3) #[batch_size, encode_length, width=1, input_channel=1]
        print ("cnn_input.shape: ", self.cnn_input.shape)

        ### Conv-Deconv ###

        def CNN(self, batch_size,
                filter_size_list, dialations_size, input_channel_list, output_channel_list,
                de_filter_size_list, de_output_height, de_input_channel_list, de_output_channel_list,
                keep_prob, cnn_input, is_pred, ):
            #cnn_input.shape = [batch_size, height, width, input_channel]
            #num_filter is for every kernal
            cnn_input_list = [] # input for each layer
            cnn_input_list.append(cnn_input) # take raw value as input for the first layer
            for i, filter_size in enumerate(filter_size_list):
                with tf.variable_scope("CNN-"+str(i), reuse=is_pred):
                    filter_shape = [filter_size, 1, input_channel_list[i], output_channel_list[i]]
                    # parameter for conv
                    if not is_pred:
                        # Weights;
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
                        strides=[1,2,1,1],
                        padding = 'VALID', #不进行padding，多余部分丢弃
                        name='conv',
                        dilations = [1,dialations_size[i],1,1])
                    print (conv.shape)
                    h = tf.nn.relu(tf.nn.bias_add(conv,b), name = 'relu') # [batch_size, encode_length, width=1, ouput_channel]
                    if i!= (len(filter_size_list)-1):
                        h = tf.nn.dropout(h, keep_prob)
                    cnn_input_list.append(h)
            cnn_onput_temp = tf.squeeze(h, axis = 2)#[batch_size, num_feature, num_feature_maps] = [64, 21, 40]

            '''
            full_connect_output = []
            for i in range(cnn_onput_temp.shape[2]):
                with tf.variable_scope("fully_connect"+str(i), reuse=is_pred): #input shape: [batch_size, ]
                    if not is_pred:
                        W = tf.get_variable(name='W', shape= [cnn_onput_temp.shape[1],cnn_onput_temp.shape[1]],
                                            dtype = tf.float32, initializer=tf.random_normal_initializer(mean=0, stddev=0.1))
                        b = tf.get_variable(name='b', shape=[cnn_onput_temp.shape[1]] , dtype = tf.float32, initializer = tf.constant_initializer(value=0.1))
                    else:
                        W = tf.get_variable(name='W')
                        b = tf.get_variable(name='b')
                    full_connect_output_temp = tf.nn.xw_plus_b(cnn_onput_temp[:, :, i], W, b, name='full_connect_output_temp')#[batch_size, 21]
                    #print ("full_connect_output_temp.shape: ", full_connect_output_temp.shape)
                    full_connect_output.append(tf.expand_dims(full_connect_output_temp, axis=2))
            full_connect_output = tf.concat(full_connect_output, axis = 2) #[batch_size, 21, 40]
            # relu, remove if necessary
            full_connect_output = tf.nn.relu(full_connect_output, name = 'relu')
            print ("full_connect_output.shape: ", full_connect_output.shape)
            '''
            #deconv
            de_cnn_input_list = []
            de_cnn_input_list.append(tf.expand_dims(cnn_onput_temp, axis=2)) #take the last output from cnn as first input for decnn, shape = [batch_size, 21, 1, 80]
            #de_cnn_input_list.append(tf.expand_dims(full_connect_output, axis=2)) #take the last output from cnn as first input for decnn
            for i, de_filter_size in enumerate(de_filter_size_list):
                with tf.variable_scope("De_CNN-"+str(i), reuse=is_pred):
                    de_filter_shape = [de_filter_size, 1, de_output_channel_list[i],  de_input_channel_list[i]]
                    de_output_shape = [batch_size, de_output_height[i], 1, de_output_channel_list[i]]
                    if not is_pred:
                        # Weights
                        W = tf.get_variable(name='W', shape=de_filter_shape, dtype=tf.float32,initializer=tf.random_normal_initializer(mean=0, stddev=0.1))
                        b = tf.get_variable(name='b', shape=[de_output_channel_list[i]], dtype = tf.float32, initializer = tf.constant_initializer(value=0.1))
                        # inputs

                        batch_input_de_conv = de_cnn_input_list[i]
                    else:
                        # Weights
                        W = tf.get_variable(name='W')
                        b = tf.get_variable(name='b')
                        # inputs
                        batch_input_de_conv = de_cnn_input_list[i]
                    de_conv = tf.nn.conv2d_transpose(
                        batch_input_de_conv,  ### shape = [batch_size, encode_length]
                        W,
                        de_output_shape,
                        strides=[1,2,1,1],
                        padding = 'VALID', #不进行padding，多余部分丢弃
                        name='de_conv',
                        )
                    print (de_conv.shape)
                    de_h = tf.nn.relu(tf.nn.bias_add(de_conv,b), name = 'relu') # [batch_size, encode_length, width=1, ouput_channel]
                    if i!= (len(filter_size_list)-1):
                        de_h = tf.nn.dropout(de_h, keep_prob)
                    de_cnn_input_list.append(de_h)
            return tf.squeeze(de_h, axis=2) #[batch_size, input_length - 23, ouput_channel=40]

        filter_size_list = [2, 2, 2]
        dialations_size = [1, 1, 1]
        input_channel_list = [1,40,60]
        output_channel_list = [40,60,80]

        de_filter_size_list = [2, 2, 2]
        de_output_height = [42, 84, 168]
        de_input_channel_list = [80, 60, 40]
        de_output_channel_list = [60, 40, 40]

        with tf.variable_scope("RNN"): # Run CNN in RNN's scope only for reusing the variables in prediction range
            #cnn_output = CNN(self, filter_size_list, dialations_size, input_channel_list, output_channel_list, self.keep_prob, cnn_input=self.cnn_input, is_pred=False,)
            self.cnn_output = CNN(self, self.batch_size,
                filter_size_list, dialations_size, input_channel_list, output_channel_list,
                de_filter_size_list, de_output_height, de_input_channel_list, de_output_channel_list,
                keep_prob=self.keep_prob, cnn_input=self.cnn_input, is_pred=False, )
        print ("conv_deconv_output.shape: ", self.cnn_output.shape)
        #cnn_output_padding = tf.ones(shape = [self.batch_size, 23, hidden_unit], dtype= tf.float32) #padding for first 23 cnn_output
        #self.cnn_output = tf.concat([cnn_output_padding,cnn_output], axis=1)
        #print ("cnn_output.shape after padding: ", self.cnn_output.shape)
        self.lstm_input_with_cnn = tf.concat([self.cnn_output, self.lstm_input], axis = 2) #[batch_size, 168, 40+8]

        #self.embedded_input = tf.nn.relu(self.embedded_input)
        def lstm_cell(hidden_unit):
            cell = tf.contrib.rnn.LSTMCell(num_units = hidden_unit, forget_bias = 1.0,state_is_tuple=True)
            #cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.7)
            return cell

        def lstm_cell_drop_out(hidden_unit, keep_prob):
            cell = tf.contrib.rnn.LSTMCell(num_units = hidden_unit, forget_bias = 1.0,state_is_tuple=True)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
            return cell

        #Multi-layer LSTM
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
            for time_step in range(self.encode_length):
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
                #scale back
                miu_temp = tf.multiply(miu_temp, self.v) #还原scale
                miu.append(miu_temp)

            #concate miu in all timesteps
            self.miu_concat = tf.concat(miu, axis=1) # [None, window_size]

        with tf.variable_scope("Sigma"):
            sigma = []
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
                #scale back
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


##########################.   Train.     ###########################

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


        def ND_compute(mu, label, std):
            # from string to float
            label = label.astype(float)
            sum_subtract = np.absolute((mu - label))
            #two sigma
            mask_under_10 = (label<=10)
            mask = np.logical_and((mu-2*std)<=label, label<=(mu+2*std))
            mask = np.logical_and(mask, mask_under_10)
            mask = mask.astype(int)
            mask = np.abs(mask-1) #invert
            sum_subtract = sum_subtract * mask
            ND_list = []

            for i in range(sum_subtract.shape[0]): #compute ND for each series and then compute the average
                sum_subtract_temp = np.sum(sum_subtract[i])
                sum_label_temp = np.sum(np.abs(label[i]))
                sum_pred_temp = np.sum(np.abs(mu[i]))
                if sum_label_temp!=0:
                    #ND_temp = sum_subtract_temp/sum_label_temp
                    ND_temp =  sum_subtract_temp/((sum_label_temp+sum_pred_temp)/2.0)
                else:
                    if sum_subtract_temp==0:
                        ND_temp = 0
                    else:
                        print("nd!=0 but sum_label==0")
                        ND_temp =  sum_subtract_temp/((sum_label_temp+sum_pred_temp)/2.0)

                if (ND_temp>50):
                    print ("ND of this serie is HIGH", ND_temp)
                    print (sum_subtract_temp)
                    print (sum_label_temp)
                    print (sum_pred_temp)
                    print ("*************************")
                    ND_temp=0.25
                ND_list.append(ND_temp)

            ND = np.mean(np.array(ND_list))
            return ND



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
        #tf.add_to_collection("RMSE_train", self.RMSE_train)
        #tf.add_to_collection("ND_train", self.ND_train)
        ## Predict
        #tf.add_to_collection("miu_pred", self.miu_concat_pred)
        #tf.add_to_collection("sigma_pred", self.sigma_concat_pred)
        #tf.add_to_collection("RMSE_pred", self.RMSE_pred)
        #tf.add_to_collection("ND_pred", self.ND_pred)
        #tf.add_to_collection("hidden_states_all", self.hidden_states_all)
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

            _, step, neg_log_likelihood, miu_train, sigma_train  = sess.run([self.train_op,
                                                         self.global_step,
                                                         self.neg_log_likelihood,
                                                         self.miu_concat,
                                                         self.sigma_concat,
                                                         ],feed_dict = feed_dict)

            return (step, neg_log_likelihood, miu_train, sigma_train)

        def pred_step(x_batch,onehot_batch, y_batch,v_batch, batch_size,forget_gate_mask,keep_prob=1.0):

            feed_dict = {
            self.input_x: x_batch,
            self.input_x_onehot: onehot_batch,
            self.input_y: y_batch,
            self.v :v_batch,
            self.batch_size: batch_size,
            self.forget_gate_mask: forget_gate_mask,
            self.keep_prob: keep_prob,
            }

            neg_log_likelihood, miu_train, sigma_train  = sess.run([
                                                         self.neg_log_likelihood,
                                                         self.miu_concat,
                                                         self.sigma_concat,
                                                         ],feed_dict = feed_dict)
            return (neg_log_likelihood, miu_train, sigma_train)


        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            #创建saver对象
            saver = tf.train.Saver()
            forget_gate_mask_sample = np.full((64, self.encode_length, self.hidden_unit), 1, dtype = np.float32)

            print ("###########     TIME     ##########")
            print (datetime.datetime.now())
            print ("###########  TRAIN START ############")

            for j in range(20): #epochs
                for index in self.index_list:
                    input_x_batch = self.input_x_all[index] #[batch_size, 192, 3]
                    input_y_batch = self.input_y_all[index]
                    input_onehot_batch = self.input_x_onehot_all[index]
                    input_v_batch = self.v_all[index]
                    batch_size = len(index)
                    forget_gate_mask_batch = forget_gate_mask_sample #[batch_size, 40]

                    for time_step_temp in range(self.encode_length, self.window_size):
                        input_x_batch_temp = input_x_batch[:, time_step_temp-168:time_step_temp, :]
                        input_y_batch_temp = input_y_batch[:, time_step_temp-168:time_step_temp]
                        input_onehot_batch_temp = input_onehot_batch
                        input_v_batch_temp = input_v_batch

                        step, neg_log_likelihood, miu_train, sigma_train = train_step(input_x_batch_temp,
                                                                        input_onehot_batch_temp,
                                                                        input_y_batch_temp,
                                                                        input_v_batch_temp,
                                                                        batch_size,
                                                                        forget_gate_mask_batch,
                                                                        keep_prob = 0.7,
                                                                        )

                    if step%2000==0:
                        print ("prediction")
                        ### prediction ###
                        neg_log_likelihood_pred_sum=0.0
                        RMSE_pred_sum=0.0
                        ND_pred_sum = 0.0

                        for index_pred in self.indexs_pred:
                            # prepare data for prediction
                            input_x_batch_pred = self.input_x_all[index_pred] #[batch_size, 192, 3]
                            input_y_batch_pred = self.input_y_all[index_pred] #[batch_size, 192]
                            input_onehot_batch_pred = self.input_x_onehot_all[index_pred]
                            input_v_batch_pred = self.v_all[index_pred]
                            batch_size_pred = len(index_pred)
                            forget_gate_mask_batch_pred =  forget_gate_mask_sample
                            # output in prediction range
                            miu_pred_output_list = []
                            sigma_pred_output_list = []
                            neg_log_likelihood_temp_list = []
                            for time_step_temp_pred in range(self.encode_length, self.window_size):
                                input_x_batch_pred_temp = input_x_batch_pred[:, time_step_temp_pred-168:time_step_temp_pred, :] #[batch_size, 168, 4]
                                input_y_batch_pred_temp = input_y_batch_pred[:, time_step_temp_pred-168:time_step_temp_pred]
                                input_onehot_batch_pred_temp = input_onehot_batch_pred
                                input_v_batch_pred_temp = input_v_batch_pred #[batch_size, 1]
                                # update input_x after the first iteration
                                if time_step_temp_pred != self.encode_length: input_x_batch_pred_temp[:, -1, 0] = next_input_x

                                neg_log_likelihood_pred, miu_pred, sigma_pred = pred_step(input_x_batch_pred_temp,
                                                                                             input_onehot_batch_pred_temp,
                                                                                             input_y_batch_pred_temp,
                                                                                             input_v_batch_pred_temp,
                                                                                             batch_size,
                                                                                             forget_gate_mask_batch_pred,
                                                                                             keep_prob = 1.0)

                                next_input_x = 	miu_pred[:, -1] #shape=[batch_size, ]
                                #rescale
                                next_input_x = np.divide(next_input_x, np.squeeze(input_v_batch_pred_temp, axis=1))
                                miu_pred_output_list.append(np.expand_dims(miu_pred[:, -1], axis=1)) #shape=[batch_size, 1]
                                sigma_pred_output_list.append(np.expand_dims(sigma_pred[:, -1], axis=1)) #shape=[batch_size, 1]
                                neg_log_likelihood_temp_list.append(neg_log_likelihood_pred)
                            miu_pred_output_list = np.concatenate(miu_pred_output_list, axis=1) #[batch_size, decode_length]=[64, 24]
                            sigma_pred_output_list = np.concatenate(sigma_pred_output_list, axis=1) #[batch_size, decode_length]=[64, 24]
                            #compute likelihood of current series
                            neg_log_likelihood_pred = np.mean(np.array(neg_log_likelihood_temp_list))
                            #compute ND of current series
                            ND_pred = ND_compute(miu_pred_output_list, input_y_batch_pred[:, self.encode_length:], sigma_pred_output_list)
                            #compute RMSE of current series
                            #RMSE_pred = RMSE_compute()
                            neg_log_likelihood_pred_sum += neg_log_likelihood_pred
                            #RMSE_pred_sum += RMSE_pred
                            ND_pred_sum += ND_pred

                        neg_log_likelihood_pred_avg = neg_log_likelihood_pred_sum/(len(self.indexs_pred))
                        #RMSE_pred_avg = RMSE_pred_sum/(len(self.indexs_pred))
                        ND_pred_avg = ND_pred_sum/(len(self.indexs_pred))
                        miu_pred = np.concatenate([input_y_batch_pred[:, :self.encode_length],miu_pred_output_list], axis=1 )
                        sigma_pred = np.concatenate([np.ones([64, 168], dtype=np.float32),sigma_pred_output_list ], axis=1)
                        print ("step:", step)
                        #print ("neg_log_likelihood_train: ", neg_log_likelihood)
                        print ("neg_log_likelihood_prediction: ", neg_log_likelihood_pred_avg)
                        #print ("RMSE_train: ", RMSE_train)
                        #print ("RMSE_prediction: ", RMSE_pred_avg)
                        #print ("ND_train: ", ND_train)
                        print ("ND_pred: ", ND_pred_avg)
                        saver.save(sess, '../../checkpoint/checkpoint_conv-deconv_Fuahui_v_40/DeepAR_model',global_step = step)
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
'''
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
'''

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
