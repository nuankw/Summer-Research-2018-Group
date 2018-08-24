for i, filter_size in enumerate(filter_size): # filter_size=[3,4,5]
			with tf.name_scope('conv-'+ str(filter_size)):
				num_padding = filter_size-1
				pad = tf.zeros([self.batch_size, num_padding, embedding_size, 1], tf.float64)
				batch_input_conv_pad = tf.concat([batch_input_conv, pad], 1) #第二个param是concate的维度
				filter_shape = [filter_size, embedding_size, 1, num_filters] #[filter_height, filter_width, in_channels, out_channels]
				# parameter for conv
				W = tf.Variable(
					tf.truncated_normal(filter_shape, stddev=0.1), name='W')
				### ??? ###
				b = tf.Variable(
					tf.constant(0.1, shape = [num_filters]), name='b')
				#print (batch_input_conv_pad.shape)
				self.conv = tf.nn.conv2d(
					batch_input_conv_pad,
					W,
					strides=[1,1,1,1],
					padding = 'VALID', #不进行padding，多余部分丢弃
					name='conv') #conv.shape: [batch_size, sequence_length(因为stride是1),1, num_filters(也就是feature maps的个数)]
				#激活函数
				h = tf.nn.relu(tf.nn.bias_add(self.conv,b), name = 'relu')
				self.pooled = tf.nn.max_pool(
					h, #输入数据的size：[batch, height, width, channels]
					ksize = [1, max_pool_size, 1,1],# 池化窗口的size：[1, height, width, 1]
					strides= [1,2,1,1],#步长：[1, stride_vertical, stride_horizontal, 1]
					padding='SAME'
					)
				self.real_len = np.int32(np.ceil(sequence_length/2.0)) #pool后长度
				print ('real_len: ',self.real_len)
				#reduced = np.int32()
				self.squeezed = tf.squeeze(self.pooled, [2])
				pooled_all.append(self.squeezed)