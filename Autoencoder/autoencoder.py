'''
Chen Song, Nuan Wen, Yunkai Zhang
8/22/2018
'''
from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import data_helper
import matplotlib.pyplot as plt

# Training Parameters
learning_rate = 0.01
num_steps = 30000
batch_size = 64

display_step = 1000
examples_to_show = 10

# Network Parameters
num_hidden_1 = 16 # 1st layer num features
num_hidden_2 = 12 # 2nd layer num features (the latent dim)
num_input = 24 # period of electricity
window_size = 192

# import electricity data
with tf.device('/cpu:0'):
	(shift_train_data,
 	shift_train_onehot,
 	v_all,
 	shift_train_label,
 	param,
 	index_list,
 	indexs_pred_list) = data_helper.prepare(training = True)

train_data = shift_train_data

def next_batch(num, data):
    idx = np.random.choice(data.shape[0], num)
    window_idx = np.random.randint(window_size, size=num)
    data_shuffle = [data[idx[i], window_idx[i]+1:window_idx[i]+25, 0] for i in range(idx.size)]
    return np.asarray(data_shuffle)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, num_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([num_input])),
}

# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, tf.transpose(weights['encoder_h2'])),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, tf.transpose(weights['encoder_h1'])),
                                   biases['decoder_b2']))
    return layer_2

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start Training
# Start a new TF session
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training
    for i in range(1, num_steps+1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x = next_batch(batch_size, train_data)

        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))

    # Testing
    # Encode and decode images from test set and visualize their reconstruction.
    n = 4
    for i in range(n):
        # MNIST test set
        batch_x = next_batch(n, train_data)
        # Encode and decode the digit image
        g = sess.run(decoder_op, feed_dict={X: batch_x})

    print("Original series: ")
    print(batch_x)

    print("Reconstructed series: ")
    print(g)
