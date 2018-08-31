"""
Chen Song, Nuan Wen, Yunkai Zhang
08/31/2018
"""
from __future__ import division, print_function, absolute_import
from scipy.signal import argrelextrema
from scipy.signal import find_peaks_cwt
import tensorflow as tf
import numpy as np
import math
import sys
import matplotlib.pyplot as plt

learning_rate = 0.01
num_steps = 30000
batch_size = 64
display_step = 1000
num_hidden_1 = 16
num_hidden_2 = 12
num_input = 24 # period of electricity
window_size = 192
predict_size = 24
average_padding = 12 #number of points to take average of when padding the zeros after convolving
training = True

def next_batch(num, data):
    idx = np.random.choice(data.shape[0], num)
    window_idx = np.random.randint(window_size-num_input, size=num)
    data_shuffle = [data[idx[i], window_idx[i]:window_idx[i]+num_input, 0] for i in range(idx.size)]
    return np.asarray(data_shuffle)

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

def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2

def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, tf.transpose(weights['encoder_h2'])),
                                   biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, tf.transpose(weights['encoder_h1'])),
                                   biases['decoder_b2']))
    return layer_2

encoder_op = encoder(X)
decoder_op = decoder(encoder_op)
y_pred = decoder_op
y_true = X

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

def compute_p(small_set = False):
    # import electricity data
    if(small_set):
		train_data = np.load('./data/elect_train_data.npy')
    else:
        train_data = np.load('./data/elect_data.npy')

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(num_steps):
            # Return the next batch of eletricity window
            batch_x = next_batch(batch_size, train_data)
            _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
            if i % display_step == 0 or i == 1:
                print('Step %i has loss: %f' % (i, l))

        print("= = = = =Testing= = = = =")
        # Encode and decode windows and check their differences after reconstruction
        n = 4 #number of samples to show
        for i in range(n):
            batch_x = next_batch(n, train_data)
            recons = sess.run(decoder_op, feed_dict={X: batch_x})

        print("Original series: ")
        print(batch_x)
        print("Reconstructed series: ")
        print(recons)

        #compute p for all the windows
        print("= = = = =Now computing p values= = = = =")
        p_features = np.zeros([train_data.shape[0], window_size-num_input-predict_size+1, num_hidden_2]) #encoded windows
        p_value = np.zeros([train_data.shape[0], window_size]) #distance between adjacent windows
        for i in range(train_data.shape[0]):
            if(i % 50 == 0):
                sys.stdout.write('\r')
                sys.stdout.write('{} percent complete.'.format(round(i*1.0/train_data.shape[0]*100, 2)))
                sys.stdout.flush()
            p_features[i, 0] = sess.run(encoder_op, feed_dict={X: train_data[i, 0:num_input, 0].reshape(1, num_input)})
            for j in range(1, window_size-num_input-predict_size):
                p_features[i, j] = sess.run(encoder_op, feed_dict={X: train_data[i, j:j+num_input, 0].reshape(1, num_input)})
                p_value[i,j-1+num_input] = np.linalg.norm(p_features[i,j-1]-p_features[i,j], 2)*1.0/math.sqrt(np.linalg.norm(p_features[i,j], 2)*np.linalg.norm(p_features[i,j-1], 2))
            p_value[i, :num_input] = np.average(p_value[i, num_input:num_input+average_padding])
            p_value[i, window_size-predict_size-1:] = np.average(p_value[i, window_size-predict_size-1-average_padding:window_size-predict_size-1])
    if(small_set):
        np.save("./data/elect_train_p_value", p_value)
    else:
        np.save("./data/elect_p_value", p_value)
    return (train_data, p_value)

#convolve time series y to make it smoother
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def label_points_original(train_data, p_value):
    change_points = np.zeros([train_data.shape[0]])
    debug_points = np.zeros([train_data.shape[0], window_size])
    for i in range(train_data.shape[0]):
        average = np.average(p_value[i, :])
        if average != 0:
            p_conv = np.zeros(window_size)
            p_conv = smooth(p_value[i, :], (int)(num_input*3/4.0))
            average = np.average(p_conv)
            std = np.std(p_conv)
            limit = average + std
            relative_max = find_peaks_cwt(p_conv[num_input:window_size-predict_size], np.arange(1, 12))
            max_len = len(relative_max)-1
            second_max = False
            first_index = -1
            while max_len >= 0:
                if p_conv[relative_max[max_len]] > limit:
                    debug_points[i, relative_max[max_len]] = 1
                    if second_max == True:
                        if first_index-relative_max[max_len] > (int)(num_input/4.0*5):
                            max_len = -1
                        elif abs(first_index-relative_max[max_len]-num_input) <= (int)(num_input/4.0):
                            change_points[i] = first_index
                    else:
                        second_max = True
                        first_index = relative_max[max_len]
                        change_points[i] = first_index
                max_len -= 1
                if max_len == -1 and second_max == True:
                    change_points[i] = first_index
    return change_points, debug_points

def label_points_custom(train_data, p_value):
    change_points = np.zeros([train_data.shape[0]])
    debug_points = np.zeros([train_data.shape[0], window_size])
    for i in range(train_data.shape[0]):
        average = np.average(p_value[i, :])
        if average != 0:
            p_conv = np.zeros(window_size)
            encountered = False
            p_conv = smooth(p_value[i, :], num_input)
            average = np.average(p_conv)
            std = np.std(p_conv)
            limit = average - std
            relative_min = argrelextrema(p_conv[num_input:window_size-predict_size], np.less, order=(int)(num_input/3.0))[0]+num_input
            relative_max = argrelextrema(p_conv[num_input:window_size-predict_size], np.greater, order=(int)(num_input/3.0))[0]+num_input
            for j in range(len(relative_min)-1, -1, -1):
                if p_conv[relative_min[j]]+std/2 < p_conv[relative_min[j]+(int)(num_input/3.0)] and p_conv[relative_min[j]]+std/2 < p_conv[relative_min[j]-(int)(num_input/3.0)]:
                    debug_points[i, relative_min[j]] = 1
                    if encountered == False and relative_min[j] != window_size-predict_size-1:
                        change_points[i] = relative_min[j] + (int)(num_input/3.0)
                        encountered = True
    return change_points, debug_points

'''For testing purposes'''
#(train_data, p_value) = compute_p(small_set = True)
if training:
    shift_train_data = np.load("./data/elect_train_data.npy")
    shift_train_onehot = np.load("./data/elect_train_onehot.npy")
    v_all = np.load("./data/elect_train_v.npy")
    shift_train_label = np.load("./data/elect_train_label.npy")
    param = np.load("./data/elect_train_param.npy")
    index_list = np.load("./data/elect_train_index.npy")
    indexs_pred = np.load("./data/elect_train_pred_index.npy")
    p_value = np.load('./data/elect_train_p_value.npy')
else:
    shift_train_data = np.load("./data/elect_data.npy")
    shift_train_onehot = np.load("./data/elect_onehot.npy")
    v_all = np.load("./data/elect_v.npy")
    shift_train_label = np.load("./data/elect_label.npy")
    param = np.load("./data/elect_param.npy")
    index_list = np.load("./data/elect_index.npy")
    indexs_pred = np.load("./data/elect_pred_index.npy")
    p_value = np.load('./data/elect_p_value.npy')
x = np.arange(window_size)
figure_ind = np.random.randint(shift_train_data.shape[0], size = 50)
change_points, debug_points= label_points_custom(shift_train_data[figure_ind], p_value[figure_ind])

with tf.Session() as sess:
    checkpoint_dir = "./checkpoint/"
    #ckpt contains all the checkpoint info
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    print(ckpt)
    #ckpt.model_checkpoint_path is the name of the new checkpoint, add .meta to load the graph
    if (ckpt and ckpt.model_checkpoint_path):
        #load the computation graph
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path+".meta")
        #load parameters
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("No model can be loaded!")

    train_op = tf.get_collection('train_op')[0]
    label = tf.get_collection("label")[0]

    miu_train = tf.get_collection("miu_train")[0]
    sigma_train = tf.get_collection("sigma_train")[0]
    RMSE_train = tf.get_collection("RMSE_train")[0]
    ND_train = tf.get_collection("ND_train")[0]

    miu_pred = tf.get_collection("miu_pred")[0]
    sigma_pred = tf.get_collection("sigma_pred")[0]
    RMSE_pred = tf.get_collection("RMSE_pred")[0]
    ND_pred = tf.get_collection("ND_pred")[0]

    input_x_batch_pred = shift_train_data[figure_ind]
    input_onehot_batch_pred =shift_train_onehot[figure_ind]
    input_v_batch_pred = v_all[figure_ind]
    input_y_batch_pred = shift_train_label[figure_ind]
    batch_size_pred = len(figure_ind)
    feed_dict ={'input_x:0': input_x_batch_pred, 'input_onehot:0': input_onehot_batch_pred, 'input_y:0': input_y_batch_pred, 'input_v:0' :input_v_batch_pred, 'input_batch:0': batch_size_pred}
    RMSE_test, ND_test, miu_test, sigma_test = sess.run([RMSE_pred, ND_pred, miu_pred, sigma_pred], feed_dict = feed_dict)
    for i in range(50):
        f = plt.figure()
        plt.subplot(2, 1, 1)
        label_temp = input_y_batch_pred[i,:]
        pred_temp = miu_test[i,:]
        plt.plot(x,np.divide(np.float32(label_temp),input_v_batch_pred[i,0]), color='b')
        plt.plot(x[-predict_size:],np.divide(np.float32(pred_temp[-predict_size:]),input_v_batch_pred[i,0]), color='r')
        plt.scatter(x, debug_points[i], color = 'y', marker = 'x')
        plt.axvline(window_size-predict_size, color='k', linestyle = '-')
        plt.axvline(change_points[i], color='g', linestyle = "dashed")
        plt.ylabel('electricity')
        conv = smooth(p_value[figure_ind[i], :], num_input)
        plt.subplot(2, 1, 2)
        plt.plot(x, p_value[figure_ind[i], :], 'r-')
        plt.plot(x, conv, 'g-')
        plt.xlabel('time')
        plt.ylabel('p-value')
        average_conv = np.average(conv[num_input:window_size-predict_size])
        std_conv = np.std(conv[num_input:window_size-predict_size])
        plt.axhline(average_conv-std_conv, color='y', linestyle = "dashed")
        plt.axhline(average_conv, color='c', linestyle = "dashed")
        f.savefig("elect_pvalue_"+str(i)+".png")
        plt.close()
