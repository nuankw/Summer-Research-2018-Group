import numpy as np
import matplotlib.pylab as plt
import ruptures as rpt
import data_helper

training = True #change to False to run on the whole dataset

shift_train_data = np.load("./data/elect_pre_train_data.npy")
shift_train_onehot = np.load("./data/elect_train_onehot.npy")
v_all = np.load("./data/elect_train_v.npy")
shift_train_label = np.load("./data/elect_train_label.npy")
param = np.load("./data/elect_train_param.npy")
index_list = np.load("./data/elect_train_index.npy")
indexs_pred = np.load("./data/elect_train_pred_index.npy")

shift_train_pvalue = [] #change points for each window

n, dim = 192, 1  # number of samples, dimension
model = "rbf"  # "l1", "rbf", "linear", "normal", "ar"
for i in range(shift_train_data.shape[0]):
    change_points = np.zeros(192)
    signal = shift_train_data[i, :, 0]
    # change point detection
    algo = rpt.Window(width=48, model=model).fit(signal[:-24])
    sigma = np.std(signal)
    my_bkps = algo.predict(pen=2*np.log(n)*dim*sigma**2)
    for j in my_bkps:
        if j < 160:
            change_points[j-1] = 1
    # save results
    shift_train_pvalue.append(change_points)
shift_train_pvalue = np.array(shift_train_pvalue)
if(training):
    np.save("./data/elect_train_p_value", shift_train_pvalue)
else:
    np.save("./data/elect_p_value", shift_train_pvalue)
