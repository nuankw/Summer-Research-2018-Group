import numpy as np
import matplotlib.pylab as plt
import ruptures as rpt
import data_helper


(shift_train_data, shift_train_onehot, v_all, shift_train_label,
param,index_list,indexs_pred_list) = data_helper.prepare(training = True)
print(shift_train_data.shape)
n, dim = 500, 1  # number of samples, dimension
n_bkps, sigma = 3, 5  # number of change points, noise standart deviation
signal = shift_train_data[700, :, 0]
print(type(signal))
print("signal shape: ", signal.shape)
print(signal.shape)
# change point detection
model = "l1"  # "l1", "rbf", "linear", "normal", "ar"
algo = rpt.Window(width=10, model=model).fit(signal)
sigma = np.std(signal)
my_bkps = algo.predict(pen=np.log(192)*1*sigma**2)
print("my_bkps: ", my_bkps)
# show results
rpt.show.display(signal, my_bkps, figsize=(10, 6))
plt.show()
