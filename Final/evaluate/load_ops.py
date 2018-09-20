import numpy as np
import matplotlib.pyplot as plt

def plot(label, label_f, prediction, sigma_test, prediction_f, sigma_test_f, window_size, num_plot = 8, 
            y_range = None, y_range_f = None, forget_gate=np.array([[-1],[-1]]), forget_gate_f=np.array([[-1],[-1]]), save_num=None):
    x = np.arange(192)
    pred_minus_sigma = prediction - sigma_test
    pred_minus_sigma_f = prediction_f - sigma_test_f
    pred_plus_sigma = prediction + sigma_test
    pred_plus_sigma_f = prediction_f + sigma_test_f
    f = plt.figure(figsize=(50,32))

    base_1 = num_plot/2
    base_2 = 2

    for i in range(int(num_plot/2)):

        label_temp = label[i].reshape([window_size,])
        label_temp_f = label_f[i].reshape([window_size,])
        pred_temp = prediction[i].reshape([window_size,])
        pred_temp_f = prediction_f[i].reshape([window_size,])
        if forget_gate[0,0]!=-1 :
            forget_gate_temp = forget_gate[i].reshape([window_size,])
        if forget_gate_f[0,0]!=-1 :
            forget_gate_temp_f = forget_gate_f[i].reshape([window_size,])
                
        # no forget_gate

        plt.subplot(base_1, base_2, i*2+1)
        labels, = plt.plot(x,label_temp, color='b')
        preds, = plt.plot(x,pred_temp, color='r')
        stddev = plt.fill_between(x, pred_minus_sigma[i].reshape([window_size,]) ,
                                 pred_plus_sigma[i].reshape([window_size,])  ,
                                 color='blue',
                                 alpha=0.2)
        if forget_gate[0,0]!=-1 :
            forget_gates, = plt.plot(x, forget_gate_temp, color='g')
            plt.legend(handles = [labels, preds, stddev, forget_gates], labels = ['Ground_truth', 'Prediction', 'Single stddev ', 'forget_gate'], loc = 'upper left', fontsize =12)
        else:
            plt.legend(handles = [labels, preds, stddev], labels = ['Ground_truth', 'Prediction', 'Single stddev '], loc = 'upper left', fontsize =12)
        plt.axvline(168, color='k', linestyle = "dashed")
        #参考线
        ax = plt.gca()
        ax.tick_params(axis = 'x', which = 'major', labelsize = 24)
        ax.tick_params(axis = 'y', which = 'major', labelsize = 24)
        axes = plt.gca()
        if y_range:  axes.set_ylim(y_range)

        # forget_gate

        plt.subplot(base_1, base_2, i*2+2)
        labels, = plt.plot(x,label_temp_f, color='b')
        preds, = plt.plot(x,pred_temp_f, color='r')
        stddev_f = plt.fill_between(x, pred_minus_sigma_f[i].reshape([window_size,]) ,
                                 pred_plus_sigma_f[i].reshape([window_size,])  ,
                                 color='blue',
                                 alpha=0.2)
        if forget_gate_f[0,0]!=-1:
            forget_gates, = plt.plot(x, forget_gate_temp_f, color='g')
            plt.legend(handles = [labels, preds, stddev_f, forget_gates], labels = ['Ground_truth', 'Prediction', 'Single stddev ', 'forget_gate'], loc = 'upper left', fontsize =12)
        else:
            plt.legend(handles = [labels, preds, stddev_f], labels = ['Ground_truth', 'Prediction', 'Single stddev '], loc = 'upper left', fontsize =12)
        plt.axvline(168, color='k', linestyle = "dashed")
        #参考线
        ax = plt.gca()
        ax.tick_params(axis = 'x', which = 'major', labelsize = 24)
        ax.tick_params(axis = 'y', which = 'major', labelsize = 24)
        axes = plt.gca()
        if y_range:  axes.set_ylim(y_range_f)
        if save_num: f.savefig(str(save_num)+".png")

def compute_new_v(values):
    v = np.mean(values)+1
    return v
    
def rescale(values, v_old, v_new):
    values_temp = values*v_old
    return np.true_divide(values_temp, v_new)

def compute_forget_index(p_values): 
    #compute the index of the last change point
    index_list = []
    for serie in range(p_values.shape[0]):
        for i in reversed(range(p_values.shape[1])): #from 191 to 0
            if p_values[serie,i,0]==0: 
                index_list.append(i)
                break
            elif i==0:
                index_list.append(-1)
    index = np.array(index_list)
    return index #shape = [64,]
                
def compute_hidden_states_relation(serie, i, window_size): #serie.shape: [window_size, hidden_unit]
    if i==0 or i==window_size-1:
        return 0
    dot_product_temp = np.dot(serie[i], serie[i+1])
    #dot_product_temp = np.mean(serie[i])
    return dot_product_temp

              
def compute_forget_gate_relation(serie, i,window_size): #serie.shape: [window_size, hidden_unit]
    if i==0 or i==window_size-1:
        return 0
    #dot_product_temp = np.dot(serie[i], serie[i+1])
    dot_product_temp = np.linalg.norm(serie[i])
    return dot_product_temp


def norm(value):
    mean = np.mean(value)
    if mean !=0:
        return value/mean
    else:
        return value


# compute ND and RMSE
def RMSE_compute(mu, label, std, encode_length, decode_length, window_size):
    mu_pred = mu[:, encode_length:window_size]
    label_pred = label[:, encode_length:window_size]
    std_pred = std[:, encode_length:window_size]
    
    RMSE = np.square((label_pred - mu_pred))
    norm = np.absolute(label_pred)
    norm = np.mean(norm)
    norm_pred = np.absolute(mu_pred)
    norm_pred = np.mean(norm_pred)
    
    #two sigma
    mask_under_10 = label_pred<=10
    mask = np.logical_and((mu_pred-2*std_pred)<label_pred, label_pred<(mu_pred+2*std_pred))
    mask = np.logical_and(mask, mask_under_10)
    mask = mask.astype(int)
    mask = np.absolute(mask-1) #invert
    RMSE = RMSE * mask
    #print (mask)
    
    RMSE = np.sqrt(np.mean(RMSE))
    if norm!=0:
        RMSE_norm = RMSE/norm
        return RMSE_norm
    else:
        if RMSE==0:
            return 0
        else:
            RMSE_norm = RMSE/((norm_pred+norm)/2.0)
            return RMSE_norm

def ND_compute(index_pred_number, mu, label,std, encode_length, decode_length, window_size):
    mu_pred = mu[:, encode_length:window_size] 
    label_pred = label[:, encode_length:window_size]
    std_pred = std[:, encode_length:window_size]
    sum_subtract = np.absolute((mu_pred-label_pred))
    
    #two sigma
    mask_under_10 = label_pred<=10
    mask = np.logical_and((mu_pred-2*std_pred)<=label_pred, label_pred<=(mu_pred+2*std_pred))
    mask = np.logical_and(mask, mask_under_10)
    mask = mask.astype(int)
    mask = np.abs(mask-1) #invert
    sum_subtract = sum_subtract * mask
    ND_list = []
    
    for i in range(sum_subtract.shape[0]): #compute ND for each series and then compute the average
        sum_subtract_temp = np.sum(sum_subtract[i])
        sum_label_temp = np.sum(np.abs(label_pred[i]))
        sum_pred_temp = np.sum(np.abs(mu_pred[i]))
        if sum_label_temp!=0:
            ND_temp = sum_subtract_temp/sum_label_temp
            #ND_temp =  sum_subtract_temp/((sum_label_temp+sum_pred_temp)/2.0)
        else:
            if sum_subtract_temp==0:
                ND_temp = 0
            else:
                print("nd!=0 but sum_label==0")
                ND_temp =  sum_subtract_temp/((sum_label_temp+sum_pred_temp)/2.0)
        
        if (ND_temp>50):
            print ("index_pred_number: ", index_pred_number)
            print ("index_number: ", i)
            print ("ND of this serie is HIGH", ND_temp)
            print (sum_subtract_temp)
            print (sum_label_temp)
            print (sum_pred_temp)
            print ("*************************")
            ND_temp=0.25
        ND_list.append(ND_temp)
            
        
    ND = np.mean(np.array(ND_list))
    
    return ND