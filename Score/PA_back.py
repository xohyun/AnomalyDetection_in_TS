import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

class PA_back():
    def score(self, true_list, errors):
        l_quantile = np.quantile(np.array(errors), 0.025)
        u_quantile = np.quantile(np.array(errors), 0.975)
        in_range = np.logical_and(
            np.array(errors) >= l_quantile, np.array(errors) <= u_quantile)
        # pred_list = [0 for i in errors if i in in_range]
        np_errors = np.array(errors)
        # pred_list = [i for i in np_errors if i in np.where((i >= l_quantile and i <= u_quantile), 0, 1)]
        pred_list = np.zeros(len(errors))
        for i in range(len(np_errors)):
            if errors[i] >= l_quantile and errors[i] <= u_quantile:
                pred_list[i] = 0
            else:
                pred_list[i] = 1
        
        # pred_list_back = [0 for i in range(int(5*len(pred_list)/50)+50)]
        # true_list_back = [0 for i in range(int(5*len(pred_list)/50)+50)]
        # pred_list_back = np.zeros(int(5*len(pred_list)/50)+50)
        # true_list_back = np.zeros(int(5*len(pred_list)/50)+50)
        pred_list = np.array(pred_list)
        true_list = np.array(true_list)
        pred_list_back = np.zeros(68468)
        true_list_back = np.zeros(68468)       
        for i in range(684690):
            pred_list_back[(10*i):(10*i+10)] = pred_list[(50*i):(50*i+10)]
            true_list_back[(10*i):(10*i+10)] = true_list[(50*i):(50*i+10)]

        f1 = f1_score(true_list, pred_list, average='macro')
        precision = precision_score(true_list, pred_list, average="macro")
        recall = recall_score(true_list, pred_list, average="macro")

        print(confusion_matrix(true_list, pred_list))
        print(f"f1 {f1} / precision {precision} / recall {recall}")

        return f1, precision, recall


# 참고용으로 가져다놓음
def get_anomaly_time(original, prediction) : 
    temp = np.zeros(shape=(68468,), dtype=np.float32)
    original = original.squeeze(axis = 1)
    stride = 10
    window_size = 50

    for i in range(len(prediction)) :
        if prediction[i] == 0 :
            temp[i*stride : (i*stride + window_size)] = np.nan
        elif prediction[i] == 1 : # anomaly
            temp[i*stride : (i*stride + window_size)] = original[i*stride : (i*stride + window_size)]
    return temp