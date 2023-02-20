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
        
        f1 = f1_score(true_list, pred_list, average='macro')
        precision = precision_score(true_list, pred_list, average="macro")
        recall = recall_score(true_list, pred_list, average="macro")

        print(confusion_matrix(true_list, pred_list))
        print(f"f1 {f1} / precision {precision} / recall {recall}")

        return f1, precision, recall