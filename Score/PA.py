import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

def PA(true_list, errors):
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

## TadGAN
# errors, predictions_vs = reconstruction_errors(x_real, x_hat, score_window=self.args.seq_len, step_size=1) #score_window=config.window_size
# error_range = find_anomalies(errors, index=range(len(errors)), anomaly_padding=5)
# pred_list = np.zeros(len(true_list))
# for i in error_range:
#     start = int(i[0])
#     end = int(i[1])
#     pred_list[start:end] = 1

# drawing(config, anomaly_value, pd.DataFrame(test_dataset.scaled_test))

## batch 안에서 cosine similarity
# diff = cos(x, pred).cpu().tolist()

# batch_pred = np.where(((np.array(diff)<0) & (np.array(diff)>-0.1)), 1, 0)
# batch_pred = np.where(np.array(diff)<0.7, 1, 0)
# y = np.where(y>0.69, 1, 0)

# diffs.extend(diff)
# pred_list.extend(batch_pred)
        