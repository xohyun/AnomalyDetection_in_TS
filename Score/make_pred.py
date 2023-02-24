import numpy as np
import torch


class Pred_making():
    '''
    point adjust with quantile
    '''

    def quantile_score(self, true_list, errors):
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

        true_list = np.array(true_list)
        pred_list = np.array(pred_list)

        return true_list, pred_list

    def distance_var_calc_score(self, true_list, dist_list):
        # # target-feature 만의 variace를 계산해야 할 것인가 전체-feature를 할 것인가?
        # # feature 간 또는 sequence 순서에 따른 모든 변동성에 차이를 고려할 것인가?
        #  ** threshold =
        # print(dist_list.shape)
        # forecast_variances = torch.var(dist_list[:, 0], dim=1).numpy()
        # # numpy array 라면 가로축(시간축)에 따라 각 feature 표준편차 계산
        # # forecast_variances = np.var(dist_list[:,0], axis=-1, ddof=1)
        # dist = np.sqrt((dist_list[:, 1] - forecast_variances)**2)

        #  ** pred_list = [n for n in dist if n > threshold else 0]

        # # return true_list, pred_list
        pass


# TadGAN
# errors, predictions_vs = reconstruction_errors(x_real, x_hat, score_window=self.args.seq_len, step_size=1) #score_window=config.window_size
# error_range = find_anomalies(errors, index=range(len(errors)), anomaly_padding=5)
# pred_list = np.zeros(len(true_list))
# for i in error_range:
#     start = int(i[0])
#     end = int(i[1])
#     pred_list[start:end] = 1

# drawing(config, anomaly_value, pd.DataFrame(test_dataset.scaled_test))

# batch 안에서 cosine similarity
# diff = cos(x, pred).cpu().tolist()

# batch_pred = np.where(((np.array(diff)<0) & (np.array(diff)>-0.1)), 1, 0)
# batch_pred = np.where(np.array(diff)<0.7, 1, 0)
# y = np.where(y>0.69, 1, 0)

# diffs.extend(diff)
# pred_list.extend(batch_pred)
