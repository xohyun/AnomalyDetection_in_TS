import numpy as np
import torch


class Pred_making():
    '''
    point adjust with quantile
    '''

    def quantile_score(self, true_list, errors):
        l_quantile = np.quantile(np.array(errors), 0.0) # 0.025
        u_quantile = np.quantile(np.array(errors), 0.95) # 0.975
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
    
    def _fixed_threshold(errors, k=3):
        """
        Calculate the threshold. (TadGAN)
        The fixed threshold is defined as k standard deviations away from the mean.
        Args:
            errors (ndarray):
                Array of errors.
        Returns:
            float:
                Calculated threshold value.
        """
        mean = errors.mean()
        std = errors.std()

        return mean + k * std
    
    def fix_threshold(self, true_list, errors):
        threshold = self._fixed_threshold(errors)
        pred_list = np.zeros(len(errors))
        above_idx = np.where(np.array(errors) > threshold)
        pred_list[above_idx] = 1
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

    def variance_score(self, true_list, errors, dist_list):
        _, pred_list_recon = self.quantile_score(true_list, errors)
        
        recon_var = dist_list['recon_var_list']
        fore_var = dist_list['fore_var_list']
        recon_var = torch.cat(recon_var)
        fore_var = torch.cat(fore_var)
        recon_var = torch.tensor(recon_var, device='cpu').numpy()
        fore_var = torch.tensor(recon_var, device='cpu').numpy()

        diff_var = abs(recon_var - fore_var)
        pred_list = np.zeros(fore_var.shape[0])
        for i in range(fore_var.shape[1]): # for feature num
            d = diff_var[:,i]
            u_quantile = np.quantile(np.array(d), 0.95) # 0.975
            if d[i] >= u_quantile:
                pred_list[i] = 1

        print("--",sum(pred_list), len(pred_list))
        print("##", sum(pred_list_recon), len(pred_list_recon))

        sum_pred = [pred_list[i] + pred_list_recon[i] for i in range(len(pred_list))]
        sum_pred = [1 if i > 0 else 0 for i in sum_pred]

        pred_list = sum_pred
        return true_list, pred_list
    
    def variance_score_with_weighted_sum(self, true_list, errors, dist_list):
        error_sum = np.sum(errors, axis=1)
        
        recon_var = dist_list['recon_var_list'] # variance of reconstruct part
        fore_var = dist_list['fore_var_list'] # variance of forecast part
        recon_var = torch.cat(recon_var)
        fore_var = torch.cat(fore_var)
        recon_var = torch.tensor(recon_var, device='cpu').numpy() # numpy format
        fore_var = torch.tensor(recon_var, device='cpu').numpy()

        diff_var = abs(recon_var - fore_var)

        new_error = 0.8*error_sum + 0.2*diff_var # weighted sum
        pred_list = np.zeros(new_error.shape[0])
        for i in range(fore_var.shape[1]): # for feature num
            d = new_error[:,i]
            u_quantile = np.quantile(np.array(d), 0.80) # 0.975
            if d[i] >= u_quantile:
                pred_list[i] = 1

        print("--",sum(pred_list), len(pred_list))

        return true_list, pred_list
    
    def variance_score_with_corr(self, args, true_list, errors, dist_list):
        """
        consider correlation of score
        Args:
            true_list:
                Array of true answer
            errors (ndarray):
                Array of errors.
            dist_list  : 
                Array of distances.
        Returns:
            true_list, pred_list
        """
        #---# train data #---#
        train_info = np.load(f'{args.save_path}train_output.npy', allow_pickle=True)
        train_recon = []; train_fore = []; train_var = []; train_x = []
        for i in range(len(train_info)):
            data = train_info[i]
            recon = data['reconstructs']; train_recon.append(recon)
            fore = data['forecasts']; train_fore.append(fore)
            var = data['variances']; train_var.append(var)
            xs = data['x']; train_x.append(xs) 
            # print(recon.shape, fore.shape, var.shape) # [16,50,25] / [16,13,25] / [16, 25]
        train_recon = torch.cat(train_recon)
        train_fore = torch.cat(train_fore)
        train_var = torch.cat(train_var)
        train_x = torch.cat(train_x)
        recon_error = abs(train_recon - train_x[:,:train_recon.shape[1],:]) # difference
        print(train_recon.shape, train_fore.shape, train_var.shape, train_x.shape, "-=======")

        relations = []
        for i in range(len(recon_error)):
            recon = recon_error[i] # [50,25]
            var = train_var[i] # [25]
            relation = recon * var #?????
            relations.append(relation)
        relations = torch.stack(relations)
        train_mm = torch.mean(relations, dim=0) # [50,25]
        
        #---# test data #---#
        recon_var = dist_list['recon_var_list']
        fore_var = dist_list['fore_var_list']
        recon_var = torch.cat(recon_var)
        fore_var = torch.cat(fore_var)
        recon_var = torch.tensor(recon_var, device='cpu').numpy()
        fore_var = torch.tensor(recon_var, device='cpu').numpy()
        
        #---# difference of variances #---#
        diff_var = abs(recon_var - fore_var)

        # new_error = 0.8*error_sum + 0.2*diff_var # weighted sum
        new_error = errors
        print(new_error.shape,"------")

        seq_len = errors.shape[1] # seq_len
        
        values = np.zeros((seq_len, seq_len))
        for i in range(len(new_error)):
            value = new_error[i] @ new_error[i].T
            values =  values + value
            
        '''pred_list = np.zeros(new_error.shape[0])
        for i in range(fore_var.shape[1]): # for feature num
            d = new_error[:,i]
            u_quantile = np.quantile(np.array(d), 0.95) # 0.975
            if d[i] >= u_quantile:
                pred_list[i] = 1'''
        print(values)
        print(values.shape)
        raise
        print("--",sum(pred_list), len(pred_list))

        return true_list, pred_list

    def variance_score_online(self, true_list, errors, dist_list):
        train_output = np.load(f'{self.args.save_path}train_output.npy')
        train_recon = train_output['reconstruct']
        train_variance = train_output['']

        # 바로바로 output
        
        #---# train #---#
        # train output의 distribution??을 이용해서 online detection에 이용
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
