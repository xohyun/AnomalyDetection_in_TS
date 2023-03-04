from torch.utils.data import Dataset
import numpy as np
import os

class Dataset_load(Dataset):
    def __init__(self, args):
        self.args = args
        self.mode = self.args.mode

        self.seq_len = self.args.seq_len
        self.pred_len = self.args.pred_len
        self.step_len = self.args.step_len
        
        self.data_path = self.args.data_path
        self.dataset = self.args.dataset
        self.dataset_choice = self.args.choice_data

        if self.mode == "train" or self.mode == "all":
            self.__read_train_data__()
        elif self.mode == "test":
            self.__read_test_data__()
        else:
            raise Exception('Check mode!!!')
    
    def __read_train_data__(self):
        data_folder = self.data_path + self.dataset
        data_list = os.listdir(data_folder)
        if self.dataset_choice:
            data_list = [i for i in data_list if any(key in i for key in self.dataset_choice)]
        train_list = [i for i in data_list if 'train' in i]; train_list.sort()
        train_data = []
            
        if self.dataset == 'WADI':
            for f in range(len(train_list)):
                load_file = np.load(os.path.join(data_folder, train_list[f]), allow_pickle=True)
                load_file = np.delete(load_file, (0,1), axis=1)
                train_data.append(load_file) 

        else:
            for f in range(len(train_list)):
                load_file = np.load(os.path.join(data_folder, train_list[f]))
                
                # import matplotlib.pyplot as plt
                # plt.figure(figsize=(15,6))
                # plt.plot(load_file[:,0])
                # plt.savefig(f"train1_{f}.jpg")

                if self.args.SR: # Spectral Residual
                    load_file_sr = spectral_residual_transform(load_file, self.seq_len)
                    saliencymap_thr = np.quantile(load_file_sr, 0.99, axis=0)
                    load_file_thr = np.quantile(load_file, 0.99, axis=0)
 
                    over_thr_idx = np.where(load_file_sr > saliencymap_thr)
                    for idx in range(len(over_thr_idx[0])): # map
                        xx = over_thr_idx[0][idx]
                        yy = over_thr_idx[1][idx]
                        load_file[xx, yy] = load_file_thr[yy]
                    
                    # plt.clf()
                    # plt.figure(figsize=(15,6))
                    # plt.plot(load_file[:,0])
                    # plt.savefig(f"train2_{f}.jpg")
                train_data.append(load_file)

        self.data_x_2d = np.concatenate(train_data)
        self.num_features = train_data[0].shape[1]
        self.data_x = self.cut_data(train_data)
        
        if self.args.valid_setting:
            self.data_x_shuffle = self.data_x.copy()
            np.random.shuffle(self.data_x_shuffle)

            self.train_data = self.data_x_shuffle[:int(len(self.data_x_shuffle)*0.7)]
            self.valid_data = self.data_x_shuffle[int(len(self.data_x_shuffle)*0.7):]

            self.data_x = self.train_data
            print(f"train : {self.train_data.shape} / valid : {self.valid_data.shape}")
        print(f"train : {self.data_x.shape}")

    def __read_test_data__(self):
        data_folder = self.data_path + self.dataset
        data_list = os.listdir(data_folder)
    
        if self.dataset_choice:
            data_list = [i for i in data_list if any(key in i for key in self.dataset_choice)]
        
        label_list = [i for i in data_list if ('label' in i and 'interpret' not in i)]; label_list.sort()
        test_list = [i for i in data_list if 'test' in i]; test_list.sort()

        label_data = []; test_data = []
        if self.dataset == 'WADI':
            for f in range(len(test_list)):
                load_file = np.load(os.path.join(data_folder, test_list[f]), allow_pickle=True)
                load_file = np.delete(load_file, (0,1), axis=1)
                label_file = np.load(os.path.join(data_folder, label_list[f]))
                test_data.append(load_file)
                label_data.append(label_file)
        else:
            for f in range(len(test_list)):
                label_file = np.load(os.path.join(data_folder, label_list[f]))
                # if len(label_file.shape) == 1:
                #     label_file = label_file.reshape(-1, 1)
                label_data.append(label_file)
                test_data.append(np.load(os.path.join(data_folder, test_list[f])))
        
        self.data_x_2d = np.concatenate(test_data)
        self.label_2d = np.concatenate(label_data)
        self.num_features = test_data[0].shape[1]
        self.data_x = self.cut_data(test_data)
        self.data_y = self.cut_data(label_data)
        
        print(f"test data shape : {self.data_x.shape} / label data shape : {self.data_y.shape}")
        
    def __getitem__(self, idx):
        if self.mode == 'test':
            return self.data_x[idx], self.data_y[idx]
        # elif self.args.model == 'LSTMVAE':
        #     return self.data_x[idx], idx
        else:
            return self.data_x[idx]

    def __len__(self):
        return len(self.data_x)
    
    def cut_data(self, list_):
        cut_data = []

        for i in range(len(list_)):
            start_index = 0
            end_index = len(list_[i]) - self.seq_len + 1
            for j in range(start_index, end_index, self.step_len):
                indices = range(j, j+self.seq_len)
                cut_data.append(list_[i][indices])
        return np.array(cut_data)

# https://github.com/microsoft/anomalydetector/blob/master/msanomalydetector/spectral_residual.py
def average_filter(values, n):
    """
    Calculate the sliding window average for the give time series.
    Mathematically, res[i] = sum_{j=i-t+1}^{i} values[j] / t, where t = min(n, i+1)
    :param values: list.
        a list of float numbers
    :param n: int, default 3.
        window size.
    :return res: list.
        a list of value after the average_filter process.
    """

    if n >= len(values):
        n = len(values)

    res = np.cumsum(values, dtype=float, axis=0) # axis plus

    res[n:] = res[n:] - res[:-n]

    res[n:] = res[n:] / n

    for i in range(1, n):
        res[i] /= (i + 1)

    return res

def spectral_residual_transform(values, seq_len):
    """
    This method transform a time series into spectral residual series
    :param values: list.
        a list of float values.
    :return: mag: list.
        a list of float values as the spectral residual values
    """
    EPS = 1e-8
    __mag_window = seq_len
    ##
    trans = np.fft.fft(values)
    mag = np.sqrt(trans.real ** 2 + trans.imag ** 2)
    eps_index = np.where(mag <= EPS)[0]
    mag[eps_index] = EPS
    
    mag_log = np.log(mag)
    mag_log[eps_index] = 0
    spectral = np.exp(mag_log - average_filter(mag_log, n=__mag_window))

    trans.real = trans.real * spectral / mag
    trans.imag = trans.imag * spectral / mag
    trans.real[eps_index] = 0
    trans.imag[eps_index] = 0

    wave_r = np.fft.ifft(trans)
    mag = np.sqrt(wave_r.real ** 2 + wave_r.imag ** 2)
    return mag