from torch.utils.data import Dataset
from scipy.signal import butter,filtfilt
import numpy as np
import matplotlib.pyplot as plt
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
        train_data = []; trend_data = []; seasonal_data = []
            
        if self.dataset == 'WADI':
            for f in range(len(train_list)):
                load_file = np.load(os.path.join(data_folder, train_list[f]), allow_pickle=True)
                load_file = np.delete(load_file, (0,1), axis=1)
                train_data.append(load_file)
        else:
            for f in range(len(train_list)):
                load_file = np.load(os.path.join(data_folder, train_list[f]))
                trend, seasonal = self.decomposition(load_file, self.args.ma_window)
                train_data.append(load_file)
                trend_data.append(trend)
                seasonal_data.append(seasonal)

        self.data_x_2d = np.concatenate(train_data)

        self.num_features = train_data[0].shape[1]
        self.data_x = self.cut_data(train_data)
        self.trend_x = self.cut_data(trend_data)
        self.seasonal_x = self.cut_data(seasonal_data)
        
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

        label_data = []; test_data = []; trend_data = []; seasonal_data = []
        if self.dataset == 'WADI':
            for f in range(len(test_list)):
                load_file = np.load(os.path.join(data_folder, test_list[f]), allow_pickle=True)
                load_file = np.delete(load_file, (0,1), axis=1)
                label_file = np.load(os.path.join(data_folder, label_list[f]))
                test_data.append(load_file)
                label_data.append(label_file)
        else:
            for f in range(len(test_list)):
                load_file = np.load(os.path.join(data_folder, test_list[f]))
                trend, seasonal = self.decomposition(load_file, self.args.ma_window)
                test_data.append(load_file)
                trend_data.append(trend)
                seasonal_data.append(seasonal)

                label_file = np.load(os.path.join(data_folder, label_list[f]))
                # if len(label_file.shape) == 1:
                #     label_file = label_file.reshape(-1, 1)
                label_data.append(label_file)
                
        
        self.data_x_2d = np.concatenate(test_data)
        self.label_2d = np.concatenate(label_data)
        self.num_features = test_data[0].shape[1]
        self.data_x = self.cut_data(test_data)
        self.data_y = self.cut_data(label_data)
        self.trend_x = self.cut_data(trend_data)
        self.seasonal_x = self.cut_data(seasonal_data)
        
        print(f"test data shape : {self.data_x.shape} / label data shape : {self.data_y.shape}")
        
    def __getitem__(self, idx):
        if self.mode == 'test':
            return self.data_x[idx], self.data_y[idx],  self.trend_x[idx], self.seasonal_x[idx]
        else:
            return self.data_x[idx], self.trend_x[idx], self.seasonal_x[idx]

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
    
    def decomposition(self, data, window): 
        fs = 50.0       # sample rate, Hz
        cutoff = 2      # desired cutoff frequency of the filter, Hz,      slightly higher than actual 1.2 Hz
        nyq = 0.5 * fs  # Nyquist Frequency
        order = 2      # sin wave can be approx represented as quadratic

        trend = moving_average(data, window)    
        detrend = data - trend
        seasonal = butter_lowpass_filter(detrend, cutoff, fs, order, nyq)

        return trend, seasonal

def moving_average(x, w):
    trend_x_feature = []
    for idx in range(x.shape[1]):
        column_feature = np.convolve(x[:,idx], np.ones(w), 'same') / w # valid
        trend_x_feature.append(column_feature.transpose())

    trend_stack = np.stack(trend_x_feature).reshape(x.shape)
    return trend_stack

def butter_lowpass_filter(data, cutoff, fs, order, nyq):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y