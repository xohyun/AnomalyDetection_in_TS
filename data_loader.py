import torch
# from torch.utils.data import Dataset, Dataloader ##이거왜??
from torch.utils.data import Dataset
import numpy as np
import os

class Dataset_load(Dataset):
    def __init__(self, args):
        self.seq_len = args.seq_len
        # self.label_lan
        self.pred_len = args.pred_len
        
        self.data_path = args.data_path
        self.dataset = args.dataset
        # 여기서부터 확인
        # type_map = {'train' : 0, 'val' : 1, 'test' : 2}
        # self.set_type = type_map[args.mode]

        # self.features = features
        # self.target = target
        # self.scale = scale
        # self.timeenc = timeenc
        # self.freq = freq

        # self.root_path = root_path
        
        self.__read_data__()

    
    def __read_data__(self):
        data_folder = self.data_path + self.dataset

        if self.dataset == 'NAB':
            data_list = os.listdir(data_folder)
            # sorted(data_list)
            # print(data_list)

            data = np.load(os.path.join(data_folder, data_list[1]))
            
            data = np.load(os.path.join(data_folder, "ambient_temperature_system_failure_labels.npy"))
            print(data.shape)
            data = np.load(os.path.join(data_folder, "ambient_temperature_system_failure_train.npy"))
            print(data.shape)
            data = np.load(os.path.join(data_folder, "ambient_temperature_system_failure_test.npy"))
            print(data.shape)

    def __getitem__(self, index):
        seq_begin = index
        seq_end = seq_begin + self.seq_len

        # r_begin = seq_end - self.label_len
        # r_end = r_begin + self.label_len + self.pred_len

        # seq_x = self.data_x[s_begin:s_end]
        # seq_y = self.data_y[r_begin:r_end]
        # seq_x_mark = self.data_stamp[s_begin:s_end]
        # seq_y_mark = self.data_stamp[r_begin:r_end]

        # return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

if __name__ == "__main__":
    from get_args import Args
    args_class = Args()
    args = args_class.args
    
    dl = Dataset_load(args)
