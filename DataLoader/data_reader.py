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


        # 여기서부터 확인
        # type_map = {'train' : 0, 'val' : 1, 'test' : 2}
        # self.set_type = type_map[args.mode]

        # self.features = args.features
        # self.target = target
        # self.scale = scale
        # self.timeenc = timeenc
        # self.freq = freq

        # self.root_path = root_path

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

        for f in range(len(train_list)):
            train_data.append(np.load(os.path.join(data_folder, train_list[f])))
            
        # if self.dataset == 'WADI':
        #     for f in range(len(train_list)):
        #         load_file = np.load(os.path.join(data_folder, train_list[f]), allow_pickle=True)
        #         print(type(load_file))
        #         train_data.append(load_file)

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
        
        label_list = [i for i in data_list if 'label' in i]; label_list.sort()
        test_list = [i for i in data_list if 'test' in i]; test_list.sort()

        label_data = []; test_data = []
        for f in range(len(test_list)):
            label_data.append(np.load(os.path.join(data_folder, label_list[f])))
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

if __name__ == "__main__":
    
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    
    from get_args import Args
    
    args_class = Args()
    args = args_class.args
    
    dl = Dataset_load(args)
