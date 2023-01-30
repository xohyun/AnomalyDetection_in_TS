from torch.utils.data import DataLoader
# from DataLoader.data_reader import Dataset_load
from DataLoader.data_reader_decom import Dataset_load

class get_dataloader():
    def __init__(self, args):
        self.args = args
        self.dataset = Dataset_load(self.args)
        data_info = {"num_features" : self.dataset.num_features, "seq_len" : self.dataset.seq_len}

        if self.args.mode == "train" or self.args.mode == "all":
            data_loaders = self.__get_dataloader_train(self.args)
        elif self.args.mode == "test":
            data_loaders = self.__get_dataloader_test(self.args)
        
        self.data_info = data_info
        self.data_loaders = data_loaders

    def __call__(self, args):
        return self.data_info, self.data_loaders

    def __get_dataloader_train(self, args):
        shuffle_flag = True
        drop_last = True

        data_loader_train = DataLoader(
            self.dataset,
            batch_size=args.batch_size,
            shuffle=shuffle_flag,
            drop_last=drop_last
        )
        data_loaders = {"train" : data_loader_train}

        if self.args.valid_setting:
            data_loader_valid = DataLoader(
                self.dataset.valid_data,
                batch_size=args.batch_size,
                shuffle=shuffle_flag,
                drop_last=drop_last
            )
            data_loaders["valid"] = data_loader_valid
        return data_loaders

    def __get_dataloader_test(self, args):
        #---# 우선 설정 #---#
        shuffle_flag = False
        drop_last = False

        data_loader_test = DataLoader(
            self.dataset,
            batch_size=args.batch_size,
            shuffle=shuffle_flag,
            drop_last=drop_last
        )
        
        data_loaders = {"test" : data_loader_test}
        return data_loaders