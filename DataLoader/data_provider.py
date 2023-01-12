from torch.utils.data import DataLoader
from DataLoader.data_reader import Dataset_load

def get_dataloader(args):
    dataset = Dataset_load(args)
    data_info = dataset.num_features ### 더 추가할경우 여기서 dictionary로 만들 예정

    #---# 우선 설정 #---#
    shuffle_flag = True
    drop_last = True

    data_loader_train = DataLoader(
        dataset.train_data,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last
    )

    data_loader_valid = DataLoader(
        dataset.valid_data,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last
    )
    
    data_loaders = {"train" : data_loader_train, "valid" : data_loader_valid}
    return data_info, data_loaders

def get_dataloader_test(args):
    dataset = Dataset_load(args)
    data_info = dataset.num_features

    #---# 우선 설정 #---#
    shuffle_flag = False
    drop_last = False

    data_loader_test = DataLoader(
        dataset,
        # batch_size=args.batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last
    )
    
    data_loaders = {"test" : data_loader_test}
    return data_info, data_loaders