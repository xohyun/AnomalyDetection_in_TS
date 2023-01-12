from torch.utils.data import DataLoader
from DataLoader.data_reader import Dataset_load

def get_dataloader(args):
    dataset = Dataset_load(args)
    
    #---# 우선 설정 #---#
    shuffle_flag = True
    drop_last = True


    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last
    )

    return dataset, data_loader