from get_args import Args
from utils.utils import fix_random_seed

import pandas as pd
from DataLoader.data_provider import get_dataloader
# from Trainer.trainer import TrainMaker
from Trainer.trainer_AE import TrainMaker
from Model.model_maker import ModelMaker

def main():
    args_class = Args()
    args = args_class.args
    
    #---# Fix seed #---#
    fix_random_seed(args)

    #---# Save a file #---#
    df = pd.DataFrame(columns = ['test_subj', 'lr', 'wd', 'epoch', 'acc', 'f1', 'loss']); idx=0

    #---#  DataLoader #---#    
    dl = get_dataloader(args)
    data_info, data_loaders = dl.data_info, dl.data_loaders

    #---# Build model #---#
    model =  ModelMaker(args, data_info).model

    #---# Model train #---#
    trainer = TrainMaker(args, model, data_loaders, data_info)

    if args.mode == "train":
        f1 = trainer.train() # fitting
        
    elif args.mode == "test":
        f1_v = trainer.evaluation(data_loaders['test'])
        print("end", f1_v)

    elif args.mode == "all":
        args.mode = "train"
        f1 = trainer.train() # fitting

        args.mode = "test"
        dl_test = get_dataloader(args)
        data_info, data_loaders = dl_test.data_info, dl_test.data_loaders

        model = ModelMaker(args, data_info).model
        trainer = TrainMaker(args, model, data_loaders, data_info)
        f1_v = trainer.evaluation(data_loaders['test'])

if __name__ == "__main__":
    main()