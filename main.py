from get_args import Args
from utils import fix_random_seed

import pandas as pd
from DataLoader.data_provider import get_dataloader, get_dataloader_test
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
    if args.mode == "train":
        data_info, data_loaders = get_dataloader(args)
    else:
        data_info, data_loaders = get_dataloader_test(args)

    #---# Build model #---#
    model =  ModelMaker(args, data_info).model

    #---# Model train #---#
    trainer = TrainMaker(args, model, data_loaders, data_info)

    if args.mode == "train":
        f1 = trainer.train() # fitting
        args.mode = "test"
        data_info, data_loaders = get_dataloader_test(args)
        
        model = ModelMaker(args, data_info, pretrained=True).model
        trainer = TrainMaker(args, model, data_loaders, data_info)
        f1_v = trainer.evaluation(data_loaders['test'])
    else:
        # f1_v, acc_v, cm_v, loss_v = trainer.evaluation() # fitting
        model = ModelMaker(args, data_info, pretrained=True).model
        trainer = TrainMaker(args, model, data_loaders, data_info)
        f1_v = trainer.evaluation(data_loaders['test'])

if __name__ == "__main__":
    main()