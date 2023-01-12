from get_args import Args
from utils import fix_random_seed

import pandas as pd
from DataLoader.data_provider import get_dataloader
from Trainer.trainer import TrainMaker
from Model.model_maker import ModelMaker

def main():
    args_class = Args()
    args = args_class.args
    
    #---# Fix seed #---#
    fix_random_seed(args)

    #---# Save a file #---#
    df = pd.DataFrame(columns = ['test_subj', 'lr', 'wd', 'epoch', 'acc', 'f1', 'loss']); idx=0

    #---#  DataLoader #---#
    data_info, data_loader = get_dataloader(args)

    #---# Build model #---#
    model =  ModelMaker(args, data_info).model

    #---# Model train #---#
    trainer = TrainMaker(args, model, data_loader, data_info)

    if args.mode == "train":
        f1_v, acc_v, cm_v, loss_v = trainer.train() # fitting
    else:
        f1_v, acc_v, cm_v, loss_v = trainer.evaluation() # fitting

if __name__ == "__main__":
    main()