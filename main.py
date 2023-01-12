from get_args import Args
from utils import fix_random_seed

import pandas as pd
from DataLoader import data_provider
from Trainer import trainer

def main():
    args_class = Args()
    args = args_class.args
    
    #---# Fix seed #---#
    fix_random_seed()

    #---# Save a file #---#
    df = pd.DataFrame(columns = ['test_subj', 'lr', 'wd', 'epoch', 'acc', 'f1', 'loss']); idx=0

    #---#  DataLoader #---#
    data_loader = data_provider(args)

    #---# Build model #---#
    model =  ModelMaker(args_class).model

    #---# Model train #---#
    trainer = trainer(args, model, data_loader)

    if args.mode == "train":
        f1_v, acc_v, cm_v, loss_v = trainer.train() # fitting
    else:
        f1_v, acc_v, cm_v, loss_v = trainer.evaluation() # fitting

if __name__ == "__main__":
    main()