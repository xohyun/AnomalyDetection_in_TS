import wandb
from get_args import Args
from utils.utils import fix_random_seed, create_folder
import pandas as pd
from DataLoader.data_provider import get_dataloader
# from Trainer.trainer import TrainMaker
# from Trainer.trainer_AE import TrainMaker
from Model.model_maker import ModelMaker
import importlib

def load_module_func(module_name):
    mod = importlib.import_module(module_name)
    return mod
    
def main():
    args_class = Args()
    args = args_class.args
    
    create_folder(args.fig_path); create_folder(args.csv_path)

    #---# import #---#
    mod = load_module_func(f"Trainer.trainer_{args.model}")

    #---# Wandb #---#
    # wandb.init(project='AD-project', name=f'{args.model}_{args.lr}_{args.wd}_{args.seq_len}_{args.step_len}')
    # wandb.config.update(args)

    #---# Fix seed #---#
    fix_random_seed(args)

    #---# Save a file #---#
    df = pd.DataFrame(columns = ['dataset', 'f1', 'precision', 'recall', 'seq_len', 'step_len', 'lr', 'wd', 'batch', 'epoch']); idx=0

    #---#  DataLoader #---#    
    dl = get_dataloader(args)
    data_info, data_loaders = dl.data_info, dl.data_loaders

    #---# Build model #---#
    model =  ModelMaker(args, data_info).model

    #---# Model train #---#
    trainer = mod.TrainMaker(args, model, data_loaders, data_info)

    if args.mode == "train":
        f1 = trainer.train() # fitting
        
    elif args.mode == "test":
        f1, precision, recall = trainer.evaluation(data_loaders['test'])
        print("end", f1)

    elif args.mode == "all":
        args.mode = "train"
        f1 = trainer.train() # fitting

        args.mode = "test"
        dl_test = get_dataloader(args)
        data_info, data_loaders = dl_test.data_info, dl_test.data_loaders

        model = ModelMaker(args, data_info).model
        trainer = mod.TrainMaker(args, model, data_loaders, data_info)
        f1, precision, recall = trainer.evaluation(data_loaders['test'])

        df.loc[idx] = [args.dataset, f1, precision, recall, args.seq_len, args.step_len, args.lr, args.wd, args.batch_size, args.epoch]
        if args.dataset == 'NAB':
            df.to_csv(f'{args.csv_path}{args.model}_{args.dataset}_{args.choice_data}_lr{args.lr}_wd{args.wd}_seq{args.seq_len}_step{args.step_len}_batch{args.batch_size}_epoch{args.epoch}.csv', header = True, index = False)
        else:
            df.to_csv(f'{args.csv_path}{args.model}_{args.dataset}_lr{args.lr}_wd{args.wd}_seq{args.seq_len}_step{args.step_len}_batch{args.batch_size}_epoch{args.epoch}.csv', 
                        header = True, index = False)

if __name__ == "__main__":
    main()