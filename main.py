import wandb
from get_args import Args
from utils.utils import fix_random_seed, create_folder
import pandas as pd
from DataLoader.data_provider import get_dataloader
from Model.model_maker import ModelMaker
from utils.utils import load_module_func, make_csv

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
    df = pd.DataFrame(columns = ['dataset', 'choice_data', 'f1', 'precision', 'recall', 
                                'seq_len', 'step_len', 'lr', 'wd', 
                                'batch', 'epoch', 'score', 'calc']); idx=0
    if args.model == 'Boosting_aug':
        df = pd.DataFrame(columns = ['dataset', 'choice_data', 'f1', 'precision', 'recall',
                                'mae', 'rmse', 'mape', 
                                'seq_len', 'step_len', 'lr', 'wd', 
                                'batch', 'epoch', 'score', 'calc']); idx=0
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
        # for train and test
        args.mode = "train"
        f1 = trainer.train() # fitting

        args.mode = "test"
        dl_test = get_dataloader(args)
        data_info, data_loaders = dl_test.data_info, dl_test.data_loaders

        model = ModelMaker(args, data_info).model
        trainer = mod.TrainMaker(args, model, data_loaders, data_info)
        f1, precision, recall, mae, rmse, mape = trainer.evaluation(data_loaders['test'])

    if args.mode != "train":
        #---# To make csv file #---#
        if args.model == 'Boosting_aug': # for save forecast performance
            df.loc[idx] = [args.dataset, args.choice_data, f1, precision, recall,
                        mae, rmse, mape, 
                        args.seq_len, args.step_len, args.lr, args.wd, 
                        args.batch_size, args.epoch, args.score, args.calc]
        else:
            df.loc[idx] = [args.dataset, args.choice_data, f1, precision, recall,
                        args.seq_len, args.step_len, args.lr, args.wd, 
                        args.batch_size, args.epoch, args.score, args.calc]
        make_csv(df, args)

if __name__ == "__main__":
    main()