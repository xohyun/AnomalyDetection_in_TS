import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def drawing_heatmap(args, train_mm):
    '''
    Plotting heatmap
    args : argument
    train_mm : weight of train datas
    '''
    df = pd.DataFrame(train_mm)
    plt.pcolor(df)
    plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
    plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
    plt.title('Heatmap', fontsize=20)
    # plt.xlabel('Year', fontsize=14)
    # plt.ylabel('Month', fontsize=14)
    plt.colorbar()
    plt.savefig(f"{args.csv_path}{args.model}_{args.dataset}_{args.choice_data}_lr{args.lr}_wd{args.wd}_seq{args.seq_len}_step{args.step_len}_batch{args.batch_size}_epoch{args.epoch}_score{args.score}_calc{args.calc}---------------------normal.png")
