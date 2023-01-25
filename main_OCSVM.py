from get_args import Args
from utils.utils import fix_random_seed, create_folder

import pandas as pd
import numpy as np
from DataLoader.data_provider import get_dataloader

from sklearn import svm
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

def get_pred_label(model_pred):
    # IsolationForest 모델 출력 (1:정상, -1:불량(사기)) 이므로 (0:정상, 1:불량(사기))로 Label 변환
    model_pred = np.where(model_pred == 1, 0, model_pred)
    model_pred = np.where(model_pred == -1, 1, model_pred)
    return model_pred

def main():
    args_class = Args()
    args = args_class.args
    
    #---# Fix seed #---#
    fix_random_seed(args)

    #---# Save a file #---#
    df = pd.DataFrame(columns = ['dataset', 'f1', 'precision', 'recall']); idx=0
    create_folder('./csvs')
    #---#  DataLoader #---#    
    dl = get_dataloader(args)
    data_loaders = dl.data_loaders

    #---# Data #---#
    if args.dataset == "NAB":
        train_x = data_loaders['train'].dataset.data_x.reshape(-1, args.seq_len)
    else:
        train_x = data_loaders['train'].dataset.data_x_2d
    
    args.mode = "test"
    dl_test = get_dataloader(args)
    data_loaders_test = dl_test.data_loaders
    if args.dataset == "NAB":
        test_x = data_loaders_test['test'].dataset.data_x.reshape(-1, args.seq_len)
        label =  data_loaders_test['test'].dataset.data_y # Label
    else:
        test_x = data_loaders_test['test'].dataset.data_x_2d
        label =  data_loaders_test['test'].dataset.label_2d # Label
        
    if label.ndim != 1:
        label = label.mean(axis=1)
        # label = np.where(label>0.79, 1, 0)
        label = np.where(label>0, 1, 0)


    #---# Take PCA to reduce feature space dimensionality #---#
    pca = PCA(n_components=3, whiten=True)
    pca = pca.fit(train_x)
    print('Explained variance percentage = %0.2f' % sum(pca.explained_variance_ratio_))
    train_x = pca.transform(train_x)
    test_x = pca.transform(test_x)

    #---# IF model #---#
    model = svm.OneClassSVM(gamma=0.001, kernel='rbf', nu=0.01)
    model.fit(train_x)

    pred = model.predict(test_x) # model prediction
    pred = get_pred_label(pred)
    f1 = f1_score(label, pred, average='macro')
    precision = precision_score(label, pred, average='macro')
    recall = recall_score(label, pred, average='macro')

    print(f'F1 Score : {f1}, Precision : {precision}, Recall : {recall}')
    # print(classification_report(val_y, val_pred))
    df.loc[idx] = [args.dataset, f1, precision, recall]
    if args.dataset == 'NAB':
        df.to_csv(f'./csvs/IF_{args.dataset}_{args.choice_data}.csv', header = True, index = False)
    else:
        df.to_csv(f'./csvs/IF_{args.dataset}.csv', header = True, index = False)
    
if __name__ == "__main__":
    main()