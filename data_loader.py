import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from utils import *

dataset = 'MSL'
dataset_folder = './original_data/SMAP_MSL'
output_folder = os.path.join('./data', dataset)
create_folder(output_folder)

file = os.path.join(dataset_folder, 'labeled_anomalies.csv')
values = pd.read_csv(file)

values = values[values['spacecraft'] == dataset]
filenames = values['chan_id'].values.tolist()

for fn in filenames:
    train = np.load(f'{dataset_folder}/train/{fn}.npy')
    test = np.load(f'{dataset_folder}/test/{fn}.npy')
    # train, min_a, max_a = normalize3(train)
    # test, _, _ = normalize3(test, min_a, max_a)

    #---# MinMaxScaler #---#
    scaler = MinMaxScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    
    #---# save train.npy and test.npy #---#
    np.save(f'{output_folder}/{fn}_train.npy', train)
    np.save(f'{output_folder}/{fn}_test.npy', test)
    labels = np.zeros(test.shape)
    indices = values[values['chan_id'] == fn]['anomaly_sequences'].values[0]
    indices = indices.replace(']', '').replace('[', '').split(', ')
    indices = [int(i) for i in indices]
    for i in range(0, len(indices), 2):
        labels[indices[i]:indices[i+1], :] = 1
    np.save(f'{output_folder}/{fn}_labels.npy', labels)