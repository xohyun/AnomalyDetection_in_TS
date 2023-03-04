import numpy as np
from utils.utils import create_folder
# np.random.binomial(n=1, p=0.5, size=20) # to make sensor data

#---# save folder #---# 
create_folder('./synthethic_dataset/synthetic')

#---# Make train data #---#
total_train_len = 100000
data = np.random.normal(0, 0.08, size=total_train_len)
data = np.array([abs(i) for i in data])
data = data.reshape(-1, 1)
np.save('./synthetic_dataset/synthetic/train.npy', data)

#---# Make test data #---#
total_test_len = 10000
anomaly_ratio = 0.02

data = np.random.normal(0, 0.08, size=total_test_len)
data = np.array([abs(i) for i in data])
data_anomaly = np.random.normal(0.5, 0.1, size=int(total_test_len*anomaly_ratio))
data_anomaly = np.array([abs(i) for i in data_anomaly])

idx_anomaly = np.random.choice(len(data), len(data_anomaly), False)
data[idx_anomaly.astype(int)] = data_anomaly

label = np.array([0 for i in range(len(data))])
label[idx_anomaly.astype(int)] = 1

data = data.reshape(-1, 1)
label = label.reshape(-1, 1)

np.save('./synthetic_dataset/synthetic/test.npy', data)
np.save('./synthetic_dataset/synthetic/label.npy', label)