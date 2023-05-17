import torch.nn as nn
import torch

class v_inference_lyr(nn.Module):
    '''
    For variance prediction
    '''
    def __init__(self, seq_len, feature_num, ratio):
        super().__init__()
        self.seq_len = seq_len
        self.feature_num = feature_num

        self.n = int(self.seq_len * ratio) * self.feature_num
        self.hidden1 = int(self.n / 4)
        self.hidden2 = int(self.n / 8)

        self.fc = nn.Sequential(
            nn.Linear(self.n, self.hidden1),
            nn.BatchNorm1d(self.hidden1),
            nn.ReLU(),
            nn.Linear(self.hidden1, self.hidden2),
            nn.BatchNorm1d(self.hidden2),
            nn.ReLU(),
            nn.Linear(self.hidden2, self.feature_num)
        )
    
    def forward(self, x):
        batch = x.shape[0]
        x = x.reshape(batch, -1)
        return self.fc(x)