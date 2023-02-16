import torch
import torch.nn as nn
import math
import numpy as np

from LSTM import LSTM


class rnn_blocks(nn.Module):
    def __init__(self, seq_len, feature_num, device):
        super().__init__()
        self.seq_len = seq_len
        self.feature_num = feature_num

        self.LSTM = LSTM(feature_num, int(seq_len*0.8), 3).to(device=device)

        # 나중에 decoder가져와서 붙여도될듯
        self.n = int(self.feature_num * self.seq_len * 0.8)
        self.hidden1 = int(self.n / 2)
        self.hidden2 = int(self.n / 8)
        self.forecast = int(self.feature_num * self.seq_len * 0.2)

        self.fc_forecast = nn.Sequential(
            nn.Linear(self.n, self.hidden1),
            nn.BatchNorm1d(self.hidden1),
            # nn.LayerNorm(64),
            # nn.LeakyReLU(),
            nn.ReLU(inplace=False),
            nn.Linear(self.hidden1, self.forecast)
        )

    def forward(self, x):
        batch = x.shape[0]
        context_vector = self.LSTM(x)  # [batch, seq_len*feature_num]

        reconstruct = context_vector.reshape(x.shape)
        forecast = self.fc_forecast(context_vector)
        forecast = forecast.reshape(batch, -1, self.feature_num)
        return reconstruct, forecast
