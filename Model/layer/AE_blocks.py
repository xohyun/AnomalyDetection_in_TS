import torch
import torch.nn as nn
import math
import numpy as np

from AE_2 import AutoEncoder


class AE_blocks(nn.Module):
    def __init__(self, seq_len, feature_num, device):
        super().__init__()
        self.seq_len = seq_len
        self.feature_num = feature_num

        self.ae = AutoEncoder(seq_len, feature_num).to(device=device)

        # 나중에 decoder가져와서 붙여도될듯
        self.n = int(self.feature_num * self.seq_len * 0.8)
        self.hidden1 = int(self.n / 2)
        self.hidden2 = int(self.n / 8)
        self.forecast = int(self.feature_num * self.seq_len * 0.2)

        self.fc_forecast = nn.Sequential(
            nn.Linear(self.hidden2, self.hidden1),
            nn.BatchNorm1d(self.hidden1),
            # nn.LayerNorm(64),
            # nn.LeakyReLU(),
            nn.ReLU(inplace=False),
            nn.Linear(self.hidden1, self.forecast)
        )

    def forward(self, x):
        batch = x.shape[0]
        latent, reconstruct_x = self.ae(x)
        # [batch, seq_len * 0.2, feature_num]
        forecast = self.fc_forecast(latent)
        forecast = forecast.reshape(batch, -1, self.feature_num)

        reconstruct_x = reconstruct_x.reshape(x.shape)
        return reconstruct_x, forecast
