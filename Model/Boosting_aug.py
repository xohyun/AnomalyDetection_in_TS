import numpy as np
import torch
import torch.nn as nn
import math

from layer.AE_blocks import AE_blocks
from layer.attention_blocks import attention_blocks
from layer.rnn_blocks import rnn_blocks


class Forecast(nn.Module):
    def __init__(self, seq_len, num_features):
        super(Forecast, self).__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.n = int(self.num_features * self.seq_len * 0.8)
        self.hidden1 = int(self.n / 2)
        self.hidden2 = int(self.n / 8)

        self.Decoder = nn.Sequential(
            nn.Linear(self.hidden2, self.hidden1),
            nn.BatchNorm1d(self.hidden1),
            # nn.LayerNorm(64),
            # nn.LeakyReLU(),
            nn.ReLU(inplace=False),
            nn.Linear(self.hidden1, self.n)
        )

    def forward(self, x):
        batch = x.shape[0]
        forecast = self.Decoder(x)
        forecast = forecast.reshape(batch, int(self.seq_len*0.2), -1)
        return forecast

###########################################################################
###########################################################################
###########################################################################
###########################################################################


class Model(torch.nn.Module):
    def __init__(self, seq_len, feature_num, stack_num, device):
        super().__init__()
        self.seq_len = seq_len
        self.feature_num = feature_num
        self.stack_num = stack_num
        self.device = device

        self.AE_block = AE_blocks(seq_len, feature_num, device)
        self.attention_block = attention_blocks(seq_len, feature_num, device)
        self.rnn_block = rnn_blocks(seq_len, feature_num, device)

    def forward(self, x):  # x shape [batch, seq_len, feature_num]
        part_idx = int(self.seq_len * 0.8)
        reconstruct_part = x[:, :part_idx, :]
        forecast_part = x[:, part_idx, :]

        forecasts = 0
        reconstructs = 0
        residual = reconstruct_part
        for stack in range(self.stack_num):
            # ---# AutoEncoder block #---#
            reconstruct_ae, forecast_ae = self.AE_block(residual)

            # ---# Attention block #---#
            residual = residual - reconstruct_ae
            reconstruct_attention, forecast_attention = self.attention_block(
                residual)

            # ---# RNN block #---#
            residual = residual - reconstruct_attention
            reconstruct_rnn, forecast_rnn = self.rnn_block(residual)

            # ---# Concat forecast #---#
            forecasts = forecasts + forecast_ae + forecast_attention + forecast_rnn
            reconstructs = reconstructs + reconstruct_ae + \
                reconstruct_attention + reconstruct_rnn

        x_hat = torch.concat((reconstructs, forecasts), dim=1)
        return x_hat
