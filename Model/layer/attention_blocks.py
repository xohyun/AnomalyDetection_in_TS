import torch
import torch.nn as nn
import math
import numpy as np

from AttentionLayer import AttentionLayer, FullAttention


class attention_blocks(nn.Module):
    def __init__(self, seq_len, feature_num, device):
        super().__init__()
        self.seq_len = seq_len
        self.feature_num = feature_num

        ### Self attention ###
        factor = 5
        dropout = 0.0
        output_attention = False
        d_model = 38  # 512
        n_heads = 1

        # self.blocks = blocks
        self.attention = AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                        d_model, n_heads, mix=False).to(device=device)

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
        attention_feature, out = self.attention(x, x, x, False)
        attention_feature = attention_feature.reshape(batch, -1)

        reconstruct = attention_feature.reshape(x.shape)
        forecast = self.fc_forecast(attention_feature)
        forecast = forecast.reshape(batch, -1, self.feature_num)
        return reconstruct, forecast
        # residuals = x.flip(dims=(1,))
        # input_mask = input_mask.flip(dims=(1,))
        # forecast = x[:, -1:]
        # for i, block in enumerate(self.blocks):
        #     backcast, block_forecast = block(residuals)
        #     residuals = (residuals - backcast) * input_mask
        #     forecast = forecast + block_forecast
        # return forecast
