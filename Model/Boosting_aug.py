import torch
import torch.nn as nn
from Model.block.AE_blocks import AE_blocks
from Model.block.attention_blocks import attention_blocks
from Model.block.rnn_blocks import rnn_blocks


class Forecast(nn.Module):
    def __init__(self, seq_len, num_features, ratio):
        super(Forecast, self).__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.ratio = ratio
        self.n = int(self.seq_len * ratio) * self.num_features
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
        forecast = forecast.reshape(batch, int(self.seq_len*(1-self.ratio)), -1)
        return forecast

###########################################################################
###########################################################################


class Model(torch.nn.Module):
    def __init__(self, seq_len, feature_num, stack_num, device, ratio):
        super().__init__()
        self.seq_len = seq_len
        self.feature_num = feature_num
        self.stack_num = stack_num
        self.device = device

        self.AE_block = AE_blocks(seq_len, feature_num, device, ratio)
        self.attention_block = attention_blocks(seq_len, feature_num, device, ratio)
        self.rnn_block = rnn_blocks(seq_len, feature_num, device, ratio)
        self.ratio = ratio

    def forward(self, x):  # x shape [batch, seq_len, feature_num]
        part_idx = int(self.seq_len * self.ratio) # index of reconstruct part
        reconstruct_part = x[:, :part_idx, :] # recon part
        forecast_part = x[:, part_idx, :] # fore part

        forecasts = 0
        reconstructs = 0
        variances = 0

        residual = reconstruct_part
        for stack in range(self.stack_num):
            # ---# AutoEncoder block #---#
            latent, reconstruct_ae, forecast_ae, var_ae = self.AE_block(
                residual, reconstruct_part)
            residual = residual - reconstruct_ae

            # ---# Attention block #---#
            out, reconstruct_att, forecast_att, var_att = self.attention_block(
                residual, reconstruct_part)
            residual = residual - reconstruct_att

            # ---# RNN block #---#
            hidden_cell, context_vector, reconstruct_rnn, forecast_rnn, var_rnn = self.rnn_block(
                residual, reconstruct_part)
            residual = residual - reconstruct_rnn

            # ---# Concat forecast #---#
            forecasts = forecasts + forecast_ae + forecast_att + forecast_rnn               # forecasts + 
            reconstructs = reconstructs + reconstruct_ae + reconstruct_att + reconstruct_rnn   # reconstructs +
            variances = variances + var_ae + var_att + var_rnn                              # variances +
        # x_hat = torch.concat((reconstructs, forecasts), dim=1)
        return {
            'out': out, # None/ #able to delete
            'latent': latent,
            'hidden_cell': hidden_cell,
            'reconstructs': reconstructs,
            'forecasts': forecasts,
            'variances': variances,
            'context_vector': context_vector
        }
