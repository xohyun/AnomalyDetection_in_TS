import numpy as np
import torch
import torch.nn as nn

# https://github.com/ServiceNow/N-BEATS/blob/master/models/nbeats.py
class ModelBlock(nn.Module):
    def __init__(self, input_size, theta_size: int, layers: int, layer_size: int):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(in_features=input_size, out_features=layer_size)] +
                                      [torch.nn.Linear(in_features=layer_size, out_features=layer_size)
                                       for _ in range(layers - 1)])
        self.basis_parameters = torch.nn.Linear(in_features=layer_size, out_features=theta_size)
    
    def forward(self, x):
        block_input = x
        for layer in self.layers:
            block_input = torch.relu(layer(block_input))
        basis_parameters = self.basis_parameters(block_input)
        return self.basis_function(basis_parameters)

class AE_blocks(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks

    def forward(self, x, input_mask):
        residuals = x.flip(dims=(1,))
        input_mask = input_mask.flip(dims=(1,))
        forecast = x[:, -1:]
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(residuals)
            residuals = (residuals - backcast) * input_mask
            forecast = forecast + block_forecast
        return forecast

class attention_blocks(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks

    def forward(self, x, input_mask):
        residuals = x.flip(dims=(1,))
        input_mask = input_mask.flip(dims=(1,))
        forecast = x[:, -1:]
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(residuals)
            residuals = (residuals - backcast) * input_mask
            forecast = forecast + block_forecast
        return forecast


######### delete
class AutoEncoder(nn.Module):
    def __init__(self, num_features, seq_len):
        super(AutoEncoder, self).__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.n = self.num_features * self.seq_len
        self.hidden1 = int(self.n / 2)
        self.hidden2 = int(self.n / 8)

        self.Encoder = nn.Sequential(
            nn.Linear(self.n, self.hidden1),
            nn.BatchNorm1d(self.hidden1),
            # nn.LeakyReLU(),
            nn.ReLU(),
            nn.Linear(self.hidden1, self.hidden2),
            nn.BatchNorm1d(self.hidden2),
            # nn.LeakyReLU(),
            nn.ReLU(),
        )
        self.Decoder = nn.Sequential(
            nn.Linear(self.hidden2, self.hidden1),
            nn.BatchNorm1d(self.hidden1),
            # nn.LayerNorm(64),
            # nn.LeakyReLU(),
            nn.ReLU(),
            nn.Linear(self.hidden1, self.n)
        )
    def forward(self, x):
        batch = x.shape[0]
        x = x.reshape(batch, -1)
        latent = self.Encoder(x)
        x_reconstruct = self.Decoder(latent)
        x_reconstruct = x_reconstruct.reshape(batch, self.seq_len, -1)
        return latent, x_reconstruct

class Model(torch.nn.Module):
    AE_block = AE_blocks()
    attention_block = attention_blocks()


    # for i, block in enumerate(self.blocks):
    #         backcast, block_forecast = block(residuals)
    #         residuals = (residuals - backcast) * input_mask
    #         forecast = forecast + block_forecast
    return NBeats(t.nn.ModuleList(
        [trend_block for _ in range(trend_blocks)] + [seasonality_block for _ in range(seasonality_blocks)]))