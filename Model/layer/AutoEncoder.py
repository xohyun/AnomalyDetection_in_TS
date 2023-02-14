'''
2023.02.15
- AutoEncoder' decompose to submodule 
'''

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, num_features, seq_len):
        super(Encoder, self).__init__()
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

    def forward(self, x):
        x = self.Encoder(x)
        return x


class Decoder(nn.Module):
    def __init__(self, num_features, seq_len):
        super(Decoder, self).__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.n = self.num_features * self.seq_len
        self.hidden1 = int(self.n / 2)
        self.hidden2 = int(self.n / 8)

        self.Decoder = nn.Sequential(
            nn.Linear(self.hidden2, self.hidden1),
            nn.BatchNorm1d(self.hidden1),
            # nn.LayerNorm(64),
            # nn.LeakyReLU(),
            nn.ReLU(),
            nn.Linear(self.hidden1, self.n)
        )

    def forward(self, x):
        x = self.Decoder(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, num_features, seq_len):
        super(AutoEncoder, self).__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.Encoder = Encoder(num_features, seq_len)
        self.Decoder = Decoder(num_features, seq_len)

    def forward(self, x):
        batch = x.shape[0]
        x = x.reshape(batch, -1)
        x = self.Encoder(x)
        x = self.Decoder(x)
        x = x.reshape(batch, self.seq_len, -1)
        return x
