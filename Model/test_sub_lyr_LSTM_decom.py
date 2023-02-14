import torch.nn as nn
import torch
import math

from layer import AutoEncoder, LSTM, AttentionLayer


class Decoder(nn.Module):
    def __init__(self, num_features, seq_len):
        super(Decoder, self).__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.n = self.num_features * self.seq_len
        self.hidden1 = int(self.n / 2)
        self.hidden2 = int(self.n / 8)

        self.Decoder = nn.Sequential(
            nn.Linear(711, 50),
            nn.BatchNorm1d(50),
            # nn.LayerNorm(64),
            # nn.LeakyReLU(),
            nn.ReLU(),
            nn.Linear(50, self.n)
        )

    def forward(self, x):
        batch = x.shape[0]
        x = self.Decoder(x)
        x = x.reshape(batch, self.seq_len, -1)
        return x


class Model(nn.Module):
    def __init__(self, configs, num_features, device):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.device = device
        self.config = configs

        self.channels = num_features  # configs.enc_in
        self.conv1d = configs.conv1d

        self.RIN = configs.RIN
        self.combination = configs.combination

        self.batch_size = configs.batch_size

        # ------------------------#
        # ---# Ensemble model #---#
        # ------------------------#
        ### AE encoder ###
        self.AE = AutoEncoder.AutoEncoder(self.channels, self.seq_len).to(
            device=self.device)

        ### LSTM model ###
        self.LSTM = LSTM.LSTM(num_features, self.seq_len,
                              3).to(device=self.device)

        ### Self attention ###
        factor = 5
        dropout = 0.0
        output_attention = False
        d_model = 38  # 512
        n_heads = 1
        embed = 'fixed'
        freq = 'h'

        self.FullAT = AttentionLayer.FullAttention(
            False, factor, attention_dropout=dropout, output_attention=output_attention)
        self.attention = AttentionLayer(
            self.FullAT, d_model, n_heads, mix=False)

        self.decoder = Decoder(
            self.channels, self.seq_len).to(device=self.device)

        ### RIN Parameters ###
        if self.RIN:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, 1))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, 1))

        ### combination ###
        if self.combination:
            self.alpha = nn.Parameter(torch.ones(1, 1, 1))

        ###
        n = self.seq_len * self.channels
        hidden = int(n / 8)
        self.fc = nn.Linear(self.seq_len*self.channels, hidden)  # hidden
        self.fc2 = nn.Linear(hidden*3, 3)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        batch = x.shape[0]

        if self.RIN:
            print('/// RIN ACTIVATED ///\r', end='')
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev
            x = x * self.affine_weight + self.affine_bias

        ae_latent = self.AE(x)
        lstm_feature = self.LSTM(x)

        dropout = 0.0
        d_model = 38  # 512
        embed = 'fixed'
        freq = 'h'

        # enc_embedding = PositionalEmbedding(d_model)
        enc_embedding = AttentionLayer.PositionalEmbedding(
            seq_len=self.seq_len, d_model=d_model, n=20000, device=self.device)
        enc_out = enc_embedding(x)  # [1,50,512]
        enc_out = enc_out + x
        attention_feature, out = self.attention(
            enc_out, enc_out, enc_out, False)
        attention_feature = attention_feature.reshape(batch, -1)
        attention_feature = self.fc(attention_feature)

        feature_concat = torch.concat(
            (ae_latent, lstm_feature, attention_feature), dim=1)
        feature = self.fc2(feature_concat)
        recon_x = self.decoder(feature_concat)

        # if self.combination:
        #     x = (seasonal_output*(self.alpha)) + (trend_output*(1-self.alpha))
        # else:
        #     x = seasonal_output + trend_output
        #     # x = trend_output + c_n

        if self.RIN:
            x = x - self.affine_bias
            x = x / (self.affine_weight + 1e-10)
            x = x * stdev
            x = x + means

        # x = x.reshape(batch, self.seq_len, -1)

        if self.config.mode == 'test' and self.combination:
            print(">>> alpha <<<", self.alpha)
        # if self.config.mode == 'test' and self.combination:
        #     return x, self.alpha
        return feature, x
