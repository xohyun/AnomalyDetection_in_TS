import torch.nn as nn
import torch
from Model.AE_decom import *

class LSTM(nn.Module):
    def __init__(self, n_features, n_seq, num_layers, hidden_dim=256):
        super(LSTM, self).__init__()
        self.n_features = n_features
        self.n_seq = n_seq
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
    
        self.lstm_layer = nn.LSTM(
            input_size = self.n_features,
            hidden_size = self.hidden_dim,
            num_layers = self.num_layers,
            batch_first = True,
            dropout = 0.25,
        )
        
        # self.act_fn = nn.ReLU()
        self.act_fn = nn.Tanh()
        middle_dim = int((self.hidden_dim)/2)
        middle_dim2 = int((self.hidden_dim)/4)
        self.fc1 = nn.Linear(self.hidden_dim, middle_dim)
        self.fc2 = nn.Linear(middle_dim, self.n_features*self.n_seq)
        self.fc3 = nn.Linear(self.hidden_dim, self.n_features*self.n_seq)
        # self.fc4 = nn.Linear(self.hidden_dim*2, self.middle_dim)
        # self.fc4 = nn.Linear(self.middle_dim, self.n_features*self.n_seq)
        # self.fc = nn.Linear(self.hidden_dim, self.pred_len)
        
    def forward(self, x):
        x, (h_n, c_n) = self.lstm_layer(x) 
        # x shape [batch_size, seq, hidden_size]
        # c_n shape [layer, batch, hidden]
        
        x = x[:, -1, :] # shape [batch, 1, hidden_size]
        x = self.act_fn(x)

        c_n = c_n[-1, :, :].reshape(x.shape) # final layer
        
        concats = torch.concat((x, c_n), axis=1) # [batch, hidden*2] 
        concats = self.fc4(concats)
        concats = concats.reshape(x.shape[0],  self.n_seq, self.n_features)
        raise
        # x = self.fc(x)
        x = self.fc1(x) # [batch, 32]
        x = self.act_fn(x)
        x = self.fc2(x) # [batch, feature*seq]
        x = x.reshape(-1, self.n_seq, self.n_features) # [batch, seq, feature]

        # print(c_n.shape) # [layer, batch, hidden]
        
        c_n = c_n.reshape(x.shape[0], -1) # [batch, hidden]
        c_n = self.fc3(c_n) # [batch, feature*seq]
        c_n = c_n.reshape(x.shape[0],  self.n_seq, self.n_features)
        raise
        return x, h_n, c_n

class Model(nn.Module):
    def __init__(self, configs, num_features, device):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.device = device
        self.config = configs

        # Decomp
        self.kernel_size = configs.moving_avg_list
        if isinstance(self.kernel_size, list):
            self.decomposition = series_decomp_multi(self.kernel_size)
        else:
            self.decomposition = series_decomp(self.kernel_size)

        self.channels = num_features # configs.enc_in
        self.conv1d = configs.conv1d

        self.RIN = configs.RIN
        self.combination =  configs.combination

        self.batch_size = configs.batch_size

        ### LSTM model ###
        self.AE_trend = AutoEncoder(self.channels, self.seq_len).to(device=self.device)
        self.LSTM_seasonal = LSTM(num_features, self.seq_len, 3)
        
        ### RIN Parameters ###
        if self.RIN:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, 1))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, 1))

        ### combination ###
        if self.combination:
          self.alpha = nn.Parameter(torch.ones(1,1,1))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        batch = x.shape[0]

        if self.RIN:
            print('/// RIN ACTIVATED ///\r', end='')
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev
            x = x * self.affine_weight + self.affine_bias
        
        seasonal_init, trend_init = self.decomposition(x)
        
        seasonal_init = seasonal_init.squeeze()
        trend_init = trend_init.squeeze()

        trend_output = self.AE_trend(trend_init)
        seasonal_output, h_n, c_n = self.LSTM_seasonal(seasonal_init)

        seasonal_output = seasonal_output + c_n
        if self.combination:
            x = (seasonal_output*(self.alpha)) + (trend_output*(1-self.alpha))     
        else:
            x = seasonal_output + trend_output
            # x = trend_output + c_n
        
        if self.RIN:
            x = x - self.affine_bias
            x = x / (self.affine_weight + 1e-10)
            x = x * stdev
            x = x + means

        x = x.reshape(batch, self.seq_len, -1)

        if self.config.mode == 'test' and self.combination:
            print(">>> alpha <<<", self.alpha)
        # if self.config.mode == 'test' and self.combination:
        #     return x, self.alpha
        return x 