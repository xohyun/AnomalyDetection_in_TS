import torch
import torch.nn as nn

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x): # [1, batch, seq_len*feature_num]
        # padding on the both ends of time series

        if self.kernel_size%2 != 0: ##  even must be modify
          front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
          end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)

        else:
          front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
          end = x[:, -1:, :].repeat(1, (self.kernel_size ) // 2, 1)

        # front, end shape [1, 2, seq_len*feature_num]

        x = torch.cat([front, x, end], dim=1) # [1, 20, 380]
        x = self.avg(x.permute(0, 2, 1)) # [1, 380, 16]
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        moving_mean=[]
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean=torch.cat(moving_mean,dim=-1)
        moving_mean = torch.sum(moving_mean*nn.Softmax(-1)(self.layer(x.unsqueeze(-1))),dim=-1)
        res = x - moving_mean
        return res, moving_mean 

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
        x = self.Encoder(x)
        x = self.Decoder(x)
        x = x.reshape(batch, self.seq_len, -1)
        return x

class Model(nn.Module):
    def __init__(self, configs, num_features, device):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
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

        ### AE model ###
        self.AE_trend = AutoEncoder(self.channels, self.seq_len).to(device=self.device)
        self.AE_seasonal = AutoEncoder(self.channels, self.seq_len).to(device=self.device)

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
        seasonal_output = self.AE_seasonal(seasonal_init)
        
        if self.combination:
            x = (seasonal_output*(self.alpha)) + (trend_output*(1-self.alpha))     
        else:
            x = seasonal_output + trend_output
        
        if self.RIN:
            x = x - self.affine_bias
            x = x / (self.affine_weight + 1e-10)
            x = x * stdev
            x = x + means

        x = x.reshape(batch, self.seq_len, -1)

        # if self.combination:
        #     return x, self.alpha
        if self.config.mode == 'test' and self.combination:
            print(">>> alpha <<<", self.alpha)
        return x 