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
