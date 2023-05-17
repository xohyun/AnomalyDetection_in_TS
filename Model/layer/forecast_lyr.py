import torch.nn as nn

class forecast_lyr(nn.Module):
    '''
    For forecast layer
    '''
    def __init__(self, seq_len, feature_num, ratio):
        super().__init__()
        self.seq_len = seq_len
        self.feature_num = feature_num

        self.n = int(self.seq_len * ratio) * self.feature_num
        self.hidden1 = int(self.n / 2)
        self.hidden2 = int(self.n / 8)
        # self.forecast = int(self.feature_num * self.seq_len * 0.2)
        self.forecast = self.seq_len * self.feature_num - self.n

        self.fc_forecast_latent = nn.Sequential(
            nn.Linear(self.hidden2, self.hidden1),
            nn.BatchNorm1d(self.hidden1),
            # nn.LayerNorm(64),
            # nn.LeakyReLU(),
            nn.ReLU(inplace=False),
            nn.Linear(self.hidden1, self.forecast)
        )

        self.fc_forecast = nn.Sequential(
            nn.Linear(self.n, self.hidden1),
            nn.BatchNorm1d(self.hidden1),
            # nn.LayerNorm(64),
            # nn.LeakyReLU(),
            nn.ReLU(inplace=False),
            nn.Linear(self.hidden1, self.forecast)
        )

    def forward(self, x, latent=False):
        if latent:
            forecast = self.fc_forecast_latent(x)
        else:
            forecast = self.fc_forecast(x)
        # forecast = forecast.reshape(batch, -1, self.feature_num)
        return forecast
