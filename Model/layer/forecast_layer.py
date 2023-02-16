import torch.nn as nn


class forecast_layer(nn.Module):
    def __init__(self, seq_len, feature_num):
        super().__init__()
        self.seq_len = seq_len
        self.feature_num = feature_num

        self.n = int(self.feature_num * self.seq_len * 0.8)
        self.hidden1 = int(self.n / 2)
        self.hidden2 = int(self.n / 8)
        self.forecast = int(self.feature_num * self.seq_len * 0.2)

        self.fc_forecast = nn.Sequential(
            nn.Linear(self.hidden2, self.hidden1),
            nn.BatchNorm1d(self.hidden1),
            # nn.LayerNorm(64),
            # nn.LeakyReLU(),
            nn.ReLU(inplace=False),
            nn.Linear(self.hidden1, self.forecast)
        )

    def forward(self, x, batch):
        forecast = self.fc_forecast(x)
        # forecast = forecast.reshape(batch, -1, self.feature_num)
        return forecast
