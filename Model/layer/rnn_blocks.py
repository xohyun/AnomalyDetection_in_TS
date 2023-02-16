import torch.nn as nn
from layer.LSTM import LSTM
from layer.forecast_layer import forecast_layer


class rnn_blocks(nn.Module):
    def __init__(self, seq_len, feature_num, device):
        super().__init__()
        self.seq_len = seq_len
        self.feature_num = feature_num

        self.LSTM = LSTM(feature_num, int(seq_len*0.8), 3).to(device=device)
        self.fc_forecast = forecast_layer(
            seq_len, feature_num).to(device=device)

    def forward(self, x):
        batch = x.shape[0]
        context_vector = self.LSTM(x)  # [batch, seq_len*feature_num]

        reconstruct = context_vector.reshape(x.shape)
        forecast = self.fc_forecast(context_vector, batch)
        forecast = forecast.reshape(batch, -1, self.feature_num)
        return reconstruct, forecast
