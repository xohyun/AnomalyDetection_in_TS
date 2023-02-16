import torch.nn as nn
from layer.AE_2 import AutoEncoder
from layer.forecast_layer import forecast_layer


class AE_blocks(nn.Module):
    def __init__(self, seq_len, feature_num, device):
        super().__init__()
        self.seq_len = seq_len
        self.feature_num = feature_num

        self.ae = AutoEncoder(seq_len, feature_num).to(device=device)
        self.fc_forecast = forecast_layer(
            seq_len, feature_num).to(device=device)

    def forward(self, x):
        batch = x.shape[0]
        latent, reconstruct_x = self.ae(x)
        # [batch, seq_len * 0.2, feature_num]
        forecast = self.fc_forecast(latent, batch)
        forecast = forecast.reshape(batch, -1, self.feature_num)
        reconstruct_x = reconstruct_x.reshape(x.shape)
        return reconstruct_x, forecast
