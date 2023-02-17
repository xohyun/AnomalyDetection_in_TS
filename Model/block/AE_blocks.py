import torch
import torch.nn as nn
from Model.layer.AE_2 import AutoEncoder
from Model.layer.forecast_lyr import forecast_lyr
from Model.layer.v_inference_lyr import v_inference_lyr

class AE_blocks(nn.Module):
    def __init__(self, seq_len, feature_num, device):
        super().__init__()
        self.seq_len = seq_len
        self.feature_num = feature_num

        self.ae = AutoEncoder(seq_len, feature_num).to(device=device)
        self.fc_forecast = forecast_lyr(
                            seq_len, feature_num).to(device=device)

        self.v_inference = v_inference_lyr(seq_len, feature_num).to(device=device)

    def forward(self, x, original_data):
        batch = x.shape[0]
        latent, reconstruct_x = self.ae(x) # [batch, seq_len * 0.2, feature_num]
        forecast = self.fc_forecast(latent, latent=True)
        forecast = forecast.reshape(batch, -1, self.feature_num)
        reconstruct = reconstruct_x.reshape(x.shape)
        var = self.v_inference(x + original_data)
        return reconstruct, forecast, var