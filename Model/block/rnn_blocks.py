import torch.nn as nn
from layer.LSTM import LSTM
from Model.layer.forecast_lyr import forecast_layer
from Model.layer.v_inference_lyr import v_inference_lyr

class rnn_blocks(nn.Module):
    def __init__(self, seq_len, feature_num, device):
        super().__init__()
        self.seq_len = seq_len
        self.feature_num = feature_num

        self.LSTM = LSTM(feature_num, int(seq_len*0.8), 3).to(device=device)
        self.fc_forecast = forecast_layer(
            seq_len, feature_num).to(device=device)

        self.v_inference = v_inference_lyr(seq_len, feature_num).to(device=device)

    def forward(self, x, original_data):
        batch = x.shape[0]
        context_vector = self.LSTM(x)  # [batch, seq_len*feature_num]

        reconstruct = context_vector.reshape(x.shape)
        forecast = self.fc_forecast(context_vector, batch)
        forecast = forecast.reshape(batch, -1, self.feature_num)
        var = self.v_inference(x + original_data)
        return reconstruct, forecast, var
