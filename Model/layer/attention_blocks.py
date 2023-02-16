import torch.nn as nn
from AttentionLayer import AttentionLayer, FullAttention
from layer.forecast_layer import forecast_layer


class attention_blocks(nn.Module):
    def __init__(self, seq_len, feature_num, device):
        super().__init__()
        self.seq_len = seq_len
        self.feature_num = feature_num

        ### Self attention ###
        factor = 5
        dropout = 0.0
        output_attention = False
        d_model = 38  # 512
        n_heads = 1

        # self.blocks = blocks
        self.attention = AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                        d_model, n_heads, mix=False).to(device=device)

        self.fc_forecast = forecast_layer(
            seq_len, feature_num).to(device=device)

    def forward(self, x):
        batch = x.shape[0]
        attention_feature, out = self.attention(x, x, x, False)
        attention_feature = attention_feature.reshape(batch, -1)

        reconstruct = attention_feature.reshape(x.shape)
        forecast = self.fc_forecast(attention_feature, batch)
        forecast = forecast.reshape(batch, -1, self.feature_num)
        return reconstruct, forecast
        # residuals = x.flip(dims=(1,))
        # input_mask = input_mask.flip(dims=(1,))
        # forecast = x[:, -1:]
        # for i, block in enumerate(self.blocks):
        #     backcast, block_forecast = block(residuals)
        #     residuals = (residuals - backcast) * input_mask
        #     forecast = forecast + block_forecast
        # return forecast
