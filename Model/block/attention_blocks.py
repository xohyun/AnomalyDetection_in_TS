import torch.nn as nn
from Model.layer.Attention_lyr import AttentionLayer, FullAttention, PositionalEmbedding
from Model.layer.forecast_lyr import forecast_lyr
from Model.layer.v_inference_lyr import v_inference_lyr

class attention_blocks(nn.Module):
    '''
    For Self-attention Block
    '''
    def __init__(self, seq_len, feature_num, device, ratio):
        super().__init__()
        self.seq_len = seq_len
        self.feature_num = feature_num

        ### Self attention ###
        factor = 5
        dropout = 0.0
        output_attention = False
        d_model = feature_num  # 512
        n_heads = 1

        # self.blocks = blocks
        self.enc_embedding = PositionalEmbedding(seq_len=self.seq_len, d_model=d_model, n=20000, device=device).to(device=device)
        self.attention = AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                        d_model, n_heads, mix=False).to(device=device)

        self.fc_forecast = forecast_lyr(seq_len, feature_num, ratio).to(device=device)
        self.v_inference = v_inference_lyr(seq_len, feature_num, ratio).to(device=device)

    def forward(self, x, original_data):
        batch = x.shape[0]
 
        #---# Positional encoding #---#
        enc_out = self.enc_embedding(x) # [1,50,512]
        x = enc_out + x ########
        
        #---# attention #---#
        attention_feature, out = self.attention(x, x, x, False)
        attention_feature = attention_feature.reshape(batch, -1)

        reconstruct = attention_feature.reshape(x.shape) # use reshape
        forecast = self.fc_forecast(attention_feature)
        forecast = forecast.reshape(batch, -1, self.feature_num)
        var = self.v_inference(reconstruct + original_data)
        return out, reconstruct, forecast, var
        
        # residuals = x.flip(dims=(1,))
        # input_mask = input_mask.flip(dims=(1,))
        # forecast = x[:, -1:]
        # for i, block in enumerate(self.blocks):
        #     backcast, block_forecast = block(residuals)
        #     residuals = (residuals - backcast) * input_mask
        #     forecast = forecast + block_forecast
        # return forecast
