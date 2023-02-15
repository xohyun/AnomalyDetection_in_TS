import numpy as np
import torch
import torch.nn as nn
import math

'''# https://github.com/ServiceNow/N-BEATS/blob/master/models/nbeats.py
class ModelBlock(nn.Module):
    def __init__(self, input_size, theta_size: int, layers: int, layer_size: int):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(in_features=input_size, out_features=layer_size)] +
                                      [torch.nn.Linear(in_features=layer_size, out_features=layer_size)
                                       for _ in range(layers - 1)])
        self.basis_parameters = torch.nn.Linear(in_features=layer_size, out_features=theta_size)
    
    def forward(self, x):
        block_input = x
        for layer in self.layers:
            block_input = torch.relu(layer(block_input))
        basis_parameters = self.basis_parameters(block_input)
        return self.basis_function(basis_parameters)
'''

class AE_blocks(nn.Module):
    def __init__(self, seq_len, feature_num, device):
        super().__init__()
        self.seq_len = seq_len
        self.feature_num = feature_num

        self.ae = AutoEncoder(seq_len, feature_num).to(device=device)
        self.forecast = Forecast(seq_len, feature_num).to(device=device)

    def forward(self, x):
        latent, reconstruct_x = self.ae(x)
        forecast = self.forecast(latent)
        reconstruct_x = reconstruct_x.reshape(x.shape)
        return reconstruct_x, forecast

class attention_blocks(nn.Module):
    def __init__(self, device):
        super().__init__()

        ### Self attention ###
        factor = 5
        dropout = 0.0
        output_attention = False
        d_model = 38 #512
        n_heads = 1

        # self.blocks = blocks
        self.attention = AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                        d_model, n_heads, mix=False).to(device=device)
        self.fc1 = nn.Linear(1520, 40*38)
        self.fc2 = nn.Linear(1520, 5)
        
    def forward(self, x):
        batch = x.shape[0]
        attention_feature, out = self.attention(x, x, x, False)
        attention_feature = attention_feature.reshape(batch, -1)
        
        reconstruct = self.fc1(attention_feature)
        reconstruct = reconstruct.reshape(x.shape)
        forecast = self.fc2(attention_feature)

        return reconstruct, forecast
        # residuals = x.flip(dims=(1,))
        # input_mask = input_mask.flip(dims=(1,))
        # forecast = x[:, -1:]
        # for i, block in enumerate(self.blocks):
        #     backcast, block_forecast = block(residuals)
        #     residuals = (residuals - backcast) * input_mask
        #     forecast = forecast + block_forecast
        # return forecast

class rnn_blocks(nn.Module):
    def __init__(self, seq_len, feature_num, device):
        super().__init__()

        self.LSTM = LSTM(feature_num, seq_len, 3).to(device=device)
        self.fc = nn.Linear(237,2) 

    def forward(self, x):
        batch = x.shape[0]
        context_vector = self.LSTM(x)
        
        reconstruct = self.fc(context_vector)
        forecast = self.fc(context_vector)
        return reconstruct, forecast
###########################################################################  delete
###########################################################################
###########################################################################

class AutoEncoder(nn.Module):
    def __init__(self, num_features, seq_len):
        super(AutoEncoder, self).__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.n = int(self.num_features * self.seq_len * 0.8)
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
        
        latent = self.Encoder(x)
        x_reconstruct = self.Decoder(latent)
        x_reconstruct = x_reconstruct.reshape(x.shape)
        return latent, x_reconstruct

class Forecast(nn.Module):
    def __init__(self, seq_len, num_features):
        super(Forecast, self).__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.n = int(self.num_features * self.seq_len * 0.8)
        self.hidden1 = int(self.n / 2)
        self.hidden2 = int(self.n / 8)

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
        forecast = self.Decoder(x)
        forecast = forecast.reshape(batch, int(self.seq_len*0.2), -1)
        return forecast

class PositionalEmbedding(nn.Module):

    def __init__(self, seq_len, d_model, n, device):
        super(PositionalEmbedding, self).__init__() # nn.Module 초기화
        # encoding : (seq_len, d_model)
        self.encoding = torch.zeros(seq_len, d_model, device=device)
        self.encoding.requires_grad = False
        # (seq_len, )
        pos = torch.arange(0, seq_len, device=device)
        # (seq_len, 1)         
        pos = pos.float().unsqueeze(dim=1) # int64 -> float32 (없어도 되긴 함)

        _2i = torch.arange(0, d_model, step=2, device=device).float()

        self.encoding[:, ::2] = torch.sin(pos / (n ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (n ** (_2i / d_model)))


    def forward(self, x):
        # x.shape : (batch, seq_len) or (batch, seq_len, d_model)
        seq_len = x.size()[1] 
        # return : (seq_len, d_model)
        # return matrix will be added to x by broadcasting
        return self.encoding[:seq_len, :]

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
                scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, 
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)
        return self.out_projection(out), attn

class LSTM(nn.Module):
    def __init__(self, n_features, n_seq, num_layers, hidden_dim=256):
        super(LSTM, self).__init__()
        self.n_features = n_features
        self.n_seq = n_seq
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.lstm_layer = nn.LSTM(
            input_size = self.n_features,
            hidden_size = self.hidden_dim,
            num_layers = self.num_layers,
            batch_first = True,
            dropout = 0.25,
        )

        n = self.n_features * self.n_seq
        self.hidden = int(n / 8)

        # self.act_fn = nn.ReLU()
        self.act_fn = nn.Tanh()
        middle_dim = int((self.hidden_dim)/2)
        middle_dim2 = int((self.hidden_dim)/4)
        self.fc1 = nn.Linear(self.hidden_dim, middle_dim)
        self.fc2 = nn.Linear(middle_dim, self.n_features*self.n_seq)
        self.fc3 = nn.Linear(self.hidden_dim, self.n_features*self.n_seq)
        self.fc4 = nn.Linear(self.hidden_dim*2, self.hidden)
        # self.fc4 = nn.Linear(self.middle_dim, self.n_features*self.n_seq)
        # self.fc = nn.Linear(self.hidden_dim, self.pred_len)

    def forward(self, x):
        x, (h_n, c_n) = self.lstm_layer(x) 
        # x shape [batch_size, seq, hidden_size]
        # c_n shape [layer, batch, hidden]

        x = x[:, -1, :] # shape [batch, 1, hidden_size]
        x = self.act_fn(x)

        c_n = c_n[-1, :, :].reshape(x.shape) # final layer

        concats = torch.concat((x, c_n), axis=1) # [batch, hidden*2] 
        concats = self.fc4(concats)


        '''# x = self.fc(x)
        x = self.fc1(x) # [batch, 32]
        x = self.act_fn(x)
        x = self.fc2(x) # [batch, feature*seq]
        x = x.reshape(-1, self.n_seq, self.n_features) # [batch, seq, feature]
        # print(c_n.shape) # [layer, batch, hidden]
        
        c_n = c_n.reshape(x.shape[0], -1) # [batch, hidden]
        c_n = self.fc3(c_n) # [batch, feature*seq]
        c_n = c_n.reshape(x.shape[0],  self.n_seq, self.n_features)'''
        # return x, h_n, c_n
        return concats
###########################################################################
###########################################################################
###########################################################################
###########################################################################

class Model(torch.nn.Module):
    def __init__(self, seq_len, feature_num, stack_num, device):
        super().__init__()
        self.seq_len = seq_len
        self.feature_num = feature_num
        self.stack_num = stack_num
        self.device =device

        self.AE_block = AE_blocks(seq_len, feature_num, device)
        self.attention_block = attention_blocks(device)
        self.rnn_block = rnn_blocks(seq_len, feature_num, device)

    def forward(self, x): # x shape [batch, seq_len, feature_num]
        part_idx = int(self.seq_len * 0.8)
        reconstruct_part = x[:, :part_idx, :]
        forecast_part = x[:, part_idx, :]

        residual = reconstruct_part
        forecasts = []
        for stack in range(self.stack_num):
            #---# AutoEncoder block #---#
            reconstruct, forecast_ae = self.AE_block(residual)
    
            #---# Attention block #---#
            residual -= reconstruct
            reconstruct, forecast_attention = self.attention_block(residual)
            
            #---# RNN block #---#
            residual -= reconstruct
            reconstruct, forecast_rnn = self.rnn_block(residual)

            #---# Concat forecast #---#
            forecasts.append(torch.concat(forecast_ae, forecast_attention, forecast_rnn))

        return forecasts