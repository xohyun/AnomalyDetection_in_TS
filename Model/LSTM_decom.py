import torch.nn as nn
import torch

class AutoEncoder(nn.Module):
    def __init__(self, num_features, seq_len):
        super(AutoEncoder, self).__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.n = self.num_features * self.seq_len
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
        # x = self.Decoder(x)
        # x = x.reshape(batch, self.seq_len, -1)
        return latent

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

# https://github.com/zhouhaoyi/Informer2020/blob/main/models/attn.py
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4; hour_size = 24
        weekday_size = 7; day_size = 32; month_size = 13

        Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        if freq=='t':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    def forward(self, x):
        x = x.long()
        
        minute_x = self.minute_embed(x[:,:,4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:,:,3])
        weekday_x = self.weekday_embed(x[:,:,2])
        day_x = self.day_embed(x[:,:,1])
        month_x = self.month_embed(x[:,:,0])
        
        return hour_x + weekday_x + day_x + month_x + minute_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)
    
    def forward(self, x):
        return self.embed(x)

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type!='timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        
        return self.dropout(x)

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
        scale = self.scale or 1./sqrt(E)

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


class Model(nn.Module):
    def __init__(self, configs, num_features, device):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.device = device
        self.config = configs

        self.channels = num_features # configs.enc_in
        self.conv1d = configs.conv1d

        self.RIN = configs.RIN
        self.combination =  configs.combination

        self.batch_size = configs.batch_size

        #------------------------#
        #---# Ensemble model #---#
        #------------------------#
        ### AE encoder ###
        self.AE = AutoEncoder(self.channels, self.seq_len).to(device=self.device)

        ### LSTM model ###
        self.LSTM = LSTM(num_features, self.seq_len, 3).to(device=self.device)
        
        ### Self attention ###
        factor = 5
        dropout = 0.0
        output_attention = False
        d_model = 512
        n_heads = 1
        embed = 'fixed'
        freq = 'h'
        
        self.attention = AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                        d_model, n_heads, mix=False)
        
        ### RIN Parameters ###
        if self.RIN:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, 1))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, 1))

        ### combination ###
        if self.combination:
          self.alpha = nn.Parameter(torch.ones(1,1,1))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        batch = x.shape[0]

        if self.RIN:
            print('/// RIN ACTIVATED ///\r', end='')
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev
            x = x * self.affine_weight + self.affine_bias
        
        ae_latent = self.AE(x)
        lstm_feature = self.LSTM(x)
        
        dropout = 0.0
        d_model = 512
        embed = 'fixed'
        freq = 'h'
        self.enc_embedding = DataEmbedding(x, d_model, embed, freq, dropout)
        enc_out = self.enc_embedding(x, x)
        attention_feature = self.attention(enc_out)
        print(">>>>", ae_latent.shape, lstm_feature.shape, attention_feature.shape)
        raise
        '''
        seasonal_init = seasonal_init.squeeze()
        trend_init = trend_init.squeeze()

        trend_output = self.AE_trend(trend_init)
        seasonal_output, h_n, c_n = self.LSTM_seasonal(seasonal_init)

        seasonal_output = seasonal_output + c_n'''
        if self.combination:
            x = (seasonal_output*(self.alpha)) + (trend_output*(1-self.alpha))     
        else:
            x = seasonal_output + trend_output
            # x = trend_output + c_n
        
        if self.RIN:
            x = x - self.affine_bias
            x = x / (self.affine_weight + 1e-10)
            x = x * stdev
            x = x + means

        x = x.reshape(batch, self.seq_len, -1)

        if self.config.mode == 'test' and self.combination:
            print(">>> alpha <<<", self.alpha)
        # if self.config.mode == 'test' and self.combination:
        #     return x, self.alpha
        return x 