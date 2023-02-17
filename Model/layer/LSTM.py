import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, n_features, n_seq, num_layers, hidden_dim=256):
        super(LSTM, self).__init__()
        self.n_features = n_features
        self.n_seq = n_seq
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.lstm_layer = nn.LSTM(
            input_size=self.n_features,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.25,
        )

        n = self.n_features * self.n_seq
        self.hidden = int(n / 8)

        # self.act_fn = nn.ReLU()
        self.act_fn = nn.Tanh()
        middle_dim = int((self.hidden_dim)/2)
        middle_dim2 = int((self.hidden_dim)/4)
        self.fc1 = nn.Linear(self.hidden_dim, middle_dim)
        self.fc2 = nn.Linear(middle_dim, self.n_features*self.n_seq)
        self.fc3 = nn.Linear(self.hidden_dim*2, n)
        self.fc4 = nn.Linear(self.hidden_dim*2, self.hidden)
        # self.fc4 = nn.Linear(self.middle_dim, self.n_features*self.n_seq)
        # self.fc = nn.Linear(self.hidden_dim, self.pred_len)

    def forward(self, x):
        x, (h_n, c_n) = self.lstm_layer(x)
        # x shape [batch_size, seq, hidden_size]
        # c_n shape [layer, batch, hidden]

        x = x[:, -1, :]  # shape [batch, 1, hidden_size]
        x = self.act_fn(x)

        c_n = c_n[-1, :, :].reshape(x.shape)  # final layer

        concats = torch.concat((x, c_n), axis=1)  # [batch, hidden*2]
        concats = self.fc3(concats)

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
