import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, num_features, seq_len):
        super(AutoEncoder, self).__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.n = self.num_features * self.seq_len
        self.hidden1 = int(self.n / 2)
        self.hidden2 = int(self.n / 4)
        self.hidden3 = int(self.n / 8)

        self.Encoder = nn.Sequential(
            nn.Linear(self.n, self.hidden1),
            nn.BatchNorm1d(self.hidden1),
            # nn.LeakyReLU(),
            nn.ReLU(),
            nn.Linear(self.hidden1, self.hidden2),
            nn.BatchNorm1d(self.hidden2),
            # nn.LeakyReLU(),
            nn.ReLU(),
            nn.Linear(self.hidden2, self.hidden3),
            nn.BatchNorm1d(self.hidden3),
            # nn.LeakyReLU(),
            nn.ReLU()
        )
        self.Decoder = nn.Sequential(
            nn.Linear(self.hidden3, self.hidden2),
            nn.BatchNorm1d(self.hidden2),
            nn.ReLU(),
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
        x = self.Encoder(x)
        x = self.Decoder(x)
        x = x.reshape(batch, self.seq_len, -1)
        return x