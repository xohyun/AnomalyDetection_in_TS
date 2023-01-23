import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, num_features, seq_len):
        super(AutoEncoder, self).__init__()
        self.hidden1 = 30
        self.hidden2 = 10

        self.Encoder = nn.Sequential(
            nn.Linear(num_features, self.hidden1),
            nn.BatchNorm1d(seq_len),
            nn.LeakyReLU(),
            # nn.ReLU(),
            nn.Linear(self.hidden1, self.hidden2),
            nn.BatchNorm1d(seq_len),
            nn.LeakyReLU(),
            # nn.ReLU()
        )
        self.Decoder = nn.Sequential(
            nn.Linear(self.hidden2, self.hidden1),
            nn.BatchNorm1d(seq_len),
            # nn.LayerNorm(64),
            nn.LeakyReLU(),
            # nn.ReLU(),
            nn.Linear(self.hidden1, num_features)
        )
        
    def forward(self, x):
        x = self.Encoder(x)
        x = self.Decoder(x)
        return x