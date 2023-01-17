import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, num_features, seq_len):
        super(AutoEncoder, self).__init__()
        self.Encoder = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.BatchNorm1d(seq_len),
            nn.LeakyReLU(),
            # nn.ReLU(),
            nn.Linear(64,128),
            nn.BatchNorm1d(seq_len),
            nn.LeakyReLU(),
            # nn.ReLU()
        )
        self.Decoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(seq_len),
            # nn.LayerNorm(64),
            nn.LeakyReLU(),
            # nn.ReLU(),
            nn.Linear(64, num_features)
        )
        
    def forward(self, x):
        x = self.Encoder(x)
        x = self.Decoder(x)
        return x