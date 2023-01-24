import torch
import torch.nn as nn

# OmniAnomaly (KDD 2019)
class OmniAnomaly(nn.Module):
    def __init__(self, feats):
        super(OmniAnomaly, self).__init__()
        self.beta = 0.01
        self.n_feats = feats
        self.n_hidden = 32
        self.n_latent = 8

        self.lstm = nn.GRU(self.n_feats, self.n_hidden, 2, batch_first=True)
        
        self.encoder = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            # nn.Flatten(start_dim=1),
            nn.Linear(self.n_hidden, 2*self.n_latent)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_feats), nn.Sigmoid()
        )
    
    def forward(self, x, hidden=None):
        hidden = torch.rand(2, 1, self.n_hidden, dtype=torch.float64) if hidden is not None else hidden
        out, hidden = self.lstm(x, hidden)

        ## Encoder
        x = self.encoder(out) # [batch, seq_len, 16]
        mu, logvar = torch.split(x, [self.n_latent, self.n_latent], dim=-1) # [batch, seq_len, 8]

        ## Reparameterization trick
        std = torch.exp(0.5*logvar) # [batch, seq_len, 8]
        eps = torch.randn_like(std)
        x = mu + eps*std

        ## Decoder
        x = self.decoder(x) # [batch, seq_len, feats]
        return x.reshape(-1), mu.reshape(-1), logvar.reshape(-1), hidden