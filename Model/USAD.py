import torch.nn as nn

# USAD (KDD 2020)
class USAD(nn.Module):
    def __init__(self, feats):
        super(USAD, self).__init__()
        self.n_feats = feats
        self.n_hidden = 16
        self.n_latent = 5
        self.n_window = 5
        self.n = self.n_feats * self.n_window

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_latent), nn.ReLU(True)
        )

        self.decoder1 = nn.Sequential(
            nn.Linear(self.n_latent,  self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n), nn.Sigmoid()
        )

        self.decoder2 = nn.Sequential(
            nn.Linear(self.n_latent,  self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n), nn.Sigmoid()
        )

    def forward(self, g):
        ## Encoder
        z = self.encoder(g.viww(1, -1))

        ## Decoder (Phase 1)
        ae1 = self.decoder1(z)
        ae2 = self.decoder2(z)

        ## Encode-Decode (Phase 2)
        ae2ae1 = self.decoder2(self.encoder(ae1))
        return ae1.view(-1), ae2.view(-1), ae2ae1.view(-1)