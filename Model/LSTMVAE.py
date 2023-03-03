import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class LSTMVAE(nn.Module):
    def __init__(self, input_size, seq_len, hidden_size, latent_size, num_layers=1, batch_norm=True):
        super(LSTMVAE, self).__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        
        self.encoder_lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.encoder_fc_mu = nn.Linear(hidden_size, latent_size)
        self.encoder_fc_logvar = nn.Linear(hidden_size, latent_size)
        
        self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.decoder_fc = nn.Linear(latent_size, hidden_size)
        self.output_fc = nn.Linear(hidden_size, input_size)
        
        self.relu = nn.ReLU()
        self.batchnorm1d_encoder = nn.BatchNorm1d(hidden_size)
        self.batchnorm1d_decoder = nn.BatchNorm1d(hidden_size)

    def encode(self, x):
        x, _ = self.encoder_lstm(x)
        x = x[:, -1, :]  # get only last hidden state
        if self.batch_norm:
            x = self.batchnorm1d_encoder(x)  # apply batch normalization
        mu = self.encoder_fc_mu(x)
        logvar = self.encoder_fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        z = self.decoder_fc(z)
        print("here1", z)
        z = self.relu(z)
        print("here2", z)
        if self.batch_norm:
            z = self.batchnorm1d_decoder(z)  # apply batch normalization
        print("here3", z)
        z = z.unsqueeze(1).repeat(1, self.seq_len, 1)  # repeat along sequence length
        print("here4", z)
        z, _ = self.decoder_lstm(z)
        print("here5", z)
        z = self.output_fc(z)
        print("here6", z)
        
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        print("here7", mu)
        print("hrer7.5", logvar)
        z = self.reparameterize(mu, logvar)
        print("here8", z)
        output = self.decode(z)
        return output, mu, logvar

'''
class LSTMVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(LSTMVAE, self).__init__()

        # LSTM encoder
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # Latent space mean and variance
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

        # LSTM decoder
        self.decoder = nn.LSTM(latent_dim, hidden_dim, batch_first=True)

        # Output layer
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        _, (h_n, c_n) = self.encoder(x)
        h_n = h_n.squeeze()
        mu = self.fc_mu(h_n)
        log_var = self.fc_var(h_n)
        return mu, log_var

    def decode(self, z, x):
        z = z.unsqueeze(1).repeat(1, x.size(1), 1)
        output, _ = self.decoder(z)
        output = self.fc_out(output)
        return output

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        # batch = x.shape[0]
        # x = x.reshape(batch, -1)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        output = self.decode(z, x)
        
        return output, mu, log_var
'''