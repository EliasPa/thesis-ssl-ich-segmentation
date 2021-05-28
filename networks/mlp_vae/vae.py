import torch.nn as nn
import torch
import torch.nn.functional as F

"""

Simple Variational autoencoder [1] (using MLPs)


[1] Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114
"""
class VAE(nn.Module):

    def __init__(self, n_channels, n_latent=10):
        super(VAE, self).__init__()

        enc = [
            nn.Linear(1024, 512),
            nn.ReLU()
        ]

        self.mu_linear = nn.Linear(512, n_latent)
        self.logvar_linear = nn.Linear(512, n_latent)

        dec = [
            nn.Linear(n_latent, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
        ]

        self.encoder = nn.Sequential(*enc)
        self.decoder = nn.Sequential(*dec)

        fin = [
            nn.Sigmoid()
        ]

        self.final = nn.Sequential(*fin)

    def encode(self, x):
        x = self.encoder(x)
        return self.mu_linear(x), self.logvar_linear(x)

    def decode(self, z):
        return self.decoder(z)

    def reparam_trick(self, mu, log_var):
        rand = torch.randn_like(mu)
        return mu + (0.5*torch.exp(log_var)) * rand

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparam_trick(mu, log_var)
        x_hat = self.decode(z)
        reconstruction = self.final(x_hat)
        return reconstruction, mu, log_var

    def loss(self, reconstruction, source, mu, log_var, KL_weight):
        reconstruction_loss = F.binary_cross_entropy(reconstruction, source, reduction='sum') # if 'mean', outputs a mean of all images in batch
        kl_loss = -0.5*torch.sum(1 + log_var - mu**2 - log_var.exp())

        return reconstruction_loss + KL_weight * kl_loss
