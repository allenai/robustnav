import torch
import torch.nn as nn
from domain_adaptation.models.common import *
from domain_adaptation.models.M64.M64 import M64

class Encoder(nn.Module):
    def __init__(self, content_latent_size = 32, input_channel = 3, flatten_size = 1024):
        super(Encoder, self).__init__()
        self.encoder = carracing_encoder(input_channel)
        self.linear_mu = nn.Linear(flatten_size, content_latent_size)
        self.linear_sigma = nn.Linear(flatten_size, content_latent_size)
    
    def forward(self, x):
        x1 = self.encoder(x)
        x_flatten = x1.flatten(start_dim=1)

        mu = self.linear_mu(x_flatten)
        sigma = self.linear_sigma(x_flatten)

        latent = reparameterize(mu, sigma)

        return mu, sigma, latent

class VAE_64(M64):
    def __init__(self, content_latent_size = 32, input_channel = 3, flatten_size = 1024, **kwargs):
        super(VAE_64, self).__init__(content_latent_size, input_channel, flatten_size)
        self.encoder = Encoder(content_latent_size, input_channel, flatten_size)
        self.decoder_fc1 = nn.Linear(content_latent_size, flatten_size)
        self.decoder = carracing_decoder(flatten_size)
    
    def forward(self, x, return_latent: bool = False):
        mu, sigma, latent = self.encoder(x)
        latent_1 = self.decoder_fc1(latent)
        flatten_x = latent_1.unsqueeze(-1).unsqueeze(-1)
        recon_x = self.decoder(flatten_x)

        if return_latent:
            return mu, sigma, recon_x, flatten_x
        else:
            return mu, sigma, recon_x