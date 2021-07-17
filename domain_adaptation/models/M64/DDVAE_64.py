import torch
import torch.nn as nn
from domain_adaptation.models.common import *
from domain_adaptation.models.M64.M64 import M64

class Encoder(nn.Module):
    def __init__(self, class_latent_size:int = 8, content_latent_size:int = 32, input_channel:int = 3, flatten_size:int = 1024):
        super(Encoder, self).__init__()
        self.encoder = carracing_encoder(input_channel)
        self.clean_mu = nn.Linear(flatten_size, content_latent_size)
        self.clean_sigma = nn.Linear(flatten_size, content_latent_size)
        self.noise_mu = nn.Linear(flatten_size, class_latent_size)
        self.noise_sigma = nn.Linear(flatten_size, class_latent_size)
    
    def forward(self, x):
        x1 = self.encoder(x)
        x_flatten = x1.view(x1.size(0), -1)

        c_mu = self.clean_mu(x_flatten)
        c_sigma = self.clean_sigma(x_flatten)

        n_mu = self.noise_mu(x_flatten)
        n_sigma = self.noise_sigma(x_flatten)

        c_latent = reparameterize(c_mu, c_sigma)
        n_latent = reparameterize(n_mu, n_sigma)

        return c_mu, c_sigma, c_latent, n_mu, n_sigma, n_latent

class DDVAE_64(M64):
    def __init__(self, class_latent_size = 8, content_latent_size = 32, input_channel = 3, flatten_size = 1024):
        super(DDVAE_64, self).__init__(content_latent_size, input_channel, flatten_size)
        self.encoder = Encoder(class_latent_size, content_latent_size, input_channel, flatten_size)
        self.c_decoder_fc1 = nn.Linear(content_latent_size, flatten_size)
        self.c_decoder = carracing_decoder(flatten_size)

        self.n_decoder_fc1 = nn.Linear(class_latent_size, flatten_size)
        self.n_decoder = carracing_decoder(flatten_size)
    
    def forward(self, x):
        c_mu, c_sigma, c_latent, n_mu, n_sigma, n_latent = self.encoder(x)
        c_latent = self.c_decoder_fc1(c_latent)
        flatten_x = c_latent.unsqueeze(-1).unsqueeze(-1)
        recon_x = self.c_decoder(flatten_x)

        n_latent = self.n_decoder_fc1(n_latent)
        n_flatten_x = n_latent.unsqueeze(-1).unsqueeze(-1)
        noise_recon_x = self.n_decoder(n_flatten_x)

        return c_mu, c_sigma, recon_x, n_mu, n_sigma, noise_recon_x