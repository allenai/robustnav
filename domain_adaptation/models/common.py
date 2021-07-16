import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_loss(x, mu, logsigma, beta):
    kl = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp())
    return beta * (kl / torch.numel(x))


def vae_loss(x, mu, logsigma, recon_x, beta=1):
    recon_loss = F.mse_loss(x, recon_x, reduction='mean')
    kl = kl_loss(x, mu, logsigma, beta)
    return recon_loss + kl


def reparameterize(mu, logsigma):
    std = torch.exp(0.5*logsigma)
    eps = torch.randn_like(std)
    return mu + eps*std


def carracing_encoder(input_channel):
    return nn.Sequential(
            nn.Conv2d(input_channel, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2), nn.ReLU()
        )


def carracing_decoder(flatten_size):
    return nn.Sequential(
            nn.ConvTranspose2d(flatten_size, 128, 5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 6, stride=2), nn.Sigmoid()
        )