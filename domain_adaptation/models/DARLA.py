import torch
import torch.nn as nn
from models.VAE import VAE_64
from .common import *
from models.AE import AE_64
from models.VAE import VAE_64

class DARLA_64(nn.Module):
    def __init__(self, AE_weight_path:str, **kwargs):
        super(DARLA_64, self).__init__()
        self.VAE = VAE_64(**kwargs)
        self.AE = AE_64(**kwargs)
        self.AE.load_state_dict(torch.load(AE_weight_path))
        self.encoder = self.VAE.encoder
    
    def forward(self, x):
        mu, sigma, vae_x = self.VAE(x)
        recon_x = self.AE(vae_x)

        return mu, sigma, recon_x
