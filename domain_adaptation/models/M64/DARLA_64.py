import torch
import torch.nn as nn
from domain_adaptation.common import *
from domain_adaptation.models.M64.AE import AE_64
from domain_adaptation.models.M64.VAE import VAE_64
from domain_adaptation.models.M64.M64 import M64

class DARLA_64(M64):
    def __init__(self, AE_weight_path:str, **kwargs):
        super(DARLA_64, self).__init__(kwargs['content_latent_size'], kwargs['input_channel'], kwargs['flatten_size'])
        self.VAE = VAE_64(**kwargs)
        self.AE = AE_64(**kwargs)
        self.AE.load_state_dict(torch.load(AE_weight_path))
        self.encoder = self.VAE.encoder
    
    def forward(self, x):
        mu, sigma, vae_x = self.VAE(x)
        recon_x = self.AE(vae_x)

        return mu, sigma, recon_x
