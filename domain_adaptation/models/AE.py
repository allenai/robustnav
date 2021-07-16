import torch
import torch.nn as nn
from .common import *

class Encoder(nn.Module):
    def __init__(self, content_latent_size = 32, input_channel = 3, flatten_size = 1024):
        super(Encoder, self).__init__()
        self.encoder = carracing_encoder(input_channel)
        self.fc1 = nn.Linear(flatten_size, content_latent_size)
    
    def forward(self, x):
        x1 = self.encoder(x)
        x_flatten = x1.view(x1.size(0), -1)
        return self.fc1(x_flatten)


class AE_64(nn.Module):
    def __init__(self, content_latent_size = 32, input_channel = 3, flatten_size = 1024, **kwargs):
        super(AE_64, self).__init__()
        self.encoder = Encoder(content_latent_size, input_channel, flatten_size)
        self.decoder_fc1 = nn.Linear(content_latent_size, flatten_size)
        self.decoder = carracing_decoder(flatten_size)
    
    def forward(self, x):
        latent = self.encoder(x)
        latent_1 = self.decoder_fc1(latent)
        flatten_x = latent_1.unsqueeze(-1).unsqueeze(-1)
        return self.decoder(flatten_x)