import torch.nn as nn

class M64(nn.Module):
    def __init__(self, content_latent_size = 32, input_channel = 3, flatten_size = 1024):
        super(M64, self).__init__()
        self.content_latent_size = content_latent_size
        self.input_channel = input_channel
        self.flatten = flatten_size