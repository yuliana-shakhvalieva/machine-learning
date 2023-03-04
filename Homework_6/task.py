import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import upsampling


# Task 1

class Encoder(nn.Module):
    def __init__(self, img_size=128, latent_size=512, start_channels=16, downsamplings=5):
        super().__init__()
        self.model = nn.Sequential(*self.make_model(img_size, latent_size, start_channels, downsamplings))

    def make_model(self, img_size, latent_size, start_channels, downsamplings):
        model = [nn.Conv2d(3, start_channels, 1, stride=1, padding=0), nn.ReLU()]

        for i in range(downsamplings):
            model.append(nn.Conv2d((2 ** i) * start_channels, (2 ** (i + 1)) * start_channels, 3, stride=2, padding=1))
            model.append(nn.BatchNorm2d(2 ** (i + 1) * start_channels))
            model.append(nn.ReLU())

        model.append(nn.Flatten())
        model.append(nn.Linear(start_channels * img_size ** 2 // (2 ** downsamplings), 2 * latent_size))
        model.append(nn.ReLU())
        model.append(nn.Linear(2 * latent_size, 2 * latent_size))

        return model

    def forward(self, x):
        mu, log_sigma = torch.chunk(self.model(x), 2, dim=-1)
        sigma = torch.exp(log_sigma)

        return mu + sigma * torch.randn_like(mu), (mu, sigma)


# Task 2

class Decoder(nn.Module):
    def __init__(self, img_size=128, latent_size=512, end_channels=16, upsamplings=5):
        super().__init__()        
        self.model = nn.Sequential(*self.make_model(img_size, latent_size, end_channels, upsamplings))

    def make_model(self, img_size, latent_size, end_channels, upsamplings):
        start_channels = (2 ** upsamplings) * end_channels
        size = img_size // 2**upsamplings

        model = [nn.Linear(latent_size, 2 * latent_size),
                 nn.ReLU(),
                 nn.Linear(2*latent_size, (end_channels * img_size ** 2) // (2 ** upsamplings)),
                 nn.ReLU(),
                 nn.Unflatten(-1, (start_channels, size, size))]

        for i in reversed(range(upsamplings)):
            model.append(
                nn.ConvTranspose2d((2 ** (i + 1)) * end_channels, (2 ** i) * end_channels, 4, stride=2, padding=1))
            model.append(nn.BatchNorm2d(2 ** i * end_channels))
            model.append(nn.ReLU())

        model.append(nn.Conv2d(end_channels, 3, 1, stride=1, padding=0))
        model.append(nn.Tanh())

        return model

    def forward(self, z):
        return self.model(z)
    
# Task 3

class VAE(nn.Module):
    def __init__(self, img_size=128, downsamplings=5, latent_size=256, down_channels=6, up_channels=6):
        super().__init__()
        self.encoder = Encoder(img_size=img_size, 
                               latent_size=latent_size, 
                               start_channels=down_channels, 
                               downsamplings=downsamplings)
        self.decoder = Decoder(img_size=img_size, 
                               latent_size=latent_size, 
                               end_channels=up_channels, 
                               upsamplings=downsamplings)
        self.param_kld = None

    def kld(self):
        return self.param_kld
        
    def forward(self, x):
        z = self.encode(x)
        x_pred = self.decode(z)
        kld = self.kld()

        return x_pred, kld
    
    def encode(self, x):
        z, (mu, sigma) = self.encoder.forward(x)
        self.param_kld = 0.5 * (sigma ** 2 + mu ** 2 - torch.log(sigma ** 2) - 1)
        return z
    
    def decode(self, z):
        x_pred = self.decoder.forward(z)
        return x_pred
    
    def save(self):
        torch.save(self.state_dict(), 'model.pth')
    
    def load(self):
        torch.load(__file__[:-7] + 'model.pth')