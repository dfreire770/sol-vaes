import torch
import torch.nn as nn


class ShapePrinter(nn.Module):
   def forward(self, x):
      print("x shape:", x.shape)
      return x

class Flatten(nn.Module):
  def forward(self, x):
    batch_size = x.size()[0]
    return x.view(batch_size, -1)

class UnFlatten(nn.Module):
  def forward(self, x):
    batch_size = x.size()[0]
    return x.view(1, batch_size, 4089, 79)

# torch.Size([1, 1025, 259])
# 255267
  
class ConvVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(ConvVAE, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(8, 16, kernel_size=5),
            nn.ReLU(True),
            Flatten(),
            nn.Linear(323031, self.hidden_dim),
            nn.ReLU(True)
        )
        self.mu_layer = nn.Linear(self.hidden_dim, self.latent_dim)
        self.logvar_layer = nn.Linear(self.hidden_dim, self.latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 323031),
            nn.ReLU(True),
            UnFlatten(),
            nn.ConvTranspose2d(16, 8, kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, kernel_size=5),
            nn.Tanh()
        )

    def reparam_(self, mu, logvar):
        std = torch.exp(logvar)
        epsilon = torch.rand_like(std)
        return mu + std * epsilon

    def encode(self, x):
        x = self.encoder(x)
        mu, logvar = self.mu_layer(x), self.logvar_layer(x)
        return mu, logvar

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam_(mu, logvar)
        x = self.decode(z)
        return x, mu, logvar
