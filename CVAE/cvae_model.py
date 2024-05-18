import torch
from torch.nn import functional as F
from torch import nn, optim

class CVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, class_size):
        super(CVAE, self).__init__()
        
        #x_dim  = 431
        #hidden_dim = 1025
        #latent_dim = 20
        #class_size = 4

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim #-> in example this is fixed to 400
        self.latent_dim = latent_dim
        self.class_size = class_size

        #print(self.input_dim + self.class_size)

        # Encoder layers
        self.fc1 = nn.Linear(self.input_dim + self.class_size, self.hidden_dim)
        self.fc21 = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc22 = nn.Linear(self.hidden_dim, self.latent_dim)

        # Decoder layers
        self.fc3 = nn.Linear(self.latent_dim + self.class_size, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, self.input_dim)

    def encode(self, x, c):
        #print('x:',x.shape)
        #print('c:',c.shape)
        
        #print('x[0]',x.shape[0])

        c_reshaped = c.unsqueeze(0).expand(x.shape[0], -1)
        
        # Concatenate input data and class condition
        inputs = torch.cat([x, c_reshaped],dim=1)  # Concatenate along the feature dimension
        h1 = F.relu(self.fc1(inputs))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):

        #h3 = F.relu(self.fc3(z))
        #return torch.sigmoid(self.fc4(h3))

        #print('z:',z.shape)
        #x.view(-1, self.input_dim)
        c_reshaped = c.unsqueeze(0).expand(z.shape[0], -1)
        
        # Concatenate latent variables and class condition
        inputs = torch.cat([z, c_reshaped], 1)  # Concatenate along the feature dimension
        h3 = F.relu(self.fc3(inputs))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, c):
        mu, logvar = self.encode(x.view(-1, self.input_dim), c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar
