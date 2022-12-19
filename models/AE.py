import torch
import torch.nn as nn
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self,label_dim=12, dim=8):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(label_dim, dim)
        self.fc21 = nn.Linear(dim, dim)
        self.fc22 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, dim)
        self.fc4 = nn.Linear(dim, label_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x, is_train=False):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        if is_train:
            return self.decode(z), mu, logvar

        if self.training:
            return self.decode(z), mu, logvar
        else:
            return self.decode(z), mu, torch.exp(0.5*logvar)

