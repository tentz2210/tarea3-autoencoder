# Adaptado de:
## https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb#scrollTo=40xkM9yZlFYk
## https://www.codecademy.com/article/variational-autoencoder-tutorial-vaes-explained
import torch
import torch.nn as nn

class VAEEncoder(nn.Module):
    def __init__(self, input_channels=3, latent_dim=128):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # latent vector
        self.flatten_dim = 256 * 8 * 8

        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x):
        x = self.conv_layers(x)

        x = torch.flatten(x, start_dim=1)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class VAEDecoder(nn.Module):
    def __init__(self, output_channels=3, latent_dim=128):
        super().__init__()

        # convert latent vector to tensor
        self.fc = nn.Linear(latent_dim, 256 * 8 * 8)

        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 256, 8, 8)
        # output reconstructed image
        x = self.deconv_layers(x)

        return x


class VAE(nn.Module):
    def __init__(self, input_channels=3, latent_dim=128):
        super().__init__()

        self.encoder = VAEEncoder(
            input_channels=input_channels,
            latent_dim=latent_dim,
        )

        self.decoder = VAEDecoder(
            output_channels=input_channels,
            latent_dim=latent_dim,
        )

    def reparameterize(self, mu, logvar):
        # calculate standard deviation
        std = torch.exp(0.5 * logvar)
        # gaussian noise
        epsilon = torch.randn_like(std)
        # reparameterize
        z = mu + epsilon * std

        return z

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)

        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)

        return reconstruction, mu, logvar, z