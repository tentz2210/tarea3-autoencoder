# Adaptado de:
## https://debuggercafe.com/unet-from-scratch-using-pytorch/

import torch
import torch.nn as nn

class DoubleConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        """
        In the original paper implementation, the convolution operations were
        not padded but we are padding them here. This is because, we need the 
        output result size to be same as input size.
        """
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNetEncoder(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down1 = DoubleConvolution(input_channels, 64)
        self.down2 = DoubleConvolution(64, 128)
        self.down3 = DoubleConvolution(128, 256)
        self.down4 = DoubleConvolution(256, 512)
        self.bottleneck = DoubleConvolution(512, 1024)

    def forward(self, x):
        # downsamples from 128x128 to 64x64
        d1 = self.down1(x)
        p1 = self.pool(d1)

        # downsamples from 64x64 to 32x32
        d2 = self.down2(p1)
        p2 = self.pool(d2) 

        # downsamples from 64x64 to 32x32
        d3 = self.down3(p2)
        p3 = self.pool(d3)

        # downsamples from 32x32 to 16x16
        d4 = self.down4(p3)
        p4 = self.pool(d4)

        # downsamples from 16x16 to 8x8 with 1024 channels
        b = self.bottleneck(p4)

        # set skip connections
        skips = [d1, d2, d3, d4]

        return b, skips


class UNetDecoder(nn.Module):
    def __init__(self, output_channels=3):
        super().__init__()

        # upsamples from 8x8 to 16x16
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConvolution(1024, 512)

        # upsamples from 16x16 to 32x32
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConvolution(512, 256)

        # upsamples from 32x32 to 64x64
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConvolution(256, 128)

        # upsamples from 64x64 to 128x128
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConvolution(128, 64)

        # image output layer
        self.out = nn.Sequential(
            nn.Conv2d(64, output_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, bottleneck, skips):
        # get skip connections
        d1, d2, d3, d4 = skips

        x = self.up1(bottleneck)
        x = torch.cat([d4, x], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([d3, x], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([d2, x], dim=1)
        x = self.conv3(x)

        x = self.up4(x)
        x = torch.cat([d1, x], dim=1)
        x = self.conv4(x)

        reconstruction = self.out(x)

        return reconstruction


class UNetAutoencoder(nn.Module):
    def __init__(self, input_channels=3, latent_dim=128):
        super().__init__()

        self.encoder = UNetEncoder(input_channels=input_channels)
        self.decoder = UNetDecoder(output_channels=input_channels)

        # flattens to latent space
        self.flatten = nn.Flatten()
        self.fc_latent = nn.Linear(1024 * 8 * 8, latent_dim)

    def encode(self, x):
        bottleneck, skips = self.encoder(x)

        # get latent space
        z = self.flatten(bottleneck)
        z = self.fc_latent(z)

        return bottleneck, skips, z

    def decode(self, bottleneck, skips):
        return self.decoder(bottleneck, skips)

    def forward(self, x):
        bottleneck, skips, z = self.encode(x)
        reconstruction = self.decode(bottleneck, skips)

        return reconstruction, z