import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, unflatten_shape):
        super().__init__()
        self.unflatten_shape = unflatten_shape
    def forward(self, input):
        return input.view(input.size(0), *self.unflatten_shape)


class CVAE_3D(nn.Module):
    def __init__(self, image_channels=3, h_dim=128, z_dim=32):
        super(CVAE_3D, self).__init__()
        print()
        print("[INFO] instantiating pytorch model: 3D CVAE")

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=image_channels, out_channels=16, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=16),
            nn.ReLU(),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=32),
            nn.ReLU(),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=64),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=128),
            nn.ReLU(),
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=128),
            nn.ReLU(),
            Flatten() # reshape layer
        )

        # fully connected layers to compute mu and sigma
        # z_dim is set by user
        # h_dim should be computed manually based on output of convs (in this case 27648)
        self.fc1 = nn.Linear(27648, z_dim)
        self.fc2 = nn.Linear(27648, z_dim)
        # self.fc1 = nn.Linear(h_dim, z_dim)
        # self.fc2 = nn.Linear(h_dim, z_dim)

        # self.fc3 = nn.Linear(z_dim, h_dim) # dense layer to connect to decoder
        self.fc3 = nn.Linear(z_dim, 27648)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.BatchNorm3d(num_features=128),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=128, out_channels=128, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=128),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=64),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=32),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=16),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=16, out_channels=image_channels, kernel_size=4, stride=1, padding=0), # dimensions should be as original
            nn.BatchNorm3d(num_features=3),
            # nn.Sigmoid(),

            # if it does not work without sigmoid:
                # check another batchnorm or relu
                # recover original dims: use nn.linear and reshape to original size
                # nn.conv3d with kernel size equal to input size
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        # std = logvar.mul(0.5).exp_()
        # eps = torch.randn(*mu.size())
        eps = torch.rand_like(std)
        # z = mu + std * eps
        z = eps.mul(std).add_(mu)
        return z

    def bottleneck(self, h):
        # print("[INFO] bottleneck h size:", h.size())
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        # print("[INFO] h size:", h.size()) # torch.Size([10, 27648])
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.decoder(z)
        return z

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        # print("[INFO] Input data shape:", x.size())

        # Step 1: compute representation (fetch it separately for later clustering)
        z_representation = self.representation(x)
        # print("[INFO] Forward z_representation:", z_representation.size())
        # print("[INFO] Reshaped latent z", z_representation.view(z_representation.size(0), 8, 8).size())

        # Step 2: call full CVAE --> encode & decode
        z, mu, logvar = self.encode(x)
        z = self.fc3(z)
        # print("[INFO] Latent z after dense fc:", z.size())
        # print("[INFO] mu:", mu.size())
        # print("[INFO] logvar", logvar.size())

        return self.decode(z), mu, logvar, z_representation


class CVAE_3D_II(nn.Module):
    def __init__(self, image_channels=3, h_dim=128, z_dim=32, input_shape=(3, 64, 64, 64)):
        super(CVAE_3D_II, self).__init__()
        print()
        print("[INFO] instantiating pytorch model: 3D CVAE")

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=image_channels, out_channels=32, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=32),
            nn.ReLU(),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=64),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=128),
            nn.ReLU(),
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=256),
            nn.ReLU(),
            nn.Conv3d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(),
            nn.Conv3d(in_channels=512, out_channels=512, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(),
            Flatten() # reshape layer
        )

        # Automatically compute the flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            h = self.encoder[:-1](dummy)  # pass through all except Flatten
            self.unflatten_shape = h.shape[1:]  # (C, D, H, W)
            self.flattened_size = int(np.prod(self.unflatten_shape))

        # fully connected layers to compute mu and sigma
        self.fc1 = nn.Linear(self.flattened_size, z_dim)
        self.fc2 = nn.Linear(self.flattened_size, z_dim)
        self.fc3 = nn.Linear(z_dim, self.flattened_size)

        self.decoder = nn.Sequential(
            UnFlatten(self.unflatten_shape),
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=512, out_channels=256, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=256),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=128),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=64),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=32),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=16),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=16, out_channels=image_channels, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=image_channels)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        # std = logvar.mul(0.5).exp_()
        # eps = torch.randn(*mu.size())
        eps = torch.rand_like(std)
        # z = mu + std * eps
        z = eps.mul(std).add_(mu)
        return z

    def bottleneck(self, h):
        # print("[INFO] bottleneck h size:", h.size())
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        # print("[INFO] h size:", h.size()) # torch.Size([10, 27648])
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.decoder(z)
        return z

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        # print("[INFO] Input data shape:", x.size())

        # Step 1: compute representation (fetch it separately for later clustering)
        z_representation = self.representation(x)
        # print("[INFO] Forward z_representation:", z_representation.size())
        # print("[INFO] Reshaped latent z", z_representation.view(z_representation.size(0), 8, 8).size())

        # Step 2: call full CVAE --> encode & decode
        z, mu, logvar = self.encode(x)
        z = self.fc3(z)
        # print("[INFO] Latent z after dense fc:", z.size())
        # print("[INFO] mu:", mu.size())
        # print("[INFO] logvar", logvar.size())

        return self.decode(z), mu, logvar, z_representation
    
class ResBlock3D(nn.Module):
    def __init__(self, channels):
        super(ResBlock3D, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        )
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.activation(x + self.block(x))

class SimpleAE3D(nn.Module):
    def __init__(self, image_channels=1, z_dim=8, input_shape=(1, 32, 32, 32)):
        super(SimpleAE3D, self).__init__()
        print("[INFO] Initializing SimpleAE3D - Iteration 3 (Residual & Deeper)")

        # ENCODER
        self.encoder = nn.Sequential(
            nn.Conv3d(image_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            ResBlock3D(32),

            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            ResBlock3D(64),

            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            ResBlock3D(128),

            Flatten()
        )

        # Compute size after conv
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            h = self.encoder(dummy)
            self.flattened_size = h.shape[1]

        # LATENT SPACE
        self.fc_enc = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.LeakyReLU(),
            nn.Linear(512, z_dim)
        )

        self.fc_dec = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, self.flattened_size)
        )

        self.unflatten_shape = (128, input_shape[1]//8, input_shape[2]//8, input_shape[3]//8)

        # DECODER
        self.decoder = nn.Sequential(
            UnFlatten(self.unflatten_shape),

            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            ResBlock3D(64),

            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            ResBlock3D(32),

            nn.ConvTranspose3d(32, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.fc_enc(h)
        h_dec = self.fc_dec(z)
        x_hat = self.decoder(h_dec)
        return x_hat