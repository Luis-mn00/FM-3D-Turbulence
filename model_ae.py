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
    
def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=8, num_channels=in_channels, eps=1e-6, affine=True)
    # return torch.nn.GroupNorm(num_groups=20, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv, padding_mode="circular"):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv3d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        padding_mode=padding_mode)

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv, padding_mode="circular"):
        super().__init__()
        self.with_conv = with_conv
        self.padding_mode = padding_mode
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv3d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1, 0, 1)
            if self.padding_mode == "circular":
                x = torch.nn.functional.pad(x, pad, mode="circular")
            else:
                x = torch.nn.functional.pad(x, pad)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool3d(x, kernel_size=2, stride=2)
        return x
    
class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, padding_mode="circular"):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv3d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     padding_mode=padding_mode)
        
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout3d(dropout)
        self.conv2 = torch.nn.Conv3d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     padding_mode=padding_mode)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv3d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1,
                                                     padding_mode=padding_mode)
            else:
                self.nin_shortcut = torch.nn.Conv3d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h
    
class UNetAutoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        ch, out_ch, ch_mult = config.Model.ch, config.Model.out_ch, tuple(config.Model.ch_mult)
        num_res_blocks = config.Model.num_res_blocks
        dropout = config.Model.dropout
        in_channels = config.Model.in_channels
        resolution = config.Data.grid_size
        resamp_with_conv = config.Model.resamp_with_conv
        latent_dim = config.Model.latent_dim  # flat latent dimension

        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        self.conv_in = nn.Conv3d(in_channels, ch, kernel_size=3, padding=1)

        # Downsampling
        in_ch_mult = (1,) + ch_mult
        curr_res = resolution
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
            down = nn.Module()
            down.block = block
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res //= 2
            self.down.append(down)

        # Mid layers
        self.mid = nn.Sequential(
            ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout),
            ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        )

        # Save shape info for flattening/unflattening
        self.latent_shape = (block_in, curr_res, curr_res, curr_res)
        self.flat_dim = block_in * curr_res**3
        self.latent_dim = latent_dim

        self.flatten = nn.Flatten(start_dim=1)

        # MLP bottleneck - adjust hidden layer size as you want
        mlp_hidden_dim = latent_dim * 2  # or something else like latent_dim * 2
        self.to_latent_mlp = nn.Sequential(
            nn.Linear(self.flat_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, latent_dim),
            nn.ReLU()
        )
        self.from_latent_mlp = nn.Sequential(
            nn.Linear(latent_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, self.flat_dim),
            nn.ReLU()
        )

        # Upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks + 1):
                if i_block == num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in + skip_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
            up = nn.Module()
            up.block = block
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res *= 2
            self.up.insert(0, up)

        # Output
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv3d(block_in, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        hs = [self.conv_in(x)]

        # Encoder
        for i_level in range(self.num_resolutions):
            for block in self.down[i_level].block:
                hs.append(block(hs[-1]))
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # Mid layers
        h = self.mid(hs[-1])

        # Flatten and encode to latent vector with MLP
        flat = self.flatten(h)
        z = self.to_latent_mlp(flat)

        # Decode latent vector back to conv shape with MLP
        flat_back = self.from_latent_mlp(z)
        h = flat_back.view(-1, *self.latent_shape)

        # Decoder
        for i_level in reversed(range(self.num_resolutions)):
            for i_block, block in enumerate(self.up[i_level].block):
                h = torch.cat([h, hs.pop()], dim=1)
                h = block(h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # Output
        h = self.norm_out(h)
        return self.conv_out(h), z