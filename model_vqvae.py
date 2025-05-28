import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

class VectorQuantization(nn.Module):
    def __init__(self, embedding_size, num_embedding, vq_commit, decay=0.99, eps=1e-5):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_embedding = num_embedding
        self.decay = decay
        self.eps = eps
        embedding = torch.randn(self.embedding_size, self.num_embedding)
        self.register_buffer('embedding', embedding)
        self.register_buffer('cluster_size', torch.zeros(self.num_embedding))
        self.register_buffer('embedding_mean', embedding.clone())
        self.vq_commit = vq_commit

    def forward(self, input):
        input = input.transpose(1, -1).contiguous()
        flatten = input.view(-1, self.embedding_size)
        dist = (
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ self.embedding
                + self.embedding.pow(2).sum(0, keepdim=True)
        )
        _, embedding_ind = dist.min(1)
        embedding_onehot = F.one_hot(embedding_ind, self.num_embedding).type(flatten.dtype)
        embedding_ind = embedding_ind.view(*input.shape[:-1])
        quantize = self.embedding_code(embedding_ind)
        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(embedding_onehot.sum(0), alpha=1 - self.decay)
            embedding_sum = flatten.transpose(0, 1) @ embedding_onehot
            self.embedding_mean.data.mul_(self.decay).add_(embedding_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                    (self.cluster_size + self.eps) / (n + self.num_embedding * self.eps) * n
            )
            embedding_normalized = self.embedding_mean / cluster_size.unsqueeze(0)
            self.embedding.data.copy_(embedding_normalized)
        diff = self.vq_commit * F.mse_loss(quantize.detach(), input)
        quantize = input + (quantize - input).detach()
        quantize = quantize.transpose(1, -1).contiguous()
        return quantize, diff, embedding_ind

    def embedding_code(self, embedding_ind):
        return F.embedding(embedding_ind, self.embedding.transpose(0, 1))

class ResBlock(nn.Module):
    def __init__(self, hidden_size, res_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm3d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_size, res_size, 3, 1, 1),
            nn.BatchNorm3d(res_size),
            nn.ReLU(inplace=True),
            nn.Conv3d(res_size, hidden_size, 3, 1, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input
        return out


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_res_block, res_size, stride):
        super().__init__()
        if stride == 8:
            blocks = [
                nn.Conv3d(input_size, hidden_size // 2, 4, 2, 1),
                nn.BatchNorm3d(hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.Conv3d(hidden_size // 2, hidden_size // 2, 4, 2, 1),
                nn.BatchNorm3d(hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.Conv3d(hidden_size // 2, hidden_size, 4, 2, 1),
                nn.BatchNorm3d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Conv3d(hidden_size, hidden_size, 3, 1, 1),
            ]
        elif stride == 4:
            blocks = [
                nn.Conv3d(input_size, hidden_size // 2, 4, 2, 1),
                nn.BatchNorm3d(hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.Conv3d(hidden_size // 2, hidden_size, 4, 2, 1),
                nn.BatchNorm3d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Conv3d(hidden_size, hidden_size, 3, 1, 1),
            ]
        elif stride == 2:
            blocks = [
                nn.Conv3d(input_size, hidden_size // 2, 4, 2, 1),
                nn.BatchNorm3d(hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.Conv3d(hidden_size // 2, hidden_size, 3, 1, 1),
            ]
        else:
            raise ValueError('Not valid stride')
        for i in range(num_res_block):
            blocks.append(ResBlock(hidden_size, res_size))
        blocks.extend([
            nn.BatchNorm3d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_size, output_size, 1, 1, 0)])
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_res_block, res_size, stride):
        super().__init__()
        blocks = [nn.Conv3d(input_size, hidden_size, 3, 1, 1)]
        for i in range(num_res_block):
            blocks.append(ResBlock(hidden_size, res_size))
        blocks.extend([
            nn.BatchNorm3d(hidden_size),
            nn.ReLU(inplace=True)])
        if stride == 8:
            blocks.extend([
                nn.ConvTranspose3d(hidden_size, hidden_size // 2, 4, 2, 1),
                nn.BatchNorm3d(hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(hidden_size // 2, hidden_size // 2, 4, 2, 1),
                nn.BatchNorm3d(hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(hidden_size // 2, output_size, 4, 2, 1),
            ])
        elif stride == 4:
            blocks.extend([
                nn.ConvTranspose3d(hidden_size, hidden_size // 2, 4, 2, 1),
                nn.BatchNorm3d(hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(hidden_size // 2, output_size, 4, 2, 1),
            ])
        elif stride == 2:
            blocks.extend([
                nn.ConvTranspose3d(hidden_size, output_size, 4, 2, 1)
            ])
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)
    
class Encoder_fl(nn.Module):
    def __init__(self, input_size, hidden_size, output_channels, num_res_block, res_size, stride, latent_dim):
        super().__init__()
        self.stride = stride
        if stride == 8:
            blocks = [
                nn.Conv3d(input_size, hidden_size // 2, 4, 2, 1),
                nn.BatchNorm3d(hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.Conv3d(hidden_size // 2, hidden_size // 2, 4, 2, 1),
                nn.BatchNorm3d(hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.Conv3d(hidden_size // 2, hidden_size, 4, 2, 1),
                nn.BatchNorm3d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Conv3d(hidden_size, hidden_size, 3, 1, 1),
            ]
        elif stride == 4:
            blocks = [
                nn.Conv3d(input_size, hidden_size // 2, 4, 2, 1),
                nn.BatchNorm3d(hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.Conv3d(hidden_size // 2, hidden_size, 4, 2, 1),
                nn.BatchNorm3d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Conv3d(hidden_size, hidden_size, 3, 1, 1),
            ]
        elif stride == 2:
            blocks = [
                nn.Conv3d(input_size, hidden_size // 2, 4, 2, 1),
                nn.BatchNorm3d(hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.Conv3d(hidden_size // 2, hidden_size, 3, 1, 1),
            ]
        else:
            raise ValueError('Not valid stride')

        for _ in range(num_res_block):
            blocks.append(ResBlock(hidden_size, res_size))

        blocks.extend([
            nn.BatchNorm3d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_size, output_channels, 1, 1, 0),
        ])
        self.blocks = nn.Sequential(*blocks)

        self.output_channels = output_channels
        self.latent_dim = latent_dim
        self.latent_shape = None  # to be defined in first forward pass
        self.mlp = None  # lazy init after knowing flattened size

    def forward(self, x):
        out = self.blocks(x)  # shape: [B, C, D, H, W]
        B, C, D, H, W = out.shape
        self.latent_shape = (C, D, H, W)
        out_flat = out.view(B, -1)  # [B, C*D*H*W]

        # Lazy init of MLP
        if self.mlp is None:
            self.mlp = nn.Sequential(
                nn.Linear(C * D * H * W, self.latent_dim),
                nn.ReLU(),
                nn.Linear(self.latent_dim, self.latent_dim)
            )
            self.mlp.to(x.device)

        return self.mlp(out_flat)  # [B, latent_dim]


class Decoder_fl(nn.Module):
    def __init__(self, latent_dim, output_size, hidden_size, num_res_block, res_size, stride, latent_shape):
        super().__init__()
        self.latent_shape = latent_shape  # (C, D, H, W)
        self.latent_dim = latent_dim
        flat_dim = latent_shape[0] * latent_shape[1] * latent_shape[2] * latent_shape[3]

        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, flat_dim),
            nn.ReLU()
        )

        blocks = [nn.Conv3d(latent_shape[0], hidden_size, 3, 1, 1)]
        for _ in range(num_res_block):
            blocks.append(ResBlock(hidden_size, res_size))

        blocks.extend([
            nn.BatchNorm3d(hidden_size),
            nn.ReLU(inplace=True),
        ])

        if stride == 8:
            blocks.extend([
                nn.ConvTranspose3d(hidden_size, hidden_size // 2, 4, 2, 1),
                nn.BatchNorm3d(hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(hidden_size // 2, hidden_size // 2, 4, 2, 1),
                nn.BatchNorm3d(hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(hidden_size // 2, output_size, 4, 2, 1),
            ])
        elif stride == 4:
            blocks.extend([
                nn.ConvTranspose3d(hidden_size, hidden_size // 2, 4, 2, 1),
                nn.BatchNorm3d(hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(hidden_size // 2, output_size, 4, 2, 1),
            ])
        elif stride == 2:
            blocks.extend([
                nn.ConvTranspose3d(hidden_size, output_size, 4, 2, 1),
            ])
        self.blocks = nn.Sequential(*blocks)

    def forward(self, z):
        x = self.mlp(z)  # [B, flat_dim]
        x = x.view(z.size(0), *self.latent_shape)  # [B, C, D, H, W]
        return self.blocks(x)




class VQVAE(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, depth=2, num_res_block=2, res_size=32, embedding_size=64,
                 num_embedding=512, d_mode=['exact', 'physics'], d_commit=[0.1, 0.0001], vq_commit=0.25, loss_power_vg=2, device='cpu'):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size, embedding_size, num_res_block, res_size, stride=2 ** depth)
        self.quantizer = VectorQuantization(embedding_size, num_embedding, vq_commit)
        self.decoder = Decoder(embedding_size, input_size, hidden_size, num_res_block, res_size, stride=2 ** depth)
        self.d_mode = d_mode
        self.d_commit = d_commit
        self.loss_power = loss_power_vg
        self.device = device

    def encode(self, input):
        x = input
        encoded = self.encoder(x)
        quantized, diff, code = self.quantizer(encoded)
        return quantized, diff, code

    def decode(self, quantized):
        decoded = self.decoder(quantized)
        return decoded

    def decode_code(self, code):
        quantized = self.quantizer.embedding_code(code).transpose(1, -1).contiguous()
        decoded = self.decode(quantized)
        return decoded

    def forward(self, input, Epoch=None):
        output = {'loss': torch.tensor(0, device=self.device, dtype=torch.float32)}
        x = input['uvw']
        quantized, diff, output['code'] = self.encode(x)
        decoded = self.decode(quantized)
        output['uvw'] = decoded
        output['duvw'] = utils.spectral_derivative_3d(output['uvw'])
        output['loss'] = F.mse_loss(output['uvw'], input['uvw']) + diff
        
        for i in range(len(self.d_mode)):
            if self.d_mode[i] == 'exact':
                output['loss'] += self.d_commit[i] * utils.weighted_mse_loss(output['duvw'], input['duvw'])
            elif self.d_mode[i] == 'physics':
                if Epoch and (Epoch > 25):
                    output['loss'] += self.d_commit[i] * utils.physics(output['duvw'], input['duvw'])
            else:
                raise ValueError('Not valid d_mode')
        return output

class AE(nn.Module):
    def __init__(self, input_size=3, image_size=128, hidden_size=128, depth=2, num_res_block=2, res_size=32,
                embedding_size=64, z_dim=512, d_mode=['exact', 'physics'], d_commit=[0.1, 0.0001],
                loss_power_vg=2, device='cpu'):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size, embedding_size, num_res_block, res_size, stride=2 ** depth)
        #self.encoder = Encoder_fl(input_size, hidden_size, embedding_size, num_res_block, res_size, stride=2 ** depth, latent_dim=z_dim)
        
        #dummy_input = torch.randn(1, input_size, image_size, image_size, image_size)  # Example input
        #z_flat = self.encoder(dummy_input)  # [4, 256*16*16*16]
        #latent_shape = self.encoder.latent_shape  # Save shape for decoder
            
        self.decoder = Decoder(z_dim, input_size, hidden_size, num_res_block, res_size, stride=2 ** depth)
        #self.decoder = Decoder_fl(z_dim, input_size, hidden_size, num_res_block, res_size, stride=2 ** depth, latent_shape=latent_shape)
        self.embedding_size = z_dim
        self.d_mode = d_mode
        self.d_commit = d_commit
        self.loss_power = loss_power_vg
        self.device = device

    def encode(self, x):
        encoded = self.encoder(x)
        return encoded

    def decode(self, z):
        return self.decoder(z)

    def forward(self, input, Epoch=None):
        output = {'loss': torch.tensor(0, device=self.device, dtype=torch.float32)}
        x = input['uvw']
        z = self.encode(x)
        decoded = self.decode(z)
        output['uvw'] = decoded
        output['duvw'] = utils.spectral_derivative_3d(output['uvw'])

        # Reconstruction loss
        recon_loss = F.mse_loss(output['uvw'], input['uvw'])

        output['loss'] = recon_loss

        for i in range(len(self.d_mode)):
            if self.d_mode[i] == 'exact':
                output['loss'] += self.d_commit[i] * utils.weighted_mse_loss(output['duvw'], input['duvw'])
            elif self.d_mode[i] == 'physics':
                if Epoch and (Epoch > 25):
                    output['loss'] += self.d_commit[i] * utils.physics(output['duvw'], input['duvw'])
            else:
                raise ValueError('Not valid d_mode')

        return output
    
class VAE(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, depth=2, num_res_block=2, res_size=32,
                embedding_size=64, d_mode=['exact', 'physics'], d_commit=[0.1, 0.0001],
                loss_power_vg=2, device='cpu'):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size, embedding_size * 2, num_res_block, res_size, stride=2 ** depth)
        self.decoder = Decoder(embedding_size, input_size, hidden_size, num_res_block, res_size, stride=2 ** depth)
        self.embedding_size = embedding_size
        self.d_mode = d_mode
        self.d_commit = d_commit
        self.loss_power = loss_power_vg
        self.device = device

    def encode(self, x):
        encoded = self.encoder(x)
        # Split last channel into mean and logvar
        mu, logvar = torch.chunk(encoded, 2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, input, Epoch=None):
        output = {'loss': torch.tensor(0, device=self.device, dtype=torch.float32)}
        x = input['uvw']
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z)
        output['uvw'] = decoded
        output['duvw'] = utils.spectral_derivative_3d(output['uvw'])

        # Reconstruction loss
        recon_loss = F.mse_loss(output['uvw'], input['uvw'])

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.numel()

        output['loss'] = recon_loss + kl_loss

        for i in range(len(self.d_mode)):
            if self.d_mode[i] == 'exact':
                output['loss'] += self.d_commit[i] * utils.weighted_mse_loss(output['duvw'], input['duvw'])
            elif self.d_mode[i] == 'physics':
                if Epoch and (Epoch > 25):
                    output['loss'] += self.d_commit[i] * utils.physics(output['duvw'], input['duvw'])
            else:
                raise ValueError('Not valid d_mode')

        return output
