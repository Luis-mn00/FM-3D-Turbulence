import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: [batch, 1] or [batch]
        half_dim = self.dim // 2
        device = t.device
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -(np.log(10000.0) / (half_dim - 1)))
        emb = t.float() * emb  # [batch, half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

class LatentModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.z_dim = config.Model.z_dim
        self.time_embed_dim = config.Model.time_embed_dim
        hidden_dim = config.Model.hidden_dim
        n_layers = config.Model.n_layers

        # Time embedding: sinusoidal, then MLP
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(self.time_embed_dim),
            nn.Linear(self.time_embed_dim, hidden_dim),
            nn.ReLU()
        )

        # Main MLP: [z, t_emb] -> z
        layers = []
        input_dim = self.z_dim + hidden_dim
        for i in range(n_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, self.z_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, z, t):
        # z: [batch, z_dim], t: [batch] or [batch, 1]
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_emb = self.time_mlp[0](t)  # Sinusoidal embedding
        t_emb = self.time_mlp[1:](t_emb)
        x = torch.cat([z, t_emb], dim=-1)
        return self.mlp(x)
