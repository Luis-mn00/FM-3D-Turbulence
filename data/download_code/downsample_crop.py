import torch
import numpy as np
import torch.nn.functional as F

data = torch.load(f'data/data__0.1_128.pt', weights_only=False)
if isinstance(data, np.ndarray):
    data = torch.from_numpy(data)

print(data.shape)
N_time, N_channels, D, H, W = data.shape

data_reshaped = data.view(N_time * N_channels, 1, D, H, W).float()
data_down = F.interpolate(data_reshaped, size=(64, 64, 64), mode='trilinear', align_corners=False)
data_down = data_down.view(N_time, N_channels, 64, 64, 64)
print(data_down.shape)

# Save the downsampled data
output_path = 'data/data__0.1_64.pt'
torch.save(data_down, output_path)
print(f"Saved downsampled data to {output_path}")