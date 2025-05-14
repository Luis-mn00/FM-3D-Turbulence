import torch
import matplotlib.pyplot as plt

# Load the dataset with weights_only=False (only if you trust the source of the file)
data = torch.load('vorticity_0.1_128.pt', weights_only=False)

# Select parameters for plotting
snapshot_idx = 50  # Index of the time snapshot
channel_idx = 0   # Index of the channel
slice_idx = 44    # Index of the slice along the z-axis

# Extract the specific snapshot, channel, and slice
snapshot = data[snapshot_idx]  # Shape: (channels, Nx, Ny, Nz)
channel_data = snapshot[channel_idx]  # Shape: (Nx, Ny, Nz)
slice_data = channel_data[:, :, slice_idx]  # Shape: (Nx, Ny)

# Plot the selected slice
plt.figure(figsize=(8, 6))
plt.imshow(slice_data, cmap='viridis', origin='lower')
plt.colorbar(label='Value')
plt.title(f'Snapshot {snapshot_idx}, Channel {channel_idx}, Slice {slice_idx}')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Save the plot instead of showing it
output_file = f'snapshot_res.png'
plt.savefig(output_file)
print(f'Plot saved as {output_file}')