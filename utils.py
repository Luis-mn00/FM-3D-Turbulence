import matplotlib.pyplot as plt

class StdScaler(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        shape = [1, -1] + [1] * (x.ndim - 2)
        mean = self.mean.reshape(*shape)
        std = self.std.reshape(*shape)
        return (x - mean) / std

    def inverse(self, x):
        shape = [1, -1] + [1] * (x.ndim - 2)
        mean = self.mean.reshape(*shape)
        std = self.std.reshape(*shape)
        return x * std + mean

    def scale(self):
        return self.std
    
def compute_statistics(data):
    mean = data.mean(axis=(0,2,3,4))
    std = data.std(axis=(0,2,3,4))
    return mean, std
    
def plot_slice(data, snapshot_idx, channel_idx, slice_idx):
    """
    Plot a slice of the data at a specific snapshot and channel.
    
    Parameters:
    - data: The dataset containing the flow field.
    - snapshot_idx: Index of the time snapshot to plot.
    - channel_idx: Index of the channel to plot.
    - slice_idx: Index of the slice along the z-axis.
    """
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
    output_file = f'generated_plots/snapshot_{snapshot_idx}_channel_{channel_idx}_slice_{slice_idx}.png'
    plt.savefig(output_file)
    print(f'Plot saved as {output_file}')