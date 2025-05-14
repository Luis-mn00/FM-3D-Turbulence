import torch
import numpy as np

# Load the data
dt = 0.1
grid_size = 128
data = torch.load(f'data_{dt}_{grid_size}.pt', weights_only=False)

# Extract the shape of the data tensor
N_time, N_channels, Nx, Ny, Nz = data.shape

# Print the extracted values
print(f"N_time: {N_time}, N_channels: {N_channels}, Nx: {Nx}, Ny: {Ny}, Nz: {Nz}")

# Consider only the first 3 channels (velocity)
velocity = data[:, :3, :, :, :]

# Compute the vorticity (curl)
def compute_curl(velocity):
    dvx_dy = np.gradient(velocity[:, 0, :, :, :], axis=2)
    dvx_dz = np.gradient(velocity[:, 0, :, :, :], axis=3)
    dvy_dx = np.gradient(velocity[:, 1, :, :, :], axis=1)
    dvy_dz = np.gradient(velocity[:, 1, :, :, :], axis=3)
    dvz_dx = np.gradient(velocity[:, 2, :, :, :], axis=1)
    dvz_dy = np.gradient(velocity[:, 2, :, :, :], axis=2)

    curl_x = dvz_dy - dvy_dz
    curl_y = dvx_dz - dvz_dx
    curl_z = dvy_dx - dvx_dy

    return np.stack((curl_x, curl_y, curl_z), axis=1)

vorticity = compute_curl(velocity)

# Compute the magnitude of the vorticity
vorticity_magnitude = np.linalg.norm(vorticity, axis=1)

# Save the vorticity magnitude as a .pt file
vorticity_magnitude_tensor = torch.tensor(vorticity_magnitude, dtype=torch.float32)

# Add a channel dimension to the vorticity magnitude before saving
vorticity_magnitude_tensor = vorticity_magnitude_tensor.unsqueeze(1)

# Print the shape of the vorticity and its magnitude
print(f"Vorticity magnitude shape: {vorticity_magnitude_tensor.shape}")

# Save the vorticity magnitude with the added channel dimension
torch.save(vorticity_magnitude_tensor, f'vorticity_{dt}_{grid_size}.pt')

print("Vorticity magnitude saved")

