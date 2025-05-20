import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from sklearn.metrics import mean_squared_error
import torch
import time
import yaml

from dataset import IsotropicTurbulenceDataset, BigIsotropicTurbulenceDataset
import utils

def reshape_data(data):
    # Ensure data is a numpy array
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    nsamples, N_channels, Nx, Ny, Nz = data.shape
    # Use float32 instead of float64 for memory efficiency
    channel_matrices = np.empty((N_channels, nsamples, Nx*Ny*Nz), dtype=np.float32)
    for c in range(N_channels):
        channel_data = data[:, c, :, :, :].astype(np.float32)  # Ensure float32
        channel_matrix = channel_data.reshape(nsamples, -1)
        channel_matrices[c] = channel_matrix
        print(f"Channel {c} matrix shape for SVD: {channel_matrix.shape}")
    return channel_matrices

# Perform POD using Singular Value Decomposition (SVD)
def compute_pod(data_matrix):
    U, S, Vt = np.linalg.svd(data_matrix, full_matrices=False)
    return U, S, Vt

# Reconstruct a snapshot using a given number of modes
def reconstruct_snapshot(U, S, Vt, num_modes, snapshot_idx, Nx, Ny, Nz):
    # Reconstruct the entire snapshot matrix using the first 'num_modes' modes
    snapshot_flat = U[:, :num_modes] @ np.diag(S[:num_modes]) @ Vt[:num_modes, :]
    
    # Extract the specific snapshot for snapshot_idx and reshape it
    snapshot = snapshot_flat[:, snapshot_idx].reshape(Nx, Ny, Nz) 
    return snapshot

def plot_singular_values(S):
    plt.figure(figsize=(8, 5))
    plt.semilogy(S, 'bo-', label="Singular values")
    plt.xlabel("Mode index")
    plt.ylabel("Singular Value (log scale)")
    plt.title("POD Singular Values")
    plt.legend()
    plt.grid()
    plt.savefig("generated_plots/singular_values.png", bbox_inches='tight')
    plt.close()


# Function to create the selection matrix C
def create_selection_matrix(Nx, Ny, Nz, percentage=0.2):
    total_points = Nx * Ny * Nz
    selected_points = int(percentage * total_points)
    selected_indices = np.random.choice(total_points, selected_points, replace=False)
    
    C = np.zeros((selected_points, total_points))
    C[np.arange(selected_points), selected_indices] = 1
    
    return C, selected_indices

def direct_interpolation(snapshot, indices, Nx, Ny, Nz):
    x_selected, y_selected, z_selected = np.unravel_index(indices, (Nx, Ny, Nz))
    sparse_values = snapshot.flatten()[indices]
    x_full, y_full, z_full = np.meshgrid(np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing='ij')
    interpolated_snapshot = griddata((x_selected, y_selected, z_selected), sparse_values, (x_full, y_full, z_full), method='nearest')

    return interpolated_snapshot

def perform_superresolution(U, num_modes, C, snapshot, Nx, Ny, Nz, config):
    Ur = U[:, :num_modes]
    A = torch.tensor(C @ Ur, device=config.device, dtype=torch.float32)
    b = torch.tensor(C @ np.array(snapshot.flatten()), device=config.device, dtype=torch.float32)
    
    # Solve least squares using torch
    x = torch.linalg.lstsq(A, b).solution
    
    # Compute reconstructed snapshot
    reconstructed_snapshot = (Ur @ x.cpu().numpy()).reshape(Nx, Ny, Nz)

    return reconstructed_snapshot

def plot_sparse_reconstruction(snapshot, interpolated_snapshot, reconstructed_snapshot, name):
    # Plot original snapshot, interpolated snapshot, and reconstructed snapshot
    plt.figure(figsize=(15, 5))

    # Plot the interpolated snapshot
    plt.subplot(1, 3, 1)
    plt.imshow(interpolated_snapshot, cmap="inferno")
    plt.title("Interpolated Snapshot")

    # Plot the reconstructed snapshot
    plt.subplot(1, 3, 2)
    plt.imshow(reconstructed_snapshot, cmap="inferno")
    plt.title("Reconstructed Snapshot")
    
    # Plot the original snapshot
    plt.subplot(1, 3, 3)
    plt.imshow(snapshot, cmap="inferno")
    plt.title("Original Snapshot")

    # Show the plots
    plt.savefig(name, bbox_inches='tight')
    plt.close()
    
    
# Load the configuration
print("Loading config...")
with open("configs/config_pod.yml", "r") as f:
    config = yaml.safe_load(f)
config = utils.dict2namespace(config)
print(config.device)        
    
print("Loading dataset...")
num_samples = 10
dataset = IsotropicTurbulenceDataset(dt=config.Data.dt, grid_size=config.Data.grid_size, crop=config.Data.crop, seed=config.Data.seed, size=config.Data.size, num_samples=num_samples)
#dataset = BigIsotropicTurbulenceDataset("/mnt/data4/pbdl-datasets-local/3d_jhtdb/isotropic1024coarse.hdf5", sim_group='sim0', norm=True, size=None, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, batch_size=5, pod=True)

train_dataset = dataset.train_dataset
test_dataset = dataset.test_dataset
data_matrices = reshape_data(train_dataset)
print(data_matrices.shape)

# Create all code separately for each channel
C, indices = create_selection_matrix(dataset.Nx, dataset.Ny, dataset.Nz, 0.2)
data_test_interpolated = torch.zeros((num_samples, dataset.N_channels, dataset.Nx, dataset.Ny, dataset.Nz), dtype=torch.float32, device=config.device)
data_test_reconstructed = torch.zeros((num_samples, dataset.N_channels, dataset.Nx, dataset.Ny, dataset.Nz), dtype=torch.float32, device=config.device)
for i in range(dataset.N_channels):
    print(f"Processing channel {i+1}/{dataset.N_channels}")
    data_matrix = data_matrices[i].T
    U, S, Vt = compute_pod(data_matrix)
    print("POD computed")

    # Plot singular values
    plot_singular_values(S)

    # Select the first snapshot for demonstration of sparse reconstruction
    snapshot = train_dataset[0, i, :, :, :].reshape(dataset.Nx, dataset.Ny, dataset.Nz)
    interpolated_snapshot = direct_interpolation(snapshot, indices, dataset.Nx, dataset.Ny, dataset.Nz)
    reconstruct_snapshot = perform_superresolution(U, config.Model.N_modes, C, snapshot, dataset.Nx, dataset.Ny, dataset.Nz, config)
    plot_sparse_reconstruction(snapshot[:, :, int(dataset.Nz/2)], interpolated_snapshot[:, :, int(dataset.Nz/2)], reconstruct_snapshot[:, :, int(dataset.Nz/2)], f"generated_plots/pod_reconstruction_{i+1}.png")

    for j in range(num_samples):
        snapshot = test_dataset[j, i, :, :, :].reshape(dataset.Nx, dataset.Ny, dataset.Nz)
        interpolated_snapshot = direct_interpolation(snapshot, indices, dataset.Nx, dataset.Ny, dataset.Nz)
        reconstruct_snapshot = perform_superresolution(U, config.Model.N_modes, C, snapshot, dataset.Nx, dataset.Ny, dataset.Nz, config)
        data_test_interpolated[j, i, :, :, :] = torch.tensor(interpolated_snapshot, dtype=torch.float32, device=config.device)
        data_test_reconstructed[j, i, :, :, :] = torch.tensor(reconstruct_snapshot, dtype=torch.float32, device=config.device)
        

# Calculate metrics
list_rmse = []
list_lsim = []
list_residual = []
list_diff = []
for j in range(num_samples):
    rmse = torch.sqrt(torch.mean((test_dataset[j] - data_test_reconstructed[j]) ** 2)).item()
    list_rmse.append(rmse)
    
    gt_residual = utils.compute_divergence(test_dataset[j, :3].unsqueeze(0))
    pred_residual = utils.compute_divergence(data_test_reconstructed[j, :3].unsqueeze(0))
    residual_norm = torch.sqrt(torch.mean(pred_residual**2)).item()
    residual_diff = abs(residual_norm - torch.sqrt(torch.mean(gt_residual**2)).item())
    list_residual.append(residual_norm)
    list_diff.append(residual_diff)
    
    y = test_dataset[j].unsqueeze(0).detach()
    y_pred = data_test_reconstructed[j].unsqueeze(0).detach()
    list_lsim.append(utils.LSiM_distance(y, y_pred))
     
print(f"RMSE over test snapshots (Gappy POD): {np.mean(list_rmse):.4f} +/- {np.std(list_rmse):.4f}")
print(f"LSiM over test snapshots (Gappy POD): {np.mean(list_lsim):.4f} +/- {np.std(list_lsim):.4f}")
print(f"Residual L2 norm over test snapshots (Gappy POD): {np.mean(list_residual):.4f} +/- {np.std(list_residual):.4f}")
print(f"Residual difference over test snapshots (Gappy POD): {np.mean(list_diff):.4f} +/- {np.std(list_diff):.4f}")
