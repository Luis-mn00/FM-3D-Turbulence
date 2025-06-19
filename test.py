from dataset import IsotropicTurbulenceDataset, BigIsotropicTurbulenceDataset, BigSpectralIsotropicTurbulenceDataset
import utils
import h5py
import torch
import numpy as np
import os
from pbdl.loader import Dataloader
import math
from torchfsm.plot import plot_3D_field
import matplotlib.pyplot as plt

from torchfsm.operator import Div
from torchfsm.mesh import MeshGrid
from src.core.models.box.pdedit import PDEDiT3D_S

"""
dataset = IsotropicTurbulenceDataset(grid_size=64)

utils.plot_slice(dataset.vorticity_magnitude, 20, 0, 31, "snapshot_64")

residual = utils.compute_divergence(dataset.velocity[20].unsqueeze(0))
residual = residual.unsqueeze(0)
utils.plot_slice(residual, 0, 0, 31, "residual")
"""

"""
file_path = "/mnt/data4/pbdl-datasets-local/3d_jhtdb/isotropic1024coarse.hdf5"
with h5py.File(file_path, 'r') as f:
    keys = list(f.keys())
    print(keys)
    
    fields_max = f['norm_fields_sca_max'][:]
    fields_min = f['norm_fields_sca_min'][:]
    fields_mean = f['norm_fields_sca_mean'][:]
    fields_std = f['norm_fields_sca_std'][:]

    print(fields_max)
    print(fields_min)
    print(fields_mean)
    print(fields_std)
    
    indices = list(f['sims']['sim0'].keys())
    print(len(indices))
    dataset = []
    
    for i in range(len(indices)):
        print(i)
        dataset.append(f['sims']['sim0'][indices[i]][:])
    
    dataset = np.stack(dataset, axis=0)
    dataset = torch.tensor(dataset, dtype=torch.float32)
    dataset = dataset.permute(0, 4, 1, 2, 3)
    print(dataset.shape)
    
dataset = Dataloader("isotropic1024coarse", local_datasets_dir="/mnt/data4/pbdl-datasets-local/3d_jhtdb/", time_steps=None)

file_path = "/mnt/data4/pbdl-datasets-local/3d_jhtdb/isotropic1024coarse.hdf5"
with h5py.File(file_path, 'r') as f:
    keys = list(f.keys())
    print(keys)
    
    fields_max = f['norm_fields_sca_max'][:]
    fields_min = f['norm_fields_sca_min'][:]
    fields_mean = f['norm_fields_sca_mean'][:]
    fields_std = f['norm_fields_sca_std'][:]

    print(fields_max)
    print(fields_min)
    print(fields_mean)
    print(fields_std)
    
    keys = list(f['sims']['sim0']['0'].keys())
    print(keys)
    


data = BigIsotropicTurbulenceDataset("/mnt/data4/pbdl-datasets-local/3d_jhtdb/isotropic1024coarse.hdf5", sim_group='sim0', norm=True, size=500, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, batch_size=1, grid_size=128)
train_loader = data.train_loader
sample = next(iter(train_loader))
sample = data.data_scaler.inverse(sample)
print(sample.shape)
utils.plot_slice(sample, 0, 0, int(512/2), name="original_sample_after")

mesh_grid=MeshGrid([(0, 2*torch.pi, 512),(0, 2*torch.pi, 512), (0, 2*torch.pi, 512)])
x,y,z=mesh_grid.bc_mesh_grid()
u = sample[:, :3, :, :, :]
#utils.plot_slice(u, 0, 0, 63, name="spectral_interp")

div=Div()
div_u=div(u,mesh=mesh_grid)
print(div_u.shape)

#div_u[div_u > 2] = 2
#div_u[div_u < -2] = -2
utils.plot_slice(div_u, 0, 0, int(512/2), name="residual_fsm")
print(torch.sqrt(torch.sum(div_u ** 2)))

div_fdm = utils.compute_divergence(u, h=2*math.pi/512)
div_fdm = div_fdm.unsqueeze(0)
utils.plot_slice(div_fdm, 0, 0, int(512/2), name="residual_fdm")
print(torch.sqrt(torch.sum(div_fdm ** 2)))

# Define input parameters
channel_size = 3  # Number of input channels
channel_size_out = 3  # Number of output channels
partition_size = (2, 2, 2)  # Partition size for region partitioning
drop_class_labels = True  # Whether to drop class labels
mending = False  # Whether to enable mending

# Instantiate the model
model = PDEDiT3D_S(
    channel_size=channel_size,
    channel_size_out=channel_size_out,
    drop_class_labels=drop_class_labels,
    partition_size=partition_size,
    mending=mending
)

data = BigIsotropicTurbulenceDataset("/mnt/data4/pbdl-datasets-local/3d_jhtdb/isotropic1024coarse.hdf5", sim_group='sim0', norm=True, size=500, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, batch_size=1, grid_size=128)
train_loader = data.train_loader
sample = next(iter(train_loader))
hidden_states = sample[:, :3, :, :, :].float()
batch_size = 1
timestep = torch.randint(0, 1000, (batch_size,))


# Move tensors to the same device as the model
#device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
device = 'cpu'
print(device)
model = model.to(device)
hidden_states = hidden_states.to(device)
timestep = timestep.to(device)

print(hidden_states.shape)
print(timestep)
output = model(hidden_states, timestep)
print(output.sample.shape)

dataset = BigSpectralIsotropicTurbulenceDataset(grid_size=128,
                                                    norm=True,
                                                    size=200,
                                                    train_ratio=0.8,
                                                    val_ratio=0.1,
                                                    test_ratio=0.1,
                                                    batch_size=5,
                                                    num_samples=10)

raw_data = dataset.data
print(raw_data.shape)

A = raw_data[0].unsqueeze(0)
B = raw_data[1].unsqueeze(0)
dist = utils.LSiM_distance_3D(A, B)
print(dist)


mesh_grid=MeshGrid([(0, 2*torch.pi, 128),(0, 2*torch.pi, 128), (0, 2*torch.pi, 128)])
x,y,z=mesh_grid.bc_mesh_grid()
x = x.squeeze(0).squeeze(0).squeeze(0)
y = y.squeeze(0).squeeze(0).squeeze(0)
z = z.squeeze(0).squeeze(0).squeeze(0)

#field = torch.stack([-y, x, torch.zeros_like(z)], dim=0).unsqueeze(0)
#field = torch.stack([-y*z, x*z, torch.zeros_like(z)], dim=0).unsqueeze(0)
field = torch.stack([x, y, z], dim=0).unsqueeze(0)
print(field.shape)

div=Div()
print(field.shape)
div_u=div(field,mesh=mesh_grid)
utils.plot_slice(div_u, 0, 0, int(128/2), name="residual_fsm")

div_fdm = utils.compute_divergence(field, h=2*math.pi/128)
div_fdm = div_fdm.unsqueeze(0)
utils.plot_slice(div_fdm, 0, 0, int(128/2), name="residual_fdm")

print(f"||div_u|| (Div operator): {torch.sqrt(torch.mean(div_u ** 2)).item():.8e}")
print(f"||div_fdm|| (FD method): {torch.sqrt(torch.mean(div_fdm ** 2)).item():.8e}")


dataset_original = torch.load(f'data/data_spectral_128.pt', weights_only=False)
if isinstance(dataset_original, np.ndarray):
    dataset_original = torch.from_numpy(dataset_original)

vx_original = dataset_original[0, 0, :, :, :]
vy_original = dataset_original[0, 1, :, :, :]
vz_original = dataset_original[0, 2, :, :, :]
print(vx_original.shape)

fig, ax = plt.subplots(figsize=(8, 8))
plot_3D_field(ax=ax, data=vx_original, cmap="PiYG")
plt.savefig("generated_plots/vx_original.png")
fig, ax = plt.subplots(figsize=(8, 8))
plot_3D_field(ax=ax, data=vy_original, cmap="PiYG")
plt.savefig("generated_plots/vy_original.png")
fig, ax = plt.subplots(figsize=(8, 8))
plot_3D_field(ax=ax, data=vz_original, cmap="PiYG")
plt.savefig("generated_plots/vz_original.png")

dataset_optim = torch.load(f'data/data_spectral_128_mindiv.pt', weights_only=False)
if isinstance(dataset_optim, np.ndarray):
    dataset_optim = torch.from_numpy(dataset_optim)
    
utils.plot_slice(dataset_optim.cpu(), 0, 0, int(128/2), name="sample_optim")
    
vx_optim = dataset_optim[0, 0, :, :, :]
vy_optim = dataset_optim[0, 1, :, :, :]
vz_optim = dataset_optim[0, 2, :, :, :]
print(vx_optim.shape)

fig, ax = plt.subplots(figsize=(8, 8))
plot_3D_field(ax=ax, data=vx_optim, cmap="PiYG")
plt.savefig("generated_plots/vx_optim.png")
fig, ax = plt.subplots(figsize=(8, 8))
plot_3D_field(ax=ax, data=vy_optim, cmap="PiYG")
plt.savefig("generated_plots/vy_optim.png")
fig, ax = plt.subplots(figsize=(8, 8))
plot_3D_field(ax=ax, data=vz_optim, cmap="PiYG")
plt.savefig("generated_plots/vz_optim.png")
"""

dataset_optim = torch.load(f'data/data_spectral_128_mindiv.pt', weights_only=False, map_location='cpu')
if isinstance(dataset_optim, np.ndarray):
    dataset_optim = torch.from_numpy(dataset_optim)

velocity = dataset_optim[0:10, :3, :, :, :]
print(velocity.shape)

utils.compute_energy_spectrum(velocity, "energy", device="cuda:0")

#utils.compute_energy_spectrum_original("/mnt/data4/pbdl-datasets-local/3d_jhtdb/isotropic1024coarse.hdf5", "energy_original")


sig = [0.004, 0.008, 0.012, 0.016, 0.020, 0.024, 0.028, 0.032, 0.036, 0.040, 0.044, 0.048, 0.052, 0.056, 0.060, 0.07, 0.08]
rmse = [0.3022, 0.3023, 0.3019, 0.3021, 0.3021, 0.3023, 0.3022, 0.3020, 0.3023, 0.3020, 0.3022, 0.3022, 0.3022, 0.3022, 0.3023, 0.3022, 0.3022]
lsim = [0.0818, 0.0831, 0.0834, 0.0836, 0.0845, 0.0848, 0.0833, 0.0841, 0.0842, 0.0837, 0.0862, 0.0837, 0.0839, 0.0816, 0.0870, 0.0834, 0.0863]
residual = [0.7349, 0.7369, 0.7356, 0.7346, 0.7368, 0.7353, 0.7352, 0.7363, 0.7365, 0.7353, 0.7348, 0.7347, 0.7363, 0.7364, 0.7361, 0.7369, 0.7344]
sharpness = [0.0651, 0.0653, 0.0647, 0.0650, 0.0647, 0.0654, 0.0652, 0.0646, 0.0655, 0.0653, 0.0656, 0.0657, 0.0648, 0.0651, 0.0650, 0.0660, 0.0654]

# Define the lists and labels
metrics = {
    "LSIM": lsim,
    "Residual": residual,
    "Sharpness": sharpness,
    "RMSE": rmse
}

# Normalize each list by its maximum value
normalized_metrics = {name: [value / max(values) for value in values] for name, values in metrics.items()}

# Create a single plot for comparison
plt.figure(figsize=(10, 6))
for metric_name, metric_values in normalized_metrics.items():
    plt.plot(sig, metric_values, marker='o', label=metric_name)

# Add labels, title, and legend
plt.xlabel("Sigma")
plt.ylabel("Normalized Metric Value")
plt.title("Comparison of Metrics vs Sigma")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()

# Save the plot
plt.savefig("generated_plots/metrics_vs_sigma.png")
plt.show()

    
    