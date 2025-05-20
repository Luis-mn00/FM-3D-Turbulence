from dataset import IsotropicTurbulenceDataset
import utils
import h5py
import torch
import numpy as np
import os

"""
dataset = IsotropicTurbulenceDataset(grid_size=64)

utils.plot_slice(dataset.vorticity_magnitude, 20, 0, 31, "snapshot_64")

residual = utils.compute_divergence(dataset.velocity[20].unsqueeze(0))
residual = residual.unsqueeze(0)
utils.plot_slice(residual, 0, 0, 31, "residual")
"""

current_path = os.getcwd()
print("Current path:", current_path)

# Move to the parent directory
os.chdir("../../../mnt/data4/pbdl-datasets-local/3d_jhtdb/")

# Check the new current path
new_path = os.getcwd()
print("New path after moving up:", new_path)

# Get the complete path of the first (and only) file
files = os.listdir(new_path)
if files:
    file_full_path = os.path.join(new_path, files[0])
    print("Full path to the file:", file_full_path)
else:
    print("No files found in the current directory.")

file_path = "/mnt/data4/pbdl-datasets-local/3d_jhtdb/isotropic1024coarse.hdf5"
with h5py.File(file_path, 'r') as f:
    # List all groups
    print(list(f.keys()))
    # Access a dataset
    data = f['dataset_name'][:]