from dataset import IsotropicTurbulenceDataset, BigIsotropicTurbulenceDataset
import utils
import h5py
import torch
import numpy as np
import os
from pbdl.loader import Dataloader

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
"""

data = BigIsotropicTurbulenceDataset("/mnt/data4/pbdl-datasets-local/3d_jhtdb/isotropic1024coarse.hdf5", sim_group='sim0', norm=True, size=500, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, batch_size=5)
train_loader = data.train_loader
sample = next(iter(train_loader))
print(sample.shape)

#dataset = Dataloader("isotropic1024coarse", local_datasets_dir="/mnt/data4/pbdl-datasets-local/3d_jhtdb/")