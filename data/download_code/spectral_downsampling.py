import torch
import numpy as np
import utils 
from torch.utils.data import DataLoader, Dataset
import h5py
from numpy.fft import fftn, ifftn, fftshift, ifftshift

def spectral_resize_3d(img, target_size):
    # img: (D, H, W), target_size: int
    original_shape = np.array(img.shape)
    scale_factor = np.prod(original_shape) / (target_size ** 3)  # normalize energy

    F = fftn(img)
    F_shifted = fftshift(F)
    center = original_shape // 2
    half_size = target_size // 2

    # Crop the central part of the spectrum
    cropped = F_shifted[
        center[0] - half_size:center[0] + half_size,
        center[1] - half_size:center[1] + half_size,
        center[2] - half_size:center[2] + half_size
    ]

    cropped_unshifted = ifftshift(cropped)
    resized = ifftn(cropped_unshifted)
    resized = np.real(resized) / scale_factor  # apply normalization

    return resized

file_path = "/mnt/data4/pbdl-datasets-local/3d_jhtdb/isotropic1024coarse.hdf5"
grid_size = 128
samples_list = []
with h5py.File(file_path, 'r') as f:
    keys = list(f.keys())
    print(keys)
    
    keys = list(f['sims']['sim0'].keys())
    index = 1
    for key in keys:
        print(f"Sample: {index}")
        sample = f['sims']['sim0'][key]
        sample = np.transpose(sample, (3, 0, 1, 2))
        c, d, h, w = sample.shape
        gs = grid_size
        sample_ds = np.zeros((c, gs, gs, gs), dtype=np.float32)
        for ch in range(c):
            sample_ds[ch] = spectral_resize_3d(sample[ch], gs)
        sample = torch.tensor(sample_ds, dtype=torch.float32)
        samples_list.append(sample)
        index += 1
        
samples_tensor = torch.stack(samples_list)
print(samples_tensor.shape)  
torch.save(samples_tensor, "data_spectral_128.pt")