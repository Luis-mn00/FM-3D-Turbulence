import torch
import os
import numpy as np

# Path to the folder containing the .pt datasets
folder_path = 'long_crop'

# List all .pt files in the folder
pt_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pt')]

# Sort the files to ensure consistent order
pt_files.sort()

# Load and concatenate the datasets
datasets = []
for file in pt_files:
    data = torch.load(file, weights_only=False)
    # Convert numpy arrays to tensors if necessary
    if isinstance(data, np.ndarray):
        data = torch.tensor(data)
    datasets.append(data)

# Concatenate along the first dimension
concatenated_dataset = torch.cat(datasets, dim=0)

# Save the concatenated dataset
output_path = 'data_crop_0.1_128.pt'
torch.save(concatenated_dataset, output_path)

print(f"Concatenated dataset saved with shape {concatenated_dataset.shape}")