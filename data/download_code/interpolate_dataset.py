import torch
import utils
from dataset import BigSpectralIsotropicTurbulenceDataset
import numpy as np

def preprocess_and_save_dataset():
    print("Loading dataset...")
    dataset = torch.load(f'data/data_spectral_128.pt', weights_only=False)
    if isinstance(dataset, np.ndarray):
        dataset = torch.from_numpy(dataset)
    
    velocity = dataset[:, :3, :, :, :]
    
    print("Preprocessing dataset...")
    interpolated_data, _ = utils.interpolate_dataset(velocity, 5 / 100)
    print(interpolated_data.shape)
    torch.save(interpolated_data, "data/data_spectral_128_5.pt")
    print(f"Dataset saved")

if __name__ == "__main__":
    
    preprocess_and_save_dataset()