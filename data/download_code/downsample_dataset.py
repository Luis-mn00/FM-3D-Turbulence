import torch
import utils
from dataset import BigSpectralIsotropicTurbulenceDataset
import numpy as np

def preprocess_and_save_dataset():
    print("Loading dataset...")
    dataset = torch.load(f'data/data_spectral_128_mindiv.pt', weights_only=False, map_location="cpu")
    if isinstance(dataset, np.ndarray):
        dataset = torch.from_numpy(dataset)
    
    velocity = dataset[:, :3, :, :, :]
    print(velocity.shape)
    
    print("Preprocessing dataset...")
    interpolated_data = utils.downscale_data(velocity, 4)
    print(interpolated_data.shape)
    torch.save(interpolated_data, "data/data_spectral_128_mindiv_down4.pt")
    print(f"Dataset saved")

if __name__ == "__main__":
    
    preprocess_and_save_dataset()