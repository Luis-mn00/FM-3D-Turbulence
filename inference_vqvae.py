import torch
import matplotlib.pyplot as plt
import os
from types import SimpleNamespace
from tqdm import tqdm
import yaml
import numpy as np
import utils
import random

from dataset import IsotropicTurbulenceDataset, BigIsotropicTurbulenceDataset
import utils
from model_vqvae import VQVAE, VAE, AE

# Create a folder to save plots
plot_folder = "generated_plots"
os.makedirs(plot_folder, exist_ok=True)

# Load the trained model
def load_model(config, model_path):
    #model = VQVAE(input_size=config.Model.in_channels, hidden_size=config.Model.hidden_size, depth=config.Model.depth, num_res_block=config.Model.num_res_block, res_size=config.Model.res_size, embedding_size=config.Model.embedding_size,
    #             num_embedding=config.Model.num_embedding, device=config.device).to(config.device)
    #model = AE(input_size=config.Model.in_channels, image_size=config.Data.grid_size, hidden_size=config.Model.hidden_size, depth=config.Model.depth, num_res_block=config.Model.num_res_block, res_size=config.Model.res_size, embedding_size=config.Model.embedding_size,
    #            device=config.device, z_dim=config.Model.z_dim).to(config.device)
    model = VAE(input_size=config.Model.in_channels, hidden_size=config.Model.hidden_size, depth=config.Model.depth, num_res_block=config.Model.num_res_block, res_size=config.Model.res_size, embedding_size=config.Model.embedding_size,
                device=config.device).to(config.device)
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model = model.to(config.device)
    model.eval()
    return model

def show_recons(model, samples_x):
    for i in range(samples_x.shape[0]):
        sample = samples_x[i]
        sample = sample.unsqueeze(0)
        sample = sample.to(config.device)
        input = {'uvw': sample, 'duvw': utils.spectral_derivative_3d(sample)}
        output = model(input)
        output = output['uvw']
        
        utils.plot_slice(sample.cpu().detach().numpy(), 0, 0, 62, name=f"vqvae_input_{i}")
        utils.plot_slice(output.cpu().detach().numpy(), 0, 0, 62, name=f"vqvae_output_{i}")

# Main script
if __name__ == "__main__":
    print("Loading config...")
    with open("configs/config_vqvae.yml", "r") as f:
        config = yaml.safe_load(f)
    config = utils.dict2namespace(config)
    print(config.device)
    
    print("Loading dataset...")
    num_samples = 10
    dataset = IsotropicTurbulenceDataset(dt=config.Data.dt, grid_size=config.Data.grid_size, crop=config.Data.crop, seed=config.Data.seed, size=config.Data.size, num_samples=num_samples)
    #dataset = BigIsotropicTurbulenceDataset("/mnt/data4/pbdl-datasets-local/3d_jhtdb/isotropic1024coarse.hdf5", sim_group='sim0', norm=True, size=None, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, batch_size=5, num_samples=num_samples, test=True, grid_size=config.Data.grid_size)
    samples_x = dataset.test_dataset

    print("Loading model...")
    model = load_model(config, config.Model.ae_path)
    
    show_recons(model, samples_x)
    