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
from model_latent import LatentModel
from my_config_length import UniProjectionLength
from model_vqvae import VQVAE, VAE, AE

# Create a folder to save plots
plot_folder = "generated_plots"
os.makedirs(plot_folder, exist_ok=True)

def load_latent_fm_model(config, model_path):
    model = LatentModel(config)
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model = model.to(config.device)
    model.eval()
    return model

def load_ae_model(config):
    ae = VAE(input_size=config.Model.in_channels, hidden_size=config.Model.hidden_size, depth=config.Model.depth, num_res_block=config.Model.num_res_block, res_size=config.Model.res_size, embedding_size=config.Model.embedding_size,
            device=config.device).to(config.device)
    ae.load_state_dict(torch.load(config.Model.lfm_path, map_location=config.device))
    ae = model.to(config.device)
    ae.eval()
    return ae

def integrate_ode_and_sample(config, model, x_lr, steps=10):
    model.eval().requires_grad_(False)

    xt = x_lr.to(config.device).float()
    for i, t in enumerate(torch.linspace(0, 1, steps, device=config.device), start=1):
        print(f"Step {i}/{steps}")
        pred = model(xt, t.expand(xt.size(0)))
        xt = xt + (1 / steps) * pred

    return xt

def fm_sparse_experiment(config, model, ae, nsamples, samples_x, samples_y, samples_ids, perc):
    
    losses = []
    residuals = []
    residuals_gt = []
    residuals_diff = []
    lsim = []
    
    for i in range(nsamples):
        print(f"Sample {i+1}/{nsamples}")
        x     = samples_x[i].unsqueeze(0).to(config.device)
        y     = samples_y[i].unsqueeze(0).to(config.device)

        z = ae.encode(x)
        z_pred = integrate_ode_and_sample(config, model, z, steps=100)
        y_pred = ae.decode(z_pred)
        utils.plot_2d_comparison(x[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
                                 y_pred[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
                                 y[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
                                 f"super_direct_route_{i}")

        losses.append(torch.sqrt(torch.mean((y_pred - y) ** 2)).item())
        residuals.append(torch.sqrt(torch.mean(utils.compute_divergence(y_pred[:, :3, :, :, :])**2)).item())
        residuals_gt.append(torch.sqrt(torch.mean(utils.compute_divergence(y[:, :3, :, :, :])**2)).item())
        residuals_diff.append(abs(residuals[i] - residuals_gt[i]))
        # Detach tensors before passing them to LSiM_distance
        y = y.detach()
        y_pred = y_pred.detach()
        lsim.append(utils.LSiM_distance(y, y_pred))
        
    print(f"Pixel-wise L2 error: {np.mean(losses):.4f} +/- {np.std(losses):.4f}")
    print(f"Residual L2 norm: {np.mean(residuals):.4f} +/- {np.std(residuals):.4f}") 
    print(f"Residual difference: {np.mean(residuals_diff):.4f} +/- {np.std(residuals_diff):.4f}")
    print(f"Mean LSiM: {np.mean(lsim):.4f} +/- {np.std(lsim):.4f}")

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
    samples_y = dataset.test_dataset
    perc = 5
    samples_x, samples_ids = utils.interpolate_dataset(samples_y, perc/100)

    print("Loading autoencoder...")
    ae = load_ae_model(config)
    print("Loading latent flow matching model...")
    model = load_latent_fm_model(config, config.Model.save_path)
    
    fm_sparse_experiment(config, model, ae, num_samples, samples_x, samples_y, samples_ids, perc=perc)