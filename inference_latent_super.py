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
from model_simple import Model_base
from model_ae import Autoencoder
from model_latent import LatentModel

# Create a folder to save plots
plot_folder = "generated_plots"
os.makedirs(plot_folder, exist_ok=True)

def load_autoencoder(config, ae_path):
    ae = Autoencoder(config)
    ae.load_state_dict(torch.load(ae_path, map_location=config.device))
    ae = ae.to(config.device)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False
    return ae

def load_latent_fm_model(config, model_path):
    model = LatentModel(config)
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model = model.to(config.device)
    model.eval()
    return model

# Flow matching interpolation in latent space

def fm_interp_latent(model, z_lr, z_hr, steps):
    zt = z_lr
    for i, t in enumerate(torch.linspace(0, 1, steps, device=zt.device, dtype=torch.float32), start=1):
        print(f"Latent Step {i}/{steps}")
        pred = model(zt, t.expand(zt.size(0)))
        zt = zt + (1 / steps) * (z_hr - zt + t * pred)
    return zt
    
def fm_interp_sparse_experiment_latent(config, model, ae, nsamples, samples_x, samples_y, samples_ids, perc):
    losses = []
    residuals = []
    residuals_gt = []
    residuals_diff = []
    lsim = []
    for i in range(nsamples):
        x = samples_x[i].unsqueeze(0).to(config.device)
        y = samples_y[i].unsqueeze(0).to(config.device)
        # Encode to latent space
        with torch.no_grad():
            z_lr = ae.encode(x)
            z_hr = ae.encode(y)
        # Flow matching in latent space
        z_pred = fm_interp_latent(model, z_lr, z_hr, steps=10)
        # Decode back to physical space
        with torch.no_grad():
            y_pred = ae.decode(z_pred)
        utils.plot_2d_comparison(x[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
                                 y_pred[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
                                 y[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
                                 f"super_interp_latent_{i}")
        losses.append(torch.sqrt(torch.mean((y_pred - y) ** 2)).item())
        residuals.append(torch.sqrt(torch.mean(utils.compute_divergence(y_pred)**2)).item())
        residuals_gt.append(torch.sqrt(torch.mean(utils.compute_divergence(y)**2)).item())
        residuals_diff.append(abs(residuals[i] - residuals_gt[i]))
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
    with open("configs/config_generative.yml", "r") as f:
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
    ae = load_autoencoder(config, config.Model.ae_path)
    print("Loading latent flow matching model...")
    model = load_latent_fm_model(config, config.Model.save_path)
    
    fm_interp_sparse_experiment_latent(config, model, ae, num_samples, samples_x, samples_y, samples_ids, perc=perc)