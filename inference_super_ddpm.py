import torch
import matplotlib.pyplot as plt
import os
from types import SimpleNamespace
from tqdm import tqdm
import yaml
import numpy as np
import utils
import random

from dataset import IsotropicTurbulenceDataset
import utils
from model_simple import Model_base
from diffusion import Diffusion

# Create a folder to save plots
plot_folder = "generated_plots"
os.makedirs(plot_folder, exist_ok=True)

# Load the trained model
def load_model(config, model_path):
    model = Model_base(config)
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model = model.to(config.device)
    model.eval()
    return model

def ddpm_interp_sparse_experiment(config, diffusion, model, nsamples, samples_x, samples_y, art_steps=30, art_start=160, art_K=3):
    
    losses = []
    residuals = []
    residuals_gt = []
    residuals_diff = []
    lsim = []
    
    for i in range(nsamples):
        x     = samples_x[i].unsqueeze(0).to(config.device)
        y     = samples_y[i].unsqueeze(0).to(config.device)
        noise = torch.randn((1, config.Model.in_channels, config.Data.grid_size, config.Data.grid_size, config.Data.grid_size), device=config.device).float()

        y_pred = diffusion.ddim_article(x.clone(), model, art_start, art_steps, K=art_K)
        utils.plot_2d_comparison(x[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
                                 y_pred[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
                                 y[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
                                 f"super_ddpm_interp_{i}")

        losses.append(torch.sqrt(torch.mean((y_pred - y) ** 2)).item())
        residuals.append(torch.sqrt(torch.mean(utils.compute_divergence(y_pred)**2)).item())
        residuals_gt.append(torch.sqrt(torch.mean(utils.compute_divergence(y)**2)).item())
        residuals_diff.append(abs(residuals[i] - residuals_gt[i]))
        # Detach tensors before passing them to LSiM_distance
        y = y.detach()
        y_pred = y_pred.detach()
        lsim.append(utils.LSiM_distance(y, y_pred))
        
    print(f"Pixel-wise L2 error: {np.mean(losses):.4f} +/- {np.std(losses):.4f}")
    print(f"Residual L2 norm: {np.mean(residuals):.4f} +/- {np.std(residuals):.4f}") 
    print(f"Residual difference: {np.mean(residuals_diff):.4f} +/- {np.std(residuals_diff):.4f}")
    print(f"Mean LSiM: {np.mean(lsim):.4f} +/- {np.std(lsim):.4f}")
    
def ddpm_mask_sparse_experiment(config, diffusion, model, nsamples, samples_x, samples_y, samples_ids, mask_steps=100, mask_start=1000, w_mask=1):
    
    losses = []
    residuals = []
    residuals_gt = []
    residuals_diff = []
    lsim = []
    
    for i in range(nsamples):
        x     = samples_x[i].unsqueeze(0).to(config.device)
        y     = samples_y[i].unsqueeze(0).to(config.device)
        noise = torch.randn((1, config.Model.in_channels, config.Data.grid_size, config.Data.grid_size, config.Data.grid_size), device=config.device).float()

        mask = torch.zeros(config.Data.grid_size, config.Data.grid_size, config.Data.grid_size).flatten()
        mask[samples_ids[i]] = 1
        mask = mask.reshape(config.Data.grid_size, config.Data.grid_size, config.Data.grid_size)
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, D, D, D)
        mask = mask.repeat(1, 3, 1, 1, 1)     # (1, 3, D, D, D)
        mask = mask.to(config.device)
        mask_tmp = torch.rand(noise.shape, device=noise.device) < 1.0
        mask = torch.clamp(mask + mask_tmp, max=1)

        y_pred = diffusion.ddim_mask(noise.clone(), model, x.clone(), mask_start, mask_steps, w_mask=w_mask, _mask=mask)
        utils.plot_2d_comparison(x[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
                                 y_pred[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
                                 y[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
                                 f"super_ddpm_mask_{i}")

        losses.append(torch.sqrt(torch.mean((y_pred - y) ** 2)).item())
        residuals.append(torch.sqrt(torch.mean(utils.compute_divergence(y_pred)**2)).item())
        residuals_gt.append(torch.sqrt(torch.mean(utils.compute_divergence(y)**2)).item())
        residuals_diff.append(abs(residuals[i] - residuals_gt[i]))
        # Detach tensors before passing them to LSiM_distance
        y = y.detach()
        y_pred = y_pred.detach()
        lsim.append(utils.LSiM_distance(y, y_pred))
        
    print(f"Pixel-wise L2 error: {np.mean(losses):.4f} +/- {np.std(losses):.4f}")
    print(f"Residual L2 norm: {np.mean(residuals):.4f} +/- {np.std(residuals):.4f}") 
    print(f"Residual difference: {np.mean(residuals_diff):.4f} +/- {np.std(residuals_diff):.4f}")
    print(f"Mean LSiM: {np.mean(lsim):.4f} +/- {np.std(lsim):.4f}")
    
def ddpm_diff_mask_sparse_experiment(config, diffusion, model, nsamples, samples_x, samples_y, samples_ids, mask_steps=100, mask_start=1000, w_mask=1, sig=0.044):
    
    losses = []
    residuals = []
    residuals_gt = []
    residuals_diff = []
    lsim = []
    
    for i in range(nsamples):
        x     = samples_x[i].unsqueeze(0).to(config.device)
        y     = samples_y[i].unsqueeze(0).to(config.device)
        noise = torch.randn((1, config.Model.in_channels, config.Data.grid_size, config.Data.grid_size, config.Data.grid_size), device=config.device).float()

        diffuse_masks = torch.zeros(len(samples_ids), config.Model.in_channels, config.Data.grid_size, config.Data.grid_size, config.Data.grid_size).to(config.device)
        for j in range(len(samples_ids)):
            ids = list(samples_ids[j]) + random.sample(range(256**2), int(256**2*w_mask))
            mask = utils.diffuse_mask(
                ids, A=1, sig=sig,
                Nx=config.Data.grid_size,
                Ny=config.Data.grid_size,
                Nz=config.Data.grid_size
            )
            diffuse_masks[j] = torch.tensor(mask, dtype=torch.float).unsqueeze(0).repeat(config.Model.in_channels, 1, 1, 1)

        y_pred = diffusion.ddim_mask(noise.clone(), model, x.clone(), mask_start, mask_steps, diff_mask=diffuse_masks[i].unsqueeze(0))
        utils.plot_2d_comparison(x[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
                                 y_pred[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
                                 y[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
                                 f"super_ddpm_diff_mask_{i}")

        losses.append(torch.sqrt(torch.mean((y_pred - y) ** 2)).item())
        residuals.append(torch.sqrt(torch.mean(utils.compute_divergence(y_pred)**2)).item())
        residuals_gt.append(torch.sqrt(torch.mean(utils.compute_divergence(y)**2)).item())
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
    with open("configs/config_ddpm.yml", "r") as f:
        config = yaml.safe_load(f)
    config = utils.dict2namespace(config)
    print(config.device)
    
    print("Loading dataset...")
    dataset = IsotropicTurbulenceDataset(dt=config.Data.dt, grid_size=config.Data.grid_size, crop=config.Data.crop, seed=config.Data.seed, size=config.Data.size)
    velocity = dataset.velocity
    
    # Define the dataset split ratios
    train_ratio = 0.8
    val_ratio = 0.1

    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    # Split the dataset randomly with config.Data.seed
    indices = np.arange(total_size)
    np.random.seed(config.Data.seed)
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    train_dataset = torch.utils.data.Subset(velocity, train_indices)
    val_dataset = torch.utils.data.Subset(velocity, val_indices)
    test_dataset = torch.utils.data.Subset(velocity, test_indices)
    
    num_samples = 1
    perc = 5
    samples_y = test_dataset[0:num_samples]
    samples_x, samples_ids = utils.interpolate_dataset(samples_y, perc/100)

    print("Loading model...")
    model = load_model(config, config.Model.save_path)
    
    ddpm_interp_sparse_experiment(config, model, num_samples, samples_x, samples_y, samples_ids, perc=perc)
    #ddpm_mask_sparse_experiment(config, model, num_samples, samples_x, samples_y, samples_ids, perc)
    #ddpm_diff_mask_sparse_experiment(config, model, num_samples, samples_x, samples_y, samples_ids, perc, w_mask=1, sig=0.044)
