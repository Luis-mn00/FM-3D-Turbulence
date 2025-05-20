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

def fm_interp(model, x, x_lr, steps):
    xt = x
    for i, t in enumerate(torch.linspace(0, 1, steps, device=config.device, dtype=torch.float32), start=1):
        print(f"Step {i}/{steps}")
        pred = model(xt, t.expand(xt.size(0)))
        
        xt = xt + (1 / steps) * (x_lr - xt + t*pred)
        
    return xt

def fm_mask(model, x, x_lr, steps, mask):
    xt = x
    for i, t in enumerate(torch.linspace(0, 1, steps, device=config.device, dtype=torch.float32), start=1):
        print(f"Step {i}/{steps}")
        if t > (1 - 1e-3):
            t = torch.tensor([1 - 1e-3], device=config.device)
        mask_t = mask * (1-t)
        pred = model(xt, t.expand(xt.size(0)))
        x1_pred = xt + (1-t)*pred                    
        x_masked = (1-mask_t)*x1_pred + mask_t*x_lr
        xt = xt + (1 / steps) * (x_masked - xt) / (1-t)
        
    return xt

def fm_interp_sparse_experiment(config, model, nsamples, samples_x, samples_y, samples_ids, perc):
    
    losses = []
    residuals = []
    residuals_gt = []
    residuals_diff = []
    lsim = []
    
    for i in range(nsamples):
        x     = samples_x[i].unsqueeze(0).to(config.device)
        y     = samples_y[i].unsqueeze(0).to(config.device)
        noise = torch.randn((1, config.Model.in_channels, config.Data.grid_size, config.Data.grid_size, config.Data.grid_size), device=config.device).float()

        y_pred = fm_interp(model, noise.clone(), x.clone(), steps=10)
        utils.plot_2d_comparison(x[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
                                 y_pred[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
                                 y[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
                                 f"super_interp_{i}")

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
    
def fm_mask_sparse_experiment(config, model, nsamples, samples_x, samples_y, samples_ids, perc):
    
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

        y_pred = fm_mask(model, noise.clone(), x.clone(), 10, mask)
        utils.plot_2d_comparison(x[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
                                 y_pred[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
                                 y[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
                                 f"super_mask_{i}")

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
    
def fm_diff_mask_sparse_experiment(config, model, nsamples, samples_x, samples_y, samples_ids, perc, w_mask=1, sig=0.044):
    
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

        y_pred = fm_mask(model, noise.clone(), x.clone(), 10, diffuse_masks)
        utils.plot_2d_comparison(x[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
                                 y_pred[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
                                 y[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
                                 f"super_diff_mask_{i}")

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
    with open("configs/config_fm.yml", "r") as f:
        config = yaml.safe_load(f)
    config = utils.dict2namespace(config)
    print(config.device)
    
    print("Loading dataset...")
    num_samples = 10
    dataset = IsotropicTurbulenceDataset(dt=config.Data.dt, grid_size=config.Data.grid_size, crop=config.Data.crop, seed=config.Data.seed, size=config.Data.size, num_samples=num_samples)
    #dataset = BigIsotropicTurbulenceDataset("/mnt/data4/pbdl-datasets-local/3d_jhtdb/isotropic1024coarse.hdf5", sim_group='sim0', norm=True, size=None, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, batch_size=5, num_samples=num_samples, test=True)
    samples_y = dataset.test_dataset
    perc = 5
    samples_x, samples_ids = utils.interpolate_dataset(samples_y, perc/100)

    print("Loading model...")
    model = load_model(config, config.Model.save_path)
    
    #fm_interp_sparse_experiment(config, model, num_samples, samples_x, samples_y, samples_ids, perc=perc)
    #fm_mask_sparse_experiment(config, model, num_samples, samples_x, samples_y, samples_ids, perc)
    fm_diff_mask_sparse_experiment(config, model, num_samples, samples_x, samples_y, samples_ids, perc, w_mask=1, sig=0.044)
