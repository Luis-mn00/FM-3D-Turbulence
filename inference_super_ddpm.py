import torch
import matplotlib.pyplot as plt
import os
from types import SimpleNamespace
from tqdm import tqdm
import yaml
import numpy as np
import utils
import random
import math

from dataset import IsotropicTurbulenceDataset, BigIsotropicTurbulenceDataset, BigSpectralIsotropicTurbulenceDataset
import utils
from src.core.models.box.pdedit import PDEDiT3D_S, PDEDiT3D_B, PDEDiT3D_L
from my_config_length import UniProjectionLength
from diffusion import Diffusion

# Create a folder to save plots
plot_folder = "generated_plots"
os.makedirs(plot_folder, exist_ok=True)

# Linear Beta Schedule (from beta_min to beta_max over the T timesteps)
def get_linear_beta_schedule(T, beta_min=1e-4, beta_max=0.02):
    betas = torch.linspace(beta_min, beta_max, T)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas_cumprod  

def ddim_interp(model, x, x_lr, t_start, reverse_steps, betas, alphas_cumprod):
    seq = range(0, t_start, t_start // reverse_steps)
    next_seq = [-1] + list(seq[:-1])
    n = x.size(0)  # Batch size

    with torch.no_grad():
        for i, j in zip(reversed(seq), reversed(next_seq)):
            t = (torch.ones(n) * i).to(x.device)
            #print(f"Step {i}/{t_start}, Time: {t[0].item():.4f}")

            alpha_bar_t = alphas_cumprod[i] if i < len(alphas_cumprod) else alphas_cumprod[-1]
            alpha_bar_next = alphas_cumprod[j] if 0 <= j < len(alphas_cumprod) else alpha_bar_t
            
            e = model(x, t)
            e = e.sample

            # Classic DDIM x0 prediction and update
            x0_pred = (x - e * (1 - alpha_bar_t).sqrt()) / alpha_bar_t.sqrt()
            
            x_interp = x0_pred * (1 - t / t_start) + x_lr * (t / t_start)
            
            x = alpha_bar_next.sqrt() * x_interp + (1 - alpha_bar_next).sqrt() * e

            # Free memory of intermediate tensors
            del e, x0_pred
            torch.cuda.empty_cache()

    return x

def ddim_mask(model, x, x_lr, t_start, reverse_steps, betas, alphas_cumprod, mask):
    seq = range(0, t_start, t_start // reverse_steps)
    next_seq = [-1] + list(seq[:-1])
    n = x.size(0)  # Batch size

    with torch.no_grad():
        for i, j in zip(reversed(seq), reversed(next_seq)):
            t = (torch.ones(n) * i).to(x.device)
            #print(f"Step {i}/{t_start}, Time: {t[0].item():.4f}")

            alpha_bar_t = alphas_cumprod[i] if i < len(alphas_cumprod) else alphas_cumprod[-1]
            alpha_bar_next = alphas_cumprod[j] if 0 <= j < len(alphas_cumprod) else alpha_bar_t
            
            e = model(x, t)
            e = e.sample

            # Classic DDIM x0 prediction and update
            x0_pred = (x - e * (1 - alpha_bar_t).sqrt()) / alpha_bar_t.sqrt()
            
            mask_t = mask * (t / t_start)**3
            x_masked = x0_pred * (1 - mask_t) + x_lr * mask_t
            
            x = alpha_bar_next.sqrt() * x_masked + (1 - alpha_bar_next).sqrt() * e

            # Free memory of intermediate tensors
            del e, x0_pred
            torch.cuda.empty_cache()

    return x

def ddpm_interp_sparse_experiment(dataset, config, model, nsamples, samples_x, samples_y, t_start=1000, reverse_steps=20, T=1000):
    betas, alphas_cumprod = get_linear_beta_schedule(config.Diffusion.num_diffusion_timesteps, config.Diffusion.beta_start, config.Diffusion.beta_end)
    
    losses = []
    residuals = []
    residuals_gt = []
    residuals_diff = []
    lsim = []
    blurriness = []
    
    for i in range(nsamples):
        print(f"Sample {i+1}/{nsamples}")
        x     = samples_x[i].unsqueeze(0).to(config.device)
        y     = samples_y[i].unsqueeze(0).to(config.device)
        noise = torch.randn((1, config.Model.channel_size, config.Data.grid_size, config.Data.grid_size, config.Data.grid_size), device=config.device).float()
        
        y_pred = ddim_interp(model, noise.clone(), x.clone(), t_start, reverse_steps, betas, alphas_cumprod)
        utils.plot_2d_comparison(x[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
                                 y_pred[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
                                 y[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
                                 f"super_interp_ddpm_{i}")

        losses.append(torch.sqrt(torch.mean((y_pred - y) ** 2)).item())
        residuals.append(torch.mean(torch.abs(utils.compute_divergence(dataset.data_scaler.inverse(y_pred[:, :3, :, :, :].to("cpu")), 2*math.pi/config.Data.grid_size))).item())
        residuals_gt.append(torch.mean(torch.abs(utils.compute_divergence(dataset.data_scaler.inverse(y[:, :3, :, :, :].to("cpu")), 2*math.pi/config.Data.grid_size))).item())
        residuals_diff.append(abs(residuals[i] - residuals_gt[i]))
        # Detach tensors before passing them to LSiM_distance
        y = y.detach()
        y_pred = y_pred.detach()
        lsim.append(utils.LSiM_distance_3D(y, y_pred))
        
        y = y.squeeze(0)
        y_pred = y_pred.squeeze(0)
        blurr_pred = utils.compute_blurriness(y_pred.cpu().numpy())
        blurr_gt = utils.compute_blurriness(y.cpu().numpy())
        blurriness.append(abs(blurr_pred - blurr_gt))
        
    print(f"Pixel-wise L2 error: {np.mean(losses):.4f} +/- {np.std(losses):.4f}")
    print(f"Residual L2 norm: {np.mean(residuals):.4f} +/- {np.std(residuals):.4f}") 
    print(f"Residual difference: {np.mean(residuals_diff):.4f} +/- {np.std(residuals_diff):.4f}")
    print(f"Mean LSiM: {np.mean(lsim):.4f} +/- {np.std(lsim):.4f}")
    print(f"Mean blurriness difference: {np.mean(blurriness):.4f} +/- {np.std(blurriness):.4f}")
    
def ddpm_mask_sparse_experiment(dataset, config, model, nsamples, samples_x, samples_y, samples_ids, w_mask=1, t_start=1000, reverse_steps=20, T=1000):
    betas, alphas_cumprod = get_linear_beta_schedule(config.Diffusion.num_diffusion_timesteps, config.Diffusion.beta_start, config.Diffusion.beta_end)
    
    losses = []
    residuals = []
    residuals_gt = []
    residuals_diff = []
    lsim = []
    blurriness = []
    
    for i in range(nsamples):
        print(f"Sample {i+1}/{nsamples}")
        x     = samples_x[i].unsqueeze(0).to(config.device)
        y     = samples_y[i].unsqueeze(0).to(config.device)
        noise = torch.randn((1, config.Model.channel_size, config.Data.grid_size, config.Data.grid_size, config.Data.grid_size), device=config.device).float()

        if samples_ids is not None:
            mask = torch.zeros(config.Data.grid_size, config.Data.grid_size, config.Data.grid_size).flatten()
            mask[samples_ids[i]] = 1
            mask = mask.reshape(config.Data.grid_size, config.Data.grid_size, config.Data.grid_size)
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, D, D, D)
            mask = mask.repeat(1, config.Model.channel_size, 1, 1, 1)  # (1, C, D, D, D)
            mask = mask.to(config.device)
            mask_tmp = torch.rand(noise.shape, device=noise.device) < 1.0
            mask = torch.clamp(mask + mask_tmp, max=1)
        else:
            mask = torch.rand(x.shape, device=x.device) < w_mask

        y_pred = ddim_mask(model, noise.clone(), x.clone(), t_start, reverse_steps, betas, alphas_cumprod, mask)
        utils.plot_2d_comparison(x[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
                                 y_pred[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
                                 y[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
                                 f"super_mask_ddpm_{i}")

        losses.append(torch.sqrt(torch.mean((y_pred - y) ** 2)).item())
        residuals.append(torch.mean(torch.abs(utils.compute_divergence(dataset.data_scaler.inverse(y_pred[:, :3, :, :, :].to("cpu")), 2*math.pi/config.Data.grid_size))).item())
        residuals_gt.append(torch.mean(torch.abs(utils.compute_divergence(dataset.data_scaler.inverse(y[:, :3, :, :, :].to("cpu")), 2*math.pi/config.Data.grid_size))).item())
        residuals_diff.append(abs(residuals[i] - residuals_gt[i]))
        # Detach tensors before passing them to LSiM_distance
        y = y.detach()
        y_pred = y_pred.detach()
        lsim.append(utils.LSiM_distance_3D(y, y_pred))
        
        y = y.squeeze(0)
        y_pred = y_pred.squeeze(0)
        blurr_pred = utils.compute_blurriness(y_pred.cpu().numpy())
        blurr_gt = utils.compute_blurriness(y.cpu().numpy())
        blurriness.append(abs(blurr_pred - blurr_gt))
        
    print(f"Pixel-wise L2 error: {np.mean(losses):.4f} +/- {np.std(losses):.4f}")
    print(f"Residual L2 norm: {np.mean(residuals):.4f} +/- {np.std(residuals):.4f}") 
    print(f"Residual difference: {np.mean(residuals_diff):.4f} +/- {np.std(residuals_diff):.4f}")
    print(f"Mean LSiM: {np.mean(lsim):.4f} +/- {np.std(lsim):.4f}")
    print(f"Mean blurriness difference: {np.mean(blurriness):.4f} +/- {np.std(blurriness):.4f}")
    
def ddpm_diff_mask_sparse_experiment(dataset, config, model, nsamples, samples_x, samples_y, samples_ids, w_mask=1, sig=0.044, t_start=1000, reverse_steps=20, T=1000):
    betas, alphas_cumprod = get_linear_beta_schedule(config.Diffusion.num_diffusion_timesteps, config.Diffusion.beta_start, config.Diffusion.beta_end)
    
    losses = []
    residuals = []
    residuals_gt = []
    residuals_diff = []
    lsim = []
    blurriness = []
    
    if samples_ids is not None:
        diffuse_masks = torch.zeros(len(samples_ids), config.Model.channel_size, config.Data.grid_size, config.Data.grid_size, config.Data.grid_size).to(config.device)
        for j in range(len(samples_ids)):
            # Use the correct number of total voxels for 3D
            total_voxels = config.Data.grid_size ** 3
            ids = list(samples_ids[j]) + random.sample(range(total_voxels), int(total_voxels * w_mask))
            mask = utils.diffuse_mask(
                ids, A=1, sig=sig,
                Nx=config.Data.grid_size,
                Ny=config.Data.grid_size,
                Nz=config.Data.grid_size
            )
            diffuse_masks[j] = torch.tensor(mask, dtype=torch.float).unsqueeze(0).repeat(config.Model.channel_size, 1, 1, 1)
    else:
        diffuse_masks = torch.zeros(nsamples, config.Model.channel_size, config.Data.grid_size, config.Data.grid_size, config.Data.grid_size).to(config.device)
        for j in range(nsamples):
            total_voxels = config.Data.grid_size ** 3
            ids = random.sample(range(total_voxels), int(total_voxels * w_mask))
            mask = utils.diffuse_mask(
                ids, A=1, sig=sig, 
                Nx=config.Data.grid_size,
                Ny=config.Data.grid_size,
                Nz=config.Data.grid_size
            )
            diffuse_masks[j] = torch.tensor(mask, dtype=torch.float).unsqueeze(0).repeat(config.Model.channel_size, 1, 1, 1)
    
    for i in range(nsamples):
        print(f"Sample {i+1}/{nsamples}")
        x     = samples_x[i].unsqueeze(0).to(config.device)
        y     = samples_y[i].unsqueeze(0).to(config.device)
        noise = torch.randn((1, config.Model.channel_size, config.Data.grid_size, config.Data.grid_size, config.Data.grid_size), device=config.device).float()

        y_pred = ddim_mask(model, noise.clone(), x.clone(), t_start, reverse_steps, betas, alphas_cumprod, diffuse_masks[i].unsqueeze(0))
        utils.plot_2d_comparison(x[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
                                 y_pred[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
                                 y[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
                                 f"super_diff_mask_ddpm_{i}")

        losses.append(torch.sqrt(torch.mean((y_pred - y) ** 2)).item())
        residuals.append(torch.mean(torch.abs(utils.compute_divergence(dataset.data_scaler.inverse(y_pred[:, :3, :, :, :].to("cpu")), 2*math.pi/config.Data.grid_size))).item())
        residuals_gt.append(torch.mean(torch.abs(utils.compute_divergence(dataset.data_scaler.inverse(y[:, :3, :, :, :].to("cpu")), 2*math.pi/config.Data.grid_size))).item())
        residuals_diff.append(abs(residuals[i] - residuals_gt[i]))
        # Detach tensors before passing them to LSiM_distance
        y = y.detach()
        y_pred = y_pred.detach()
        lsim.append(utils.LSiM_distance_3D(y, y_pred))
        
        y = y.squeeze(0)
        y_pred = y_pred.squeeze(0)
        blurr_pred = utils.compute_blurriness(y_pred.cpu().numpy())
        blurr_gt = utils.compute_blurriness(y.cpu().numpy())
        blurriness.append(abs(blurr_pred - blurr_gt))
        
    print(f"Pixel-wise L2 error: {np.mean(losses):.4f} +/- {np.std(losses):.4f}")
    print(f"Residual L2 norm: {np.mean(residuals):.4f} +/- {np.std(residuals):.4f}") 
    print(f"Residual difference: {np.mean(residuals_diff):.4f} +/- {np.std(residuals_diff):.4f}")
    print(f"Mean LSiM: {np.mean(lsim):.4f} +/- {np.std(lsim):.4f}")
    print(f"Mean blurriness difference: {np.mean(blurriness):.4f} +/- {np.std(blurriness):.4f}")

# Main script
if __name__ == "__main__":
    print("Loading config...")
    with open("configs/config_ddpm.yml", "r") as f:
        config = yaml.safe_load(f)
    config = utils.dict2namespace(config)
    print(config.device)
    
    # Generate samples using ODE integration
    num_samples = 50
    #dataset = IsotropicTurbulenceDataset(dt=config.Data.dt, grid_size=config.Data.grid_size, crop=config.Data.crop, seed=config.Data.seed, size=config.Data.size, batch_size=config.Training.batch_size, num_samples=num_samples, field=None)
    dataset = BigSpectralIsotropicTurbulenceDataset(grid_size=config.Data.grid_size,
                                                    norm=config.Data.norm,
                                                    size=config.Data.size,
                                                    train_ratio=0.8,
                                                    val_ratio=0.1,
                                                    test_ratio=0.1,
                                                    batch_size=config.Training.batch_size,
                                                    num_samples=num_samples)
    samples_gt = dataset.test_dataset
    
    # Load the trained model
    print("Loading model...")
    model = PDEDiT3D_B(
        channel_size=config.Model.channel_size,
        channel_size_out=config.Model.channel_size_out,
        drop_class_labels=config.Model.drop_class_labels,
        partition_size=config.Model.partition_size,
        mending=False
    )
    model.load_state_dict(torch.load(config.Model.save_path, map_location=config.device))
    model = model.to(config.device)
    model.eval()
    samples_y = dataset.test_dataset
    perc = 20
    samples_x, samples_ids = utils.interpolate_dataset(samples_y, perc/100)
    #samples_x = utils.downscale_data(samples_y, 4)
    #samples_ids = None
    
    ddpm_interp_sparse_experiment(dataset, config, model, num_samples, samples_x, samples_y, reverse_steps=50)
    ddpm_mask_sparse_experiment(dataset, config, model, num_samples, samples_x, samples_y, samples_ids, reverse_steps=50)
    ddpm_diff_mask_sparse_experiment(dataset, config, model, num_samples, samples_x, samples_y, samples_ids, reverse_steps=50)
