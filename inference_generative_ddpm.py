import torch
import matplotlib.pyplot as plt
import os
import yaml
import numpy as np
from scipy.stats import wasserstein_distance_nd
import math

from dataset import IsotropicTurbulenceDataset, BigIsotropicTurbulenceDataset, BigSpectralIsotropicTurbulenceDataset
import utils
from src.core.models.box.pdedit import PDEDiT3D_S, PDEDiT3D_B, PDEDiT3D_L
from my_config_length import UniProjectionLength
from diffusion import Diffusion

# Create a folder to save plots
plot_folder = "generated_plots"
os.makedirs(plot_folder, exist_ok=True)

def ddim(x, model, t_start, reverse_steps, betas, alphas_cumprod):
    seq = range(0, t_start, t_start // reverse_steps) 
    next_seq = [-1] + list(seq[:-1])
    #seq = list(range(t_start, 0, -t_start // reverse_steps))
    #next_seq = [-1] + (seq[:-1])
    n = x.size(0)

    for i, j in zip(reversed(seq), reversed(next_seq)):
        t = (torch.ones(n) * i).to(x.device)
        #t = torch.full((n,), i / t_start, dtype=torch.float, device=x.device)  # Normalize time to [1, 0]
        print(f"Step {i}/{t_start}, Time: {t[0].item():.4f}")

        alpha_bar_t = alphas_cumprod[i] if i < len(alphas_cumprod) else alphas_cumprod[-1]
        alpha_bar_next = alphas_cumprod[j] if 0 <= j < len(alphas_cumprod) else alpha_bar_t

        # Convert velocity to noise epsilon
        e = model(x, t)
        e = e.sample

        # Classic DDIM x0 prediction and update
        x0_pred = (x - e * (1 - alpha_bar_t).sqrt()) / alpha_bar_t.sqrt()
        x = alpha_bar_next.sqrt() * x0_pred + (1 - alpha_bar_next).sqrt() * e

        # Free memory of intermediate tensors
        del e, x0_pred
        torch.cuda.empty_cache()

    return x

# Linear Beta Schedule (from beta_min to beta_max over the T timesteps)
def get_linear_beta_schedule(T, beta_min=1e-4, beta_max=0.02):
    betas = torch.linspace(beta_min, beta_max, T)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas_cumprod

# Generate samples using the denoising model
def generate_samples_with_denoiser(config, diffusion, model, num_samples):
    betas, alphas_cumprod = get_linear_beta_schedule(config.Diffusion.num_diffusion_timesteps, config.Diffusion.beta_start, config.Diffusion.beta_end)
    samples = []
    for _ in range(num_samples):
        print(f"Generating sample {_+1}/{num_samples}")
        noise = torch.randn((1, config.Model.channel_size, config.Data.grid_size, config.Data.grid_size, config.Data.grid_size), device=config.device)
        y_pred = diffusion.ddim(noise, model, 1000, 20, plot_prog=False)
        #y_pred = ddim(noise, model, 1000, 20, betas, alphas_cumprod)
        samples.append(y_pred.cpu().detach())
        
    return samples

def residual_of_generated(samples, samples_gt, config):
    rmse_loss = np.zeros(len(samples))
    for i in range(len(samples)):
        # Ensure all tensors are on the same device
        sample = samples[i].to(config.device)
        res, = utils.compute_divergence(sample[:, :3, :, :, :], 2*math.pi/config.Data.grid_size)
        rmse_loss[i] = torch.mean(torch.abs(res))
    
    test_residuals = []
    for i in range(len(samples)):
        sample_gt = samples_gt[i].to(config.device)
        sample_gt = sample_gt.unsqueeze(0)
        res_gt, = utils.compute_divergence(sample_gt[:, :3, :, :, :], 2*math.pi/config.Data.grid_size)
        test_residuals.append(torch.mean(torch.abs(res_gt)))
        
    print(f"L2 residual: {np.mean(rmse_loss):.4f} +/- {np.std(rmse_loss):.4f}") 
    test_residuals_np = np.array([r.cpu().item() if torch.is_tensor(r) else r for r in test_residuals])
    print(f"Residual difference: {np.mean(rmse_loss - test_residuals_np)} +/- {np.std(rmse_loss - test_residuals_np)}")

    # Compute L2 norm of the difference between samples and ground truth
    l2_diff_norms = []
    for i in range(len(samples)):
        y = samples_gt[i]  # Ground truth sample
        y_pred = samples[i]  # Retrieve saved y_pred
        l2_diff_norm = torch.sqrt(torch.mean((y - y_pred) ** 2)).item()
        l2_diff_norms.append(l2_diff_norm)

    print(f"Mean L2 difference between generated samples and ground truth: {np.mean(l2_diff_norms)} +/- {np.std(l2_diff_norms)}")

def test_wasserstein(samples, samples_gt, config):
    wasserstein_cmf_distances = []
    for i in range(len(samples)):
        y = samples_gt[i]  # Ground truth sample: (C, D, D, D)
        y_pred = samples[i].squeeze(0)  # Prediction: (C, D, D, D)
        
        wasserstein_cmf_distances.append(utils.wasserstein(y, y_pred))

    # Mean and std
    mean_wasserstein = np.mean(wasserstein_cmf_distances)
    std_wasserstein = np.std(wasserstein_cmf_distances)
    print(f"Wasserstein distance: {mean_wasserstein:.4f} +/- {std_wasserstein:.4f}")

if __name__ == "__main__":
    # Load the configuration
    print("Loading config...")
    with open("configs/config_ddpm.yml", "r") as f:
        config = yaml.safe_load(f)
    config = utils.dict2namespace(config)
    print(config.device)

    # Generate samples using ODE integration
    num_samples = 1
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
    
    # Diffusion parameters
    diffusion = Diffusion(config)
    
    # Generate samples using ODE integration
    print("Generating samples...")
    samples_ddpm = generate_samples_with_denoiser(config, diffusion, model, num_samples)
    for i, sample in enumerate(samples_ddpm):
        utils.plot_slice(sample, 0, 1, 63, f"generated_ddpm_sample_{i}")
        
    residual_of_generated(samples_ddpm, samples_gt, config)
    test_wasserstein(samples_ddpm, samples_gt, config)