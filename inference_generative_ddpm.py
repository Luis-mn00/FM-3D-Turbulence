import torch
import matplotlib.pyplot as plt
import os
import yaml
import numpy as np
from scipy.stats import wasserstein_distance_nd

from dataset import IsotropicTurbulenceDataset, BigIsotropicTurbulenceDataset
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

def ddim(x, model, t_start, reverse_steps, betas, alphas_cumprod):
    seq = list(range(t_start, 0, -t_start // reverse_steps))
    next_seq = [-1] + seq[:-1]
    n = x.size(0)  # Batch size

    for i, j in zip(reversed(seq), reversed(next_seq)):
        t = torch.full((n,), i / t_start, dtype=torch.float, device=x.device)  # Normalize time to [1, 0]
        t = t_start - t * t_start
        print(f"Step {i}/{t_start}, Time: {t[0].item():.4f}")

        alpha_bar_t = alphas_cumprod[i] if i < len(alphas_cumprod) else alphas_cumprod[-1]
        alpha_bar_next = alphas_cumprod[j] if 0 <= j < len(alphas_cumprod) else alpha_bar_t

        # Convert velocity to noise epsilon
        e = model(x, t)

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
        noise = torch.randn((1, config.Model.in_channels, config.Data.grid_size, config.Data.grid_size, config.Data.grid_size), device=config.device)
        #y_pred, _ = diffusion.ddpm(noise, model, 1000, plot_prog=False)
        y_pred = ddim(noise, model, 1000, 20, betas, alphas_cumprod)
        samples.append(y_pred.cpu().detach())
        
    return samples

def residual_of_generated(samples, samples_gt, config):
    rmse_loss = np.zeros(len(samples))
    for i in range(len(samples)):
        print("Batch", i)
        # Ensure all tensors are on the same device
        sample = samples[i].to(config.device)
        res, = utils.compute_divergence(sample[:, :3, :, :, :])
        rmse_loss[i] = torch.sqrt(torch.mean(res**2))
    
    test_residuals = []
    for i in range(len(samples)):
        sample_gt = samples_gt[i].to(config.device)
        sample_gt = sample_gt.unsqueeze(0)
        res_gt, = utils.compute_divergence(sample_gt[:, :3, :, :, :])
        test_residuals.append(torch.sqrt(torch.mean(res_gt**2)))
        
    print(f"L2 residual: {np.mean(rmse_loss):.2f} +/- {np.std(rmse_loss):.2f}") 
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

    num_samples = 1
    dataset = IsotropicTurbulenceDataset(dt=config.Data.dt, grid_size=config.Data.grid_size, crop=config.Data.crop, seed=config.Data.seed, size=config.Data.size, num_samples=num_samples)
    #dataset = BigIsotropicTurbulenceDataset("/mnt/data4/pbdl-datasets-local/3d_jhtdb/isotropic1024coarse.hdf5", sim_group='sim0', norm=True, size=None, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, batch_size=5, num_samples=num_samples, test=True)
    samples_gt = dataset.test_dataset
    
    # Load the trained model
    print("Loading model...")
    model = load_model(config, config.Model.save_path)
    
    # Diffusion parameters
    diffusion = Diffusion(config)
    
    # Generate samples using ODE integration
    print("Generating samples...")
    samples_ddpm = generate_samples_with_denoiser(config, diffusion, model, num_samples)
    for i, sample in enumerate(samples_ddpm):
        utils.plot_slice(sample, 0, 1, 63, f"generated_ddpm_sample_{i}")
        
    residual_of_generated(samples_ddpm, samples_gt, config)
    test_wasserstein(samples_ddpm, samples_gt, config)