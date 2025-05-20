import torch
import matplotlib.pyplot as plt
import os
import yaml
import numpy as np
from scipy.stats import wasserstein_distance_nd

from dataset import IsotropicTurbulenceDataset, BigIsotropicTurbulenceDataset
import utils
from model_simple import Model_base
from model_latent import LatentFlowMatchingModel
from model_ae import Autoencoder

# Create a folder to save plots
plot_folder = "generated_plots"
os.makedirs(plot_folder, exist_ok=True)

# Load the trained latent flow matching model
def load_latent_model(config, model_path):
    model = LatentFlowMatchingModel(config)
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model = model.to(config.device)
    model.eval()
    return model

# Load the pre-trained autoencoder (for encoding/decoding)
def load_autoencoder(config, ae_path):
    ae = Autoencoder(config)
    ae.load_state_dict(torch.load(ae_path, map_location=config.device))
    ae = ae.to(config.device)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False
    return ae

# Integrate ODE and generate samples in latent space, then decode
def integrate_ode_and_sample_latent(config, model, ae, num_samples=1, steps=10):
    model.eval().requires_grad_(False)
    ae.eval().requires_grad_(False)
    samples = []
    for _ in range(num_samples):
        # Start with noise in latent space
        latent_shape = ae.latent_shape if hasattr(ae, 'latent_shape') else (config.Model.latent_dim,)
        zt = torch.randn((1, *latent_shape), device=config.device)
        for i, t in enumerate(torch.linspace(0, 1, steps, device=config.device), start=1):
            print(f"Latent ODE Step {i}/{steps}")
            pred = model(zt, t.expand(zt.size(0)))
            zt = zt + (1 / steps) * pred
        # Decode the final latent sample
        with torch.no_grad():
            xt = ae.decode(zt)
        samples.append(xt.cpu().detach())
    return samples

# Linear Beta Schedule (from beta_min to beta_max over the T timesteps)
def get_linear_beta_schedule(T, beta_min=1e-4, beta_max=0.02):
    betas = torch.linspace(beta_min, beta_max, T)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas_cumprod

# Convert velocity to noise epsilon (based on flow matching)
def velocity_to_epsilon(v, x, t, alpha_cum_t):
    alpha_t = alpha_cum_t.sqrt ()
    sigma_t = (1 - alpha_cum_t).sqrt()
    delta = 1e-12
    eps = (v + x / (alpha_t + delta)) / (delta + 1 + sigma_t / (alpha_t + delta))
    return eps
    
# DDIM sampling (using reverse diffusion with flow matching)
def ddim(x, model, t_start, reverse_steps, betas, alphas_cumprod):
    seq = list(range(t_start, 0, -t_start // reverse_steps))
    next_seq = [-1] + seq[:-1]
    n = x.size(0)  # Batch size

    for i, j in zip(reversed(seq), reversed(next_seq)):
        t = torch.full((n,), i / t_start, dtype=torch.float, device=x.device)  # Normalize time to [1, 0]
        t = 1 - t  # Invert to match FM
        print(f"Step {i}/{t_start}, Time: {t[0].item():.4f}")

        alpha_bar_t = alphas_cumprod[i] if i < len(alphas_cumprod) else alphas_cumprod[-1]
        alpha_bar_next = alphas_cumprod[j] if 0 <= j < len(alphas_cumprod) else alpha_bar_t
        
        # Predict velocity v_theta(x_t, t) using the model
        v = model(x, t)

        # Convert velocity to noise epsilon
        e = velocity_to_epsilon(v, x, t, alpha_bar_t)

        # Classic DDIM x0 prediction and update
        x0_pred = (x - e * (1 - alpha_bar_t).sqrt()) / alpha_bar_t.sqrt()
        x = alpha_bar_next.sqrt() * x0_pred + (1 - alpha_bar_next).sqrt() * e

        # Free memory of intermediate tensors
        del v, e, x0_pred
        torch.cuda.empty_cache()

    return x

# Generate samples using the denoising model in latent space, then decode

def generate_samples_with_denoiser_latent(config, model, ae, num_samples, t_start, reverse_steps, T):
    betas, alphas_cumprod = get_linear_beta_schedule(T)
    samples = []
    ae.eval().requires_grad_(False)
    latent_shape = ae.latent_shape if hasattr(ae, 'latent_shape') else (config.Model.latent_dim,)
    for _ in range(num_samples):
        z = torch.randn((1, *latent_shape), device=config.device).float()
        sample_latent = ddim(z, model, t_start, reverse_steps, betas, alphas_cumprod)
        with torch.no_grad():
            sample = ae.decode(sample_latent)
        samples.append(sample.cpu().detach())
    return samples

def residual_of_generated(samples, samples_gt, config):
    rmse_loss = np.zeros(len(samples))
    for i in range(len(samples)):
        print("Batch", i)
        # Ensure all tensors are on the same device
        sample = samples[i].to(config.device)
        res, = utils.compute_divergence(sample)
        rmse_loss[i] = torch.sqrt(torch.mean(res**2))
    
    test_residuals = []
    for i in range(len(samples)):
        sample_gt = samples_gt[i].to(config.device)
        sample_gt = sample_gt.unsqueeze(0)
        res_gt, = utils.compute_divergence(sample_gt)
        test_residuals.append(torch.sqrt(torch.mean(res_gt**2)))
        
    print(f"L2 residual: {np.mean(rmse_loss):.2f} +/- {np.std(rmse_loss):.2f}") 
    print(f"Residual difference: {np.mean(rmse_loss - test_residuals)} +/- {np.std(rmse_loss - test_residuals)}")

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
    with open("configs/config_generative.yml", "r") as f:
        config = yaml.safe_load(f)
    config = utils.dict2namespace(config)
    print(config.device)

    # Load the trained model
    print("Loading model...")
    model = load_latent_model(config, config.Model.lfm_path)
    print("Loading autoencoder...")
    ae = load_autoencoder(config, config.Model.ae_path)

    print("Generating samples (latent FM)...")
    num_samples = 10
    dataset = IsotropicTurbulenceDataset(dt=config.Data.dt, grid_size=config.Data.grid_size, crop=config.Data.crop, seed=config.Data.seed, size=config.Data.size, num_samples=num_samples)
    #dataset = BigIsotropicTurbulenceDataset("/mnt/data4/pbdl-datasets-local/3d_jhtdb/isotropic1024coarse.hdf5", sim_group='sim0', norm=True, size=None, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, batch_size=5, num_samples=num_samples, test=True)
    samples_gt = dataset.test_dataset

    samples_fm = integrate_ode_and_sample_latent(config, model, ae, num_samples=num_samples)
    for i, sample in enumerate(samples_fm):
        utils.plot_slice(sample, 0, 1, 63, f"generated_sample_{i}")

    samples_ddim = generate_samples_with_denoiser_latent(config, model, ae, num_samples, t_start=1000, reverse_steps=20, T=1000)
    for i, sample in enumerate(samples_ddim):
        utils.plot_slice(sample, 0, 1, 63, f"generated_sample_diff_{i}")

    residual_of_generated(samples_fm, samples_gt, config)
    test_wasserstein(samples_fm, samples_gt, config)
    residual_of_generated(samples_ddim, samples_gt, config)
    test_wasserstein(samples_ddim, samples_gt, config)