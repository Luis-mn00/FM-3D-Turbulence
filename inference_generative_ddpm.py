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

# Generate samples using the denoising model
def generate_samples_with_denoiser(config, diffusion, model, num_samples):
    samples = []
    for _ in range(num_samples):
        noise = torch.randn((1, config.Model.in_channels, config.Data.grid_size, config.Data.grid_size, config.Data.grid_size), device=config.device)
        y_pred, _ = diffusion.ddpm(noise, model, 1000, plot_prog=False)
        samples.append(y_pred)
        
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
    with open("configs/config_ddpm.yml", "r") as f:
        config = yaml.safe_load(f)
    config = utils.dict2namespace(config)
    print(config.device)

    num_samples = 10
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