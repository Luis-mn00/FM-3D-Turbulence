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

# Create a folder to save plots
plot_folder = "generated_plots"
os.makedirs(plot_folder, exist_ok=True)

# Integrate ODE and generate samples
def integrate_ode_and_sample(config, model, num_samples=1, steps=100):
    model.eval().requires_grad_(False)

    samples = []
    with torch.no_grad():
        for _ in range(num_samples):
            print(f"Generating sample {_+1}/{num_samples}")
            # Initialize random sample
            xt = torch.randn((1, config.Model.channel_size, config.Data.grid_size, config.Data.grid_size, config.Data.grid_size), device=config.device)

            for i, t in enumerate(torch.linspace(0, 1, steps, device=config.device), start=1):
                #print(f"Step {i}/{steps}")
                # Predict the flow
                pred = model(xt, t.expand(xt.size(0)))
                pred = pred.sample

                # Update xt using the ODE integration step
                xt = xt + (1 / steps) * pred

            # Only store the final generated sample
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
    #seq = range(0, t_start, t_start // reverse_steps) 
    #next_seq = [-1] + list(seq[:-1])
    seq = list(range(t_start, 0, -t_start // reverse_steps))
    next_seq = [-1] + (seq[:-1])
    n = x.size(0)

    with torch.no_grad():
        for i, j in zip(reversed(seq), reversed(next_seq)):
            #t = (torch.ones(n) * i).to(x.device)
            t = torch.full((n,), i / t_start, dtype=torch.float, device=x.device)  # Normalize time to [1, 0]
            #print(f"Step {i}/{t_start}, Time: {t[0].item():.4f}")

            alpha_bar_t = alphas_cumprod[i] if i < len(alphas_cumprod) else alphas_cumprod[-1]
            alpha_bar_next = alphas_cumprod[j] if 0 <= j < len(alphas_cumprod) else alpha_bar_t
            
            # Predict velocity v_theta(x_t, t) using the model
            #v = model(x, 1 - t / t_start)
            v = model(x, t)
            v = v.sample

            # Convert velocity to noise epsilon
            e = velocity_to_epsilon(v, x, t, alpha_bar_t)

            # Classic DDIM x0 prediction and update
            x0_pred = (x - e * (1 - alpha_bar_t).sqrt()) / alpha_bar_t.sqrt()
            x = alpha_bar_next.sqrt() * x0_pred + (1 - alpha_bar_next).sqrt() * e

            # Free memory of intermediate tensors
            del v, e, x0_pred
            torch.cuda.empty_cache()

    return x

# Generate samples using the denoising model
def generate_samples_with_denoiser(config, model, num_samples, t_start, reverse_steps, T):
    # Get the linear beta schedule
    betas, alphas_cumprod = get_linear_beta_schedule(T)

    samples = []
    for _ in range(num_samples):
        print(f"Generating sample {_+1}/{num_samples}")
        # Ensure the input tensor is in float format to match the model's parameters
        x = torch.randn((1, config.Model.channel_size, config.Data.grid_size, config.Data.grid_size, config.Data.grid_size), device=config.device).float()
        
        # Perform denoising using DDIM
        sample = ddim(x, model, t_start, reverse_steps, betas, alphas_cumprod)
        samples.append(sample.cpu().detach())
        
    return samples

def residual_of_generated(dataset, samples, samples_gt, config):
    rmse_loss = np.zeros(len(samples))
    for i in range(len(samples)):
        # Ensure all tensors are on the same device
        sample = samples[i].to(config.device)
        res = utils.compute_divergence(dataset.data_scaler.inverse(sample[:, :3, :, :, :].to("cpu")), 2*math.pi/config.Data.grid_size)
        rmse_loss[i] = torch.mean(torch.abs(res))
        #rmse_loss[i] = torch.sqrt(torch.sum(res**2))
    
    test_residuals = []
    for i in range(len(samples)):
        sample_gt = samples_gt[i].to(config.device)
        sample_gt = sample_gt.unsqueeze(0)
        res_gt = utils.compute_divergence(dataset.data_scaler.inverse(sample_gt[:, :3, :, :, :].to("cpu")), 2*math.pi/config.Data.grid_size)
        test_residuals.append(torch.mean(torch.abs(res_gt)))
        
    print(f"L2 residual: {np.mean(rmse_loss):.4f} +/- {np.std(rmse_loss):.4f} (max: {np.max(rmse_loss):.4f})") 
    # Ensure test_residuals is a numpy array on CPU
    test_residuals_np = np.array([r.cpu().item() if torch.is_tensor(r) else r for r in test_residuals])
    print(f"Residual difference: {np.mean(rmse_loss - test_residuals_np):.4f} +/- {np.std(rmse_loss - test_residuals_np):.4f} (max: {np.max(rmse_loss - test_residuals_np):.4f})")

    # Compute L2 norm of the difference between samples and ground truth
    l2_diff_norms = []
    for i in range(len(samples)):
        y = samples_gt[i]  # Ground truth sample
        y_pred = samples[i]  # Retrieve saved y_pred
        l2_diff_norm = torch.sqrt(torch.mean((y - y_pred) ** 2)).item()
        l2_diff_norms.append(l2_diff_norm)

    print(f"Mean L2 difference between generated samples and ground truth: {np.mean(l2_diff_norms):.4f} +/- {np.std(l2_diff_norms):.4f} (max: {np.max(l2_diff_norms):.4f})")

def test_wasserstein(samples, samples_gt, config):
    wasserstein_cmf_distances = []
    for i in range(len(samples)):
        y = samples_gt[i]  # Ground truth sample: (C, D, D, D)
        y_pred = samples[i].squeeze(0)  # Prediction: (C, D, D, D)
        
        wasserstein_cmf_distances.append(utils.wasserstein(y, y_pred))

    # Mean and std
    mean_wasserstein = np.mean(wasserstein_cmf_distances)
    std_wasserstein = np.std(wasserstein_cmf_distances)
    print(f"Wasserstein distance: {mean_wasserstein:.4f} +/- {std_wasserstein:.4f} (max: {np.max(wasserstein_cmf_distances):.4f})")
    
def test_blurriness(samples, samples_gt, config):
    blurriness = []
    for i in range(len(samples)):
        y = samples_gt[i]  # Ground truth sample: (C, D, D, D)
        y_pred = samples[i].squeeze(0)  # Prediction: (C, D, D, D)
        
        # Compute blurriness using Laplacian variance
        blurr_pred = utils.compute_blurriness(y_pred.cpu().numpy())
        blurr_gt = utils.compute_blurriness(y.cpu().numpy())
        blurriness.append(abs(blurr_pred - blurr_gt))

    mean_blurriness = np.mean(blurriness)
    std_blurriness = np.std(blurriness)
    print(f"Sharpness: {mean_blurriness:.4f} +/- {std_blurriness:.4f} (max: {np.max(blurriness):.4f})")
    
def test_energy_spectrum(samples, samples_gt, config):
    e_gt = utils.compute_energy_spectrum(samples_gt, f"energy_gt")
    
    # Ensure samples are converted to a tensor before passing to compute_energy_spectrum
    samples_tensor = torch.stack([s.squeeze(0) for s in samples])
    e_fm = utils.compute_energy_spectrum(samples_tensor, f"energy_fm")
    
    # Convert e_gt and e_fm to tensors before applying torch.abs
    e_gt_tensor = torch.tensor(e_gt, device=config.device)
    e_fm_tensor = torch.tensor(e_fm, device=config.device)

    diff = torch.abs(e_gt_tensor - e_fm_tensor)
    print(f"Energy spectrum difference: {torch.mean(diff):.4e} +/- {torch.std(diff):.4e} (max: {torch.max(diff):.4e})")

if __name__ == "__main__":
    # Load the configuration
    print("Loading config...")
    with open("configs/config_fm.yml", "r") as f:
        config = yaml.safe_load(f)
    config = utils.dict2namespace(config)
    print(config.device)

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
    for i in range(samples_gt.shape[0]):
        utils.plot_slice(samples_gt, i, 1, 63, f"gt_sample_{i}")
    
    print("Generating samples...")
    #samples_fm = integrate_ode_and_sample(config, model, num_samples=num_samples, steps=100)
    #for i, sample in enumerate(samples_fm):
    #    utils.plot_slice(sample, 0, 1, 63, f"generated_sample_{i}")
        
    # Generate samples using the denoising model
    samples_ddim = generate_samples_with_denoiser(config, model, num_samples, t_start=1000, reverse_steps=50, T=1000)
    for i, sample in enumerate(samples_ddim):
        utils.plot_slice(sample, 0, 1, 63, f"generated_sample_diff_{i}")
        
    #residual_of_generated(dataset, samples_fm, samples_gt, config)
    #test_wasserstein(samples_fm, samples_gt, config)
    #test_blurriness(samples_fm, samples_gt, config)
    #test_energy_spectrum(samples_fm, samples_gt, config)
    residual_of_generated(dataset, samples_ddim, samples_gt, config)
    test_wasserstein(samples_ddim, samples_gt, config)
    test_blurriness(samples_ddim, samples_gt, config)
    test_energy_spectrum(samples_ddim, samples_gt, config)