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
from model_vqvae import VQVAE, VAE, AE
from diffusion import Diffusion

# Create a folder to save plots
plot_folder = "generated_plots"
os.makedirs(plot_folder, exist_ok=True)

# Load the trained latent flow matching model
def load_latent_model(config, model_path):
    model = PDEDiT3D_B(
        channel_size=config.Model.channel_size,
        channel_size_out=config.Model.channel_size_out,
        drop_class_labels=config.Model.drop_class_labels,
        partition_size=config.Model.partition_size,
        mending=False
    )
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model = model.to(config.device)
    model.eval()
    return model

def load_ae_model(config):
    ae = AE(input_size=config_ae.Model.in_channels,
               image_size=config_ae.Data.grid_size,
               hidden_size=config_ae.Model.hidden_size,
               depth=config_ae.Model.depth,
               num_res_block=config_ae.Model.num_res_block,
               res_size=config_ae.Model.res_size,
               device=config.device,
               z_dim=config_ae.Model.z_dim).to(config.device)
    ae.load_state_dict(torch.load(config.Model.ae_path, map_location=config.device))
    ae = ae.to(config.device)
    ae.eval()
    return ae

# Generate samples using the denoising model
def generate_samples_with_denoiser_latent(config, diffusion, config_ae, model, ae, num_samples, t_start=1000, reverse_steps=20, T=1000):
    samples = []
    for _ in range(num_samples):
        print(f"Generating sample {_+1}/{num_samples}")
        # Ensure the input tensor is in float format to match the model's parameters
        x = torch.randn((1, config_ae.Model.in_channels, config_ae.Data.grid_size, config_ae.Data.grid_size, config_ae.Data.grid_size), device=config.device).float()
        z = ae.encode(x)
        #mu, logvar = ae.encode(x)
        #z = ae.reparameterize(mu, logvar)
        
        # Perform denoising using DDIM
        sample = diffusion.ddim(z, model, t_start, reverse_steps, plot_prog=False)
        sample = ae.decode(sample)  # Decode back to original space
        samples.append(sample.cpu().detach())
        
    return samples

def residual_of_generated(dataset, samples, samples_gt, config):
    rmse_loss = np.zeros(len(samples))
    for i in range(len(samples)):
        # Ensure all tensors are on the same device
        sample = samples[i].to(config.device)
        res, = utils.compute_divergence(dataset.data_scaler.inverse(sample[:, :3, :, :, :].to("cpu")), 2*math.pi/config.Data.grid_size)
        rmse_loss[i] = torch.mean(torch.abs(res))
    
    test_residuals = []
    for i in range(len(samples)):
        sample_gt = samples_gt[i].to(config.device)
        sample_gt = sample_gt.unsqueeze(0)
        res_gt, = utils.compute_divergence(dataset.data_scaler.inverse(sample_gt[:, :3, :, :, :].to("cpu")), 2*math.pi/config.Data.grid_size)
        test_residuals.append(torch.mean(torch.abs(res_gt)))
        
    print(f"L2 residual: {np.mean(rmse_loss):.4f} +/- {np.std(rmse_loss):.4f}") 
    # Ensure test_residuals is a numpy array on CPU
    test_residuals_np = np.array([r.cpu().item() if torch.is_tensor(r) else r for r in test_residuals])
    print(f"Residual difference: {np.mean(rmse_loss - test_residuals_np):.4f} +/- {np.std(rmse_loss - test_residuals_np):.4f}")

    # Compute L2 norm of the difference between samples and ground truth
    l2_diff_norms = []
    for i in range(len(samples)):
        y = samples_gt[i]  # Ground truth sample
        y_pred = samples[i]  # Retrieve saved y_pred
        l2_diff_norm = torch.sqrt(torch.mean((y - y_pred) ** 2)).item()
        l2_diff_norms.append(l2_diff_norm)

    print(f"Mean L2 difference between generated samples and ground truth: {np.mean(l2_diff_norms):.4f} +/- {np.std(l2_diff_norms):.4f}")

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
    print(f"Sharpness: {mean_blurriness:.4f} +/- {std_blurriness:.4f}")
    
def test_energy_spectrum(samples, samples_gt, config):
    e_gt = utils.compute_energy_spectrum(samples_gt, f"energy_gt")
    
    # Ensure samples are converted to a tensor before passing to compute_energy_spectrum
    samples_tensor = torch.stack([s.squeeze(0) for s in samples])
    e_fm = utils.compute_energy_spectrum(samples_tensor, f"energy_fm")
    
    # Convert e_gt and e_fm to tensors before applying torch.abs
    e_gt_tensor = torch.tensor(e_gt, device=config.device)
    e_fm_tensor = torch.tensor(e_fm, device=config.device)

    diff = torch.abs(e_gt_tensor - e_fm_tensor)
    print(f"Energy spectrum difference: {torch.mean(diff):.4e} +/- {torch.std(diff):.4e}")


if __name__ == "__main__":
    # Load the configuration
    print("Loading config...")
    with open("configs/config_ldiff.yml", "r") as f:
        config = yaml.safe_load(f)
    config = utils.dict2namespace(config)
    print(config.device)
    
    print("Loading config...")
    with open("configs/config_vqvae.yml", "r") as f:
        config_ae = yaml.safe_load(f)
    config_ae = utils.dict2namespace(config_ae)

    # Load the trained model
    print("Loading model...")
    model = load_latent_model(config, config.Model.save_path)
    print("Loading autoencoder...")
    ae = load_ae_model(config_ae)
    
    # Diffusion parameters
    diffusion = Diffusion(config)

    print("Generating samples (latent DDPM)...")
    num_samples = 50
    #dataset = IsotropicTurbulenceDataset(dt=config.Data.dt, grid_size=config_ae.Data.grid_size, crop=config.Data.crop, seed=config.Data.seed, size=config.Data.size, num_samples=num_samples)
    #dataset = BigIsotropicTurbulenceDataset("/mnt/data4/pbdl-datasets-local/3d_jhtdb/isotropic1024coarse.hdf5", sim_group='sim0', norm=True, size=None, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, batch_size=5, num_samples=num_samples, test=True, grid_size=config.Data.grid_size)
    dataset = BigSpectralIsotropicTurbulenceDataset(grid_size=config_ae.Data.grid_size,
                                                    norm=config_ae.Data.norm,
                                                    size=config.Data.size,
                                                    train_ratio=0.8,
                                                    val_ratio=0.1,
                                                    test_ratio=0.1,
                                                    batch_size=config.Training.batch_size,
                                                    num_samples=num_samples)
    samples_gt = dataset.test_dataset

    samples_ddim = generate_samples_with_denoiser_latent(config, diffusion, config_ae, model, ae, num_samples, reverse_steps=100)
    for i, sample in enumerate(samples_ddim):
        utils.plot_slice(sample, 0, 1, 63, f"generated_latent_sample_ddpm_{i}")

    residual_of_generated(dataset, samples_ddim, samples_gt, config)
    test_wasserstein(samples_ddim, samples_gt, config)
    test_blurriness(samples_ddim, samples_gt, config)
    test_energy_spectrum(samples_ddim, samples_gt, config)
