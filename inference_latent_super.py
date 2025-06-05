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
#from model_ae import Autoencoder
from model_vqvae import VQVAE, VAE, AE

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
    ae = VAE(input_size=config_ae.Model.in_channels,
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

# Flow matching interpolation in latent space

def fm_interp_latent(model, z, z_lr, steps):
    zt = z
    for i, t in enumerate(torch.linspace(0, 1, steps, device=zt.device, dtype=torch.float32), start=1):
        #print(f"Latent Step {i}/{steps}")
        pred = model(zt, t.expand(zt.size(0)))
        pred = pred.sample
        zt = zt + (1 / steps) * (z_lr - zt + t * pred)
        
    return zt
    
def fm_interp_sparse_experiment_latent(config, config_ae, model, ae, nsamples, samples_x, samples_y):
    losses = []
    residuals = []
    residuals_gt = []
    residuals_diff = []
    lsim = []
    for i in range(nsamples):
        print(f"Sample {i+1}/{nsamples}")
        x = samples_x[i].unsqueeze(0).to(config.device)
        y = samples_y[i].unsqueeze(0).to(config.device)
        noise = torch.randn((1, config_ae.Model.in_channels, config_ae.Data.grid_size, config_ae.Data.grid_size, config_ae.Data.grid_size), device=config.device).float()
        # Encode to latent space
        with torch.no_grad():
            mu1, logvar1 = ae.encode(x)
            z_lr = ae.reparameterize(mu1, logvar1)
            mu2, logvar2 = ae.encode(y)
            z_hr = ae.reparameterize(mu2, logvar2)
            mu3, logvar3 = ae.encode(noise)
            z_noise = ae.reparameterize(mu3, logvar3)
        # Flow matching in latent space
        z_pred = fm_interp_latent(model, z_noise, z_lr, steps=10)
        # Decode back to physical space
        with torch.no_grad():
            y_pred = ae.decode(z_pred)
        utils.plot_2d_comparison(x[0, 1, :, :, int(config_ae.Data.grid_size / 2)].cpu().detach().numpy(),
                                 y_pred[0, 1, :, :, int(config_ae.Data.grid_size / 2)].cpu().detach().numpy(),
                                 y[0, 1, :, :, int(config_ae.Data.grid_size / 2)].cpu().detach().numpy(),
                                 f"super_latent_interp_{i}")
        losses.append(torch.sqrt(torch.mean((y_pred - y) ** 2)).item())
        residuals.append(torch.mean(torch.abs(utils.compute_divergence(y_pred[:, :3, :, :, :], 2*math.pi/config.Data.grid_size))).item())
        residuals_gt.append(torch.mean(torch.abs(utils.compute_divergence(y[:, :3, :, :, :], 2*math.pi/config.Data.grid_size))).item())
        residuals_diff.append(abs(residuals[i] - residuals_gt[i]))
        y = y.detach()
        y_pred = y_pred.detach()
        lsim.append(utils.LSiM_distance_3D(y, y_pred))
    print(f"Pixel-wise L2 error: {np.mean(losses):.4f} +/- {np.std(losses):.4f}")
    print(f"Residual L2 norm: {np.mean(residuals):.4f} +/- {np.std(residuals):.4f}")
    print(f"Residual difference: {np.mean(residuals_diff):.4f} +/- {np.std(residuals_diff):.4f}")
    print(f"Mean LSiM: {np.mean(lsim):.4f} +/- {np.std(lsim):.4f}")

def fm_mask(fm_model, ae_model, x, x_lr, steps, mask):
    mu1, logvar1 = ae.encode(x)
    z = ae.reparameterize(mu1, logvar1)
    mu2, logvar2 = ae.encode(x_lr)
    z_lr = ae.reparameterize(mu2, logvar2)

    zt = z                              # Start FM from high-res latent
    for i, t in enumerate(torch.linspace(0, 1, steps, device=z.device, dtype=torch.float32), start=1):
        #print(f"Step {i}/{steps}")
        if t > (1 - 1e-3):
            t = torch.tensor([1 - 1e-3], device=z.device)
        mask_t = mask * (1 - t)

        # Flow Matching model prediction in latent space
        pred = fm_model(zt, t.expand(zt.size(0)))  # predicted vector field
        pred = pred.sample
        z1_pred = zt + (1 - t) * pred              # latent prediction

        # Decode to physical space to apply masking
        x1_pred = ae_model.decoder(z1_pred)

        # Apply mask in physical space
        x_masked = (1 - mask_t) * x1_pred + mask_t * x_lr

        # Re-encode to latent space
        mu, logvar = ae.encode(x_masked)
        z_masked = ae.reparameterize(mu, logvar)

        # Euler update step in latent space
        zt = zt + (1 / steps) * (z_masked - zt) / (1 - t)

    # Final decoding back to physical space
    x_final = ae_model.decoder(zt)
    return x_final


def fm_mask_sparse_experiment_latent(config, config_ae, model, ae, nsamples, samples_x, samples_y, samples_ids, w_mask=1):
    
    losses = []
    residuals = []
    residuals_gt = []
    residuals_diff = []
    lsim = []
    
    for i in range(nsamples):
        print(f"Sample {i+1}/{nsamples}")
        x     = samples_x[i].unsqueeze(0).to(config.device)
        y     = samples_y[i].unsqueeze(0).to(config.device)
        noise = torch.randn((1, config_ae.Model.in_channels, config_ae.Data.grid_size, config_ae.Data.grid_size, config_ae.Data.grid_size), device=config.device).float()

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

        y_pred = fm_mask(model, ae, noise.clone(), x.clone(), 10, mask)
        utils.plot_2d_comparison(x[0, 1, :, :, int(config_ae.Data.grid_size / 2)].cpu().detach().numpy(),
                                 y_pred[0, 1, :, :, int(config_ae.Data.grid_size / 2)].cpu().detach().numpy(),
                                 y[0, 1, :, :, int(config_ae.Data.grid_size / 2)].cpu().detach().numpy(),
                                 f"super_latent_mask_{i}")

        losses.append(torch.sqrt(torch.mean((y_pred - y) ** 2)).item())
        residuals.append(torch.mean(torch.abs(utils.compute_divergence(y_pred[:, :3, :, :, :], 2*math.pi/config.Data.grid_size))).item())
        residuals_gt.append(torch.mean(torch.abs(utils.compute_divergence(y[:, :3, :, :, :], 2*math.pi/config.Data.grid_size))).item())
        residuals_diff.append(abs(residuals[i] - residuals_gt[i]))
        # Detach tensors before passing them to LSiM_distance
        y = y.detach()
        y_pred = y_pred.detach()
        lsim.append(utils.LSiM_distance_3D(y, y_pred))
        
    print(f"Pixel-wise L2 error: {np.mean(losses):.4f} +/- {np.std(losses):.4f}")
    print(f"Residual L2 norm: {np.mean(residuals):.4f} +/- {np.std(residuals):.4f}") 
    print(f"Residual difference: {np.mean(residuals_diff):.4f} +/- {np.std(residuals_diff):.4f}")
    print(f"Mean LSiM: {np.mean(lsim):.4f} +/- {np.std(lsim):.4f}")
    
def fm_diff_mask_sparse_experiment_latent(config, config_ae, model, ae, nsamples, samples_x, samples_y, samples_ids, w_mask=1, sig=0.044):
    
    losses = []
    residuals = []
    residuals_gt = []
    residuals_diff = []
    lsim = []
    
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
        for i in range(nsamples):
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
        noise = torch.randn((1, config_ae.Model.in_channels, config_ae.Data.grid_size, config_ae.Data.grid_size, config_ae.Data.grid_size), device=config.device).float()

        y_pred = fm_mask(model, ae, noise.clone(), x.clone(), 10, diffuse_masks[i].unsqueeze(0))
        utils.plot_2d_comparison(x[0, 1, :, :, int(config_ae.Data.grid_size / 2)].cpu().detach().numpy(),
                                 y_pred[0, 1, :, :, int(config_ae.Data.grid_size / 2)].cpu().detach().numpy(),
                                 y[0, 1, :, :, int(config_ae.Data.grid_size / 2)].cpu().detach().numpy(),
                                 f"super_latent_diff_mask_{i}")

        losses.append(torch.sqrt(torch.mean((y_pred - y) ** 2)).item())
        residuals.append(torch.mean(torch.abs(utils.compute_divergence(y_pred[:, :3, :, :, :], 2*math.pi/config.Data.grid_size))).item())
        residuals_gt.append(torch.mean(torch.abs(utils.compute_divergence(y[:, :3, :, :, :], 2*math.pi/config.Data.grid_size))).item())
        residuals_diff.append(abs(residuals[i] - residuals_gt[i]))
        # Detach tensors before passing them to LSiM_distance
        y = y.detach()
        y_pred = y_pred.detach()
        lsim.append(utils.LSiM_distance_3D(y, y_pred))
        
    print(f"Pixel-wise L2 error: {np.mean(losses):.4f} +/- {np.std(losses):.4f}")
    print(f"Residual L2 norm: {np.mean(residuals):.4f} +/- {np.std(residuals):.4f}") 
    print(f"Residual difference: {np.mean(residuals_diff):.4f} +/- {np.std(residuals_diff):.4f}")
    print(f"Mean LSiM: {np.mean(lsim):.4f} +/- {np.std(lsim):.4f}")


# Main script
if __name__ == "__main__":
    # Load the configuration
    print("Loading config...")
    with open("configs/config_lfm.yml", "r") as f:
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
    
    print("Loading dataset...")
    num_samples = 10
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
    samples_y = dataset.test_dataset
    perc = 5
    samples_x, samples_ids = utils.interpolate_dataset(samples_y, perc/100)
    #samples_x = utils.downscale_data(samples_y, 4)
    #samples_ids = None
    
    print("Generating samples (latent FM)...")
    fm_interp_sparse_experiment_latent(config, config_ae, model, ae, num_samples, samples_x, samples_y)
    fm_mask_sparse_experiment_latent(config, config_ae, model, ae, num_samples, samples_x, samples_y, samples_ids)
    fm_diff_mask_sparse_experiment_latent(config, config_ae, model, ae, num_samples, samples_x, samples_y, samples_ids)