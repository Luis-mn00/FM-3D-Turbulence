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

from dataset import IsotropicTurbulenceDataset, BigIsotropicTurbulenceDataset, BigSpectralIsotropicTurbulenceDataset, SupervisedSpectralTurbulenceDataset
import utils
from src.core.models.box.pdedit import PDEDiT3D_S, PDEDiT3D_B, PDEDiT3D_L
from my_config_length import UniProjectionLength

# Create a folder to save plots
plot_folder = "generated_plots"
os.makedirs(plot_folder, exist_ok=True)

def integrate_ode_and_sample(config, model, x_lr, steps=10):
    model.eval().requires_grad_(False)

    with torch.no_grad():
        xt = x_lr.to(config.device).float()
        for i, t in enumerate(torch.linspace(0, 1, steps, device=config.device), start=1):
            #print(f"Step {i}/{steps}")
            pred = model(xt, t.expand(xt.size(0)))
            pred = pred.sample
            xt = xt + (1 / steps) * pred
            
            # Free memory of intermediate tensors
            del pred
            torch.cuda.empty_cache()

    return xt

def fm_sparse_experiment(dataset, config, model, nsamples, samples_x, samples_y):
    
    losses = []
    residuals = []
    residuals_gt = []
    residuals_diff = []
    lsim = []
    blurriness = []
    spectrum = []
    
    for i in range(nsamples):
        print(f"Sample {i+1}/{nsamples}")
        x     = samples_x[i].unsqueeze(0).to(config.device)
        y     = samples_y[i].unsqueeze(0).to(config.device)

        y_pred = integrate_ode_and_sample(config, model, x, steps=100)
        utils.plot_2d_comparison(x[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
                                 y_pred[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
                                 y[0, 1, :, :, int(config.Data.grid_size / 2)].cpu().detach().numpy(),
                                 f"super_direct_route_{i}")

        losses.append(torch.sqrt(torch.mean((y_pred - y) ** 2)).item())
        residuals.append(torch.mean(torch.abs(utils.compute_divergence(dataset.Y_scaler.inverse(y_pred[:, :3, :, :, :].to("cpu")), 2*math.pi/config.Data.grid_size))).item())
        residuals_gt.append(torch.mean(torch.abs(utils.compute_divergence(dataset.Y_scaler.inverse(y[:, :3, :, :, :].to("cpu")), 2*math.pi/config.Data.grid_size))).item())
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
        
        y = y.unsqueeze(0)
        y_pred = y_pred.unsqueeze(0)
        e_gt = utils.compute_energy_spectrum(y, "energy_gt")
        e_pred = utils.compute_energy_spectrum(y_pred, "energy_pred")
        diff = np.abs(e_gt - e_pred)
        diff = np.mean(diff)
        spectrum.append(diff)
        
    print(f"Pixel-wise L2 error: {np.mean(losses):.4f} +/- {np.std(losses):.4f} (max: {np.max(losses):.4f})")
    print(f"Residual L2 norm: {np.mean(residuals):.4f} +/- {np.std(residuals):.4f} (max: {np.max(residuals):.4f})") 
    print(f"Residual difference: {np.mean(residuals_diff):.4f} +/- {np.std(residuals_diff):.4f} (max: {np.max(residuals_diff):.4f})")
    print(f"Mean LSiM: {np.mean(lsim):.4f} +/- {np.std(lsim):.4f} (max: {np.max(lsim):.4f})")
    print(f"Mean blurriness: {np.mean(blurriness):.4f} +/- {np.std(blurriness):.4f} (max: {np.max(blurriness):.4f})")
    print(f"Mean energy spectrum difference: {np.mean(spectrum):.4e} +/- {np.std(spectrum):.4e} (max: {np.max(spectrum):.4e})")

# Main script
if __name__ == "__main__":
    print("Loading config...")
    with open("configs/config_direct_route.yml", "r") as f:
        config = yaml.safe_load(f)
    config = utils.dict2namespace(config)
    print(config.device)
    
    # Generate samples using ODE integration
    num_samples = 50
    #dataset = IsotropicTurbulenceDataset(dt=config.Data.dt, grid_size=config.Data.grid_size, crop=config.Data.crop, seed=config.Data.seed, size=config.Data.size, batch_size=config.Training.batch_size, num_samples=num_samples, field=None)
    dataset = SupervisedSpectralTurbulenceDataset(grid_size=config.Data.grid_size,
                                                    norm=config.Data.norm,
                                                    size=config.Data.size,
                                                    train_ratio=0.8,
                                                    val_ratio=0.1,
                                                    test_ratio=0.1,
                                                    batch_size=config.Training.batch_size,
                                                    num_samples=num_samples)
    
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
    
    samples_x, samples_y = dataset.test_dataset
    print(samples_y.shape)
    print(samples_x.shape)
    
    fm_sparse_experiment(dataset, config, model, num_samples, samples_x, samples_y)

