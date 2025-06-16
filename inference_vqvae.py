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
from model_vqvae import VQVAE, VAE, AE

# Create a folder to save plots
plot_folder = "generated_plots"
os.makedirs(plot_folder, exist_ok=True)

# Load the trained model
def load_model(config, model_path):
    #model = VQVAE(input_size=config.Model.in_channels, hidden_size=config.Model.hidden_size, depth=config.Model.depth, num_res_block=config.Model.num_res_block, res_size=config.Model.res_size, embedding_size=config.Model.embedding_size,
    #             num_embedding=config.Model.num_embedding, device=config.device).to(config.device)
    #model = AE(input_size=config.Model.in_channels,
    #           image_size=config.Data.grid_size,
    #           hidden_size=config.Model.hidden_size,
    #           depth=config.Model.depth,
    #           num_res_block=config.Model.num_res_block,
    #           res_size=config.Model.res_size,
    #           device=config.device,
    #           z_dim=config.Model.z_dim).to(config.device)
    model = AE(input_size=config.Model.in_channels,
               image_size=config.Data.grid_size,
               hidden_size=config.Model.hidden_size,
               depth=config.Model.depth,
               num_res_block=config.Model.num_res_block,
               res_size=config.Model.res_size,
               device=config.device,
               z_dim=config.Model.z_dim).to(config.device)
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model = model.to(config.device)
    model.eval()
    return model

def show_recons(model, samples_x):
    rmse_list = []
    for i in range(samples_x.shape[0]):
        sample = samples_x[i]
        sample = sample.unsqueeze(0)
        sample = sample.to(config.device)
        input = {'uvw': sample, 'duvw': utils.spectral_derivative_3d(sample)}
        output = model(input)
        output = output['uvw']

        # Compute RMSE for the current sample
        rmse = torch.sqrt(torch.mean((sample - output) ** 2)).item()
        rmse_list.append(rmse)

        utils.plot_slice(sample.cpu().detach().numpy(), 0, 0, 62, name=f"vqvae_input_{i}")
        utils.plot_slice(output.cpu().detach().numpy(), 0, 0, 62, name=f"vqvae_output_{i}")

    # Calculate average and standard deviation of RMSE
    avg_rmse = np.mean(rmse_list)
    std_rmse = np.std(rmse_list)
    print(f"Average RMSE: {avg_rmse:.4f}, Standard Deviation: {std_rmse:.4f}")
    
def generate_samples(model, config, num_samples):
    latent_dim = int(config.Data.grid_size / 4)  # Ensure latent_dim is an integer
    samples = []
    with torch.no_grad():
        for _ in tqdm(range(num_samples), desc="Generating samples"):
            # Generate a random input tensor
            input_tensor = torch.randn(1, config.Model.z_dim, latent_dim, latent_dim, latent_dim).to(config.device)
            output = model.decode(input_tensor)
            samples.append(output.cpu().detach())
    
    return samples

def generate_samples_vae(model, config, num_samples):
    latent_dim = int(config.Data.grid_size / 4)  # Ensure latent_dim is an integer
    samples = []
    for _ in range(num_samples):
        # Generate random latent vector
        mu = torch.zeros(1, config.Model.z_dim, latent_dim, latent_dim, latent_dim).to(config.device)
        logvar = torch.zeros(1, config.Model.z_dim, latent_dim, latent_dim, latent_dim).to(config.device)
        z = model.reparameterize(mu, logvar)  # Sample z using reparameterization trick
        output = model.decode(z)  # Decode z to generate a sample
        samples.append(output.cpu().detach())
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

# Main script
if __name__ == "__main__":
    print("Loading config...")
    with open("configs/config_vqvae.yml", "r") as f:
        config = yaml.safe_load(f)
    config = utils.dict2namespace(config)
    print(config.device)
    
    print("Loading dataset...")
    num_samples = 50
    #dataset = IsotropicTurbulenceDataset(dt=config.Data.dt, grid_size=config.Data.grid_size, crop=config.Data.crop, seed=config.Data.seed, size=config.Data.size, num_samples=num_samples)
    #dataset = BigIsotropicTurbulenceDataset("/mnt/data4/pbdl-datasets-local/3d_jhtdb/isotropic1024coarse.hdf5", sim_group='sim0', norm=True, size=None, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, batch_size=5, num_samples=num_samples, test=True, grid_size=config.Data.grid_size)
    dataset = BigSpectralIsotropicTurbulenceDataset(grid_size=config.Data.grid_size,
                                                    norm=config.Data.norm,
                                                    size=config.Data.size,
                                                    train_ratio=0.8,
                                                    val_ratio=0.1,
                                                    test_ratio=0.1,
                                                    batch_size=config.Training.batch_size,
                                                    num_samples=num_samples)
    samples_x = dataset.test_dataset

    print("Loading model...")
    model = load_model(config, config.Model.ae_path)
    
    print("Calculating reconstruction...")
    show_recons(model, samples_x)

    # Data points for compression ratio and corresponding errors for AE and VAE
    compression_ratios = [3, 6, 12]  
    #avg_rmse_ae = [, , 0.1235]  
    #std_rmse_ae = [, , 0.0085]  
    #avg_rmse_vae = [, , 0.1247]
    #std_rmse_vae = [, , 0.0085] 

    # Plotting the error vs compression ratio for both AE and VAE
    #plt.figure(figsize=(8, 6))
    #plt.errorbar(compression_ratios, avg_rmse_ae, yerr=std_rmse_ae, fmt='o-', capsize=5, label='AE')
    #plt.errorbar(compression_ratios, avg_rmse_vae, yerr=std_rmse_vae, fmt='o-', capsize=5, label='VAE')
    #plt.xlabel('Compression Ratio')
    #plt.ylabel('Reconstruction Error (RMSE)')
    #plt.title('Error vs Compression Ratio')
    #plt.grid(True)
    #plt.legend()
    #plt.savefig(os.path.join(plot_folder, 'error_ae.png'))
    #plt.show()

    print("Generating samples...")
    samples_ae = generate_samples(model, config, num_samples)
    #samples_ae = generate_samples_vae(model, config, num_samples)
    for i, sample in enumerate(samples_ae):
        utils.plot_slice(sample, 0, 1, 63, f"generated_sample_ae_{i}")

    residual_of_generated(dataset, samples_ae, samples_x, config)
    test_wasserstein(samples_ae, samples_x, config)
    test_blurriness(samples_ae, samples_x, config)
    test_energy_spectrum(samples_ae, samples_x, config)