import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import wandb
from conflictfree.utils import get_gradient_vector
from conflictfree.grad_operator import ConFIGOperator
import math

from dataset import IsotropicTurbulenceDataset, BigIsotropicTurbulenceDataset, BigSpectralIsotropicTurbulenceDataset
import utils
from my_config_length import UniProjectionLength
from src.core.models.box.pdedit import PDEDiT3D_S, PDEDiT3D_B, PDEDiT3D_L
from model_vqvae import VQVAE, VAE, AE
from diffusion import Diffusion

wandb.login(key="f4a726b2fe7929990149e82fb88da423cfa74e46")

wandb.init(project="Latent Diff")

def ddpm_standard_step(dataset, model, diffusion, y, optimizer, config):
    batch_size = y.shape[0]
    t = torch.randint(0, diffusion.num_timesteps, size=(batch_size,), device=y.device)

    x_t, noise = diffusion.forward(y, t)
    e_pred = model(x_t, t)
    e_pred = e_pred.sample
    mse_loss = (noise - e_pred).square().mean()
    
    total_loss = mse_loss
    total_loss = total_loss
    total_loss.backward() 

    optimizer.step()
    optimizer.zero_grad()

    return total_loss

# Define the training function
def train_flow_matching(config, config_ae):
    # Load the dataset
    print("Loading dataset...")
    #dataset = IsotropicTurbulenceDataset(dt=config_ae.Data.dt, grid_size=config_ae.Data.grid_size, crop=config_ae.Data.crop, seed=config_ae.Data.seed, size=config_ae.Data.size, batch_size=config.Training.batch_size)
    #dataset = BigIsotropicTurbulenceDataset("/mnt/data4/pbdl-datasets-local/3d_jhtdb/isotropic1024coarse.hdf5", sim_group='sim0', norm=True, size=None, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, batch_size=config.Training.batch_size, grid_size=config.Data.grid_size)
    dataset = BigSpectralIsotropicTurbulenceDataset(grid_size=config_ae.Data.grid_size,
                                                    norm=config_ae.Data.norm,
                                                    size=config.Data.size,
                                                    train_ratio=0.8,
                                                    val_ratio=0.1,
                                                    test_ratio=0.1,
                                                    batch_size=config.Training.batch_size,
                                                    num_samples=10)
    
    # Update the dataloaders
    train_loader = dataset.train_loader
    val_loader = dataset.val_loader
    test_loader = dataset.test_loader

    # Initialize the model
    model = PDEDiT3D_B(
        channel_size=config.Model.channel_size,
        channel_size_out=config.Model.channel_size_out,
        drop_class_labels=config.Model.drop_class_labels,
        partition_size=config.Model.partition_size,
        mending=False
    )
    model = model.to(config.device)

    # Convert learning_rate and divergence_loss_weight to float if they are strings
    if isinstance(config.Training.learning_rate, str):
        config.Training.learning_rate = float(config.Training.learning_rate)
    if isinstance(config.Training.gamma, str):
        config.Training.gamma = float(config.Training.gamma)
    if isinstance(config.Training.last_lr, str):
        config.Training.last_lr = float(config.Training.last_lr)

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.Training.learning_rate)
    optimizer.zero_grad()
    
    # Diffusion parameters
    diffusion = Diffusion(config)

    # Find the next run directory
    runs_dir = "runs"
    existing = [d for d in os.listdir(runs_dir) if d.isdigit()]
    if existing:
        next_run = f"{max([int(d) for d in existing])+1:03d}"
    else:
        next_run = "001"
    run_dir = os.path.join(runs_dir, next_run)
    os.makedirs(run_dir, exist_ok=True)

    # Load the pre-trained autoencoder
    #model = VQVAE(input_size=config.Model.in_channels, hidden_size=config.Model.hidden_size, depth=config.Model.depth, num_res_block=config.Model.num_res_block, res_size=config.Model.res_size, embedding_size=config.Model.embedding_size,
    #             num_embedding=config.Model.num_embedding, device=config.device).to(config.device)
    ae = AE(input_size=config_ae.Model.in_channels,
               image_size=config_ae.Data.grid_size,
               hidden_size=config_ae.Model.hidden_size,
               depth=config_ae.Model.depth,
               num_res_block=config_ae.Model.num_res_block,
               res_size=config_ae.Model.res_size,
               device=config.device,
               z_dim=config_ae.Model.z_dim).to(config.device)
    #ae = VAE(input_size=config_ae.Model.in_channels,
    #           image_size=config_ae.Data.grid_size,
    #           hidden_size=config_ae.Model.hidden_size,
    #           depth=config_ae.Model.depth,
    #           num_res_block=config_ae.Model.num_res_block,
    #           res_size=config_ae.Model.res_size,
    #           device=config.device,
    #           z_dim=config_ae.Model.z_dim).to(config.device)
    ae.load_state_dict(torch.load(config_ae.Model.ae_path, map_location=config.device))
    ae.eval()
    for param in ae.parameters():
        param.requires_grad = False

    # Training loop with validation loss
    print("Starting training...")
    mse_losses = []
    val_losses = []
    for epoch in range(config.Training.epochs):
        model.train()
        mse_loss = 0.0

        for batch_idx, x1 in enumerate(train_loader):
            #print(f"Batch {batch_idx+1}/{len(train_loader)}")
                
            x1 = torch.tensor(x1) if isinstance(x1, np.ndarray) else x1
            x1 = x1.to(config.device)

            # Encode to latent space
            with torch.no_grad():
                z1 = ae.encode(x1)
                #mu1, logvar1 = ae.encode(x1)
                #z1 = ae.reparameterize(mu1, logvar1)
            
            # DDPM in latent space
            total_loss = ddpm_standard_step(dataset, model, diffusion, z1, optimizer, config)
            mse_loss += total_loss.item()
            
        mse_loss /= len(train_loader)
        mse_losses.append(mse_loss)

        # Validation loss
        model.eval()
        val_loss = 0.0
        divergence_loss = 0.0
        with torch.no_grad():
            for val_batch in val_loader:
                x1 = val_batch
                x1 = x1.to(config.device)
                z1 = ae.encode(x1)
                #mu1, logvar1 = ae.encode(x1)
                #z1 = ae.reparameterize(mu1, logvar1)
                batch_size = z1.shape[0]
                t = torch.randint(0, diffusion.num_timesteps, size=(batch_size,), device=z1.device)
                z_t, noise = diffusion.forward(z1, t)
                pred = model(z_t, t)
                pred = pred.sample
                val_loss += ((noise - pred) ** 2).mean().item()
                
                a_b = diffusion.alphas_b[t].view(batch_size, 1, 1, 1)
                a_b = a_b.view(-1, 1, 1, 1, 1)
                z0_pred = (z_t - (1 - a_b).sqrt() * pred) / a_b.sqrt()
                x0_pred = model.decode(z0_pred)
                eq_residual = utils.compute_divergence(dataset.data_scaler.inverse(x0_pred[:, :3, :, :, :]), 2*math.pi/config.Data.grid_size)
                eq_res_m = torch.mean(torch.abs(eq_residual))
                divergence_loss += eq_res_m
                
        val_loss /= len(val_loader)
        divergence_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        wandb.log({
            "epoch": epoch+1,
            "train_loss": mse_loss,
            "validation_loss": val_loss,
            "validation_divergence": divergence_loss
        })
        
        # Custom LR scheduler: multiply by gamma, but do not go below last_lr
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
            new_lr = max(current_lr * config.Training.gamma, config.Training.last_lr)
            param_group['lr'] = new_lr

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 100 == 0:
            checkpoint_path = os.path.join(run_dir, f"epoch_{epoch+1}_{mse_loss:.4f}_{val_loss:.4f}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

        # Log the epoch loss and validation loss
        print(f"Epoch [{epoch + 1}/{config.Training.epochs}], Loss: {mse_loss:.4f}, Validation Loss: {val_loss:.4f}")
        
    wandb.finish()

    # Plot losses after training
    plt.figure()
    plt.plot(range(1, config.Training.epochs + 1), mse_losses, label='Train MSE Loss')
    plt.plot(range(1, config.Training.epochs + 1), val_losses, label='Validation Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    image_path = os.path.join(run_dir, "history.png")
    plt.savefig(image_path)

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

    # Train the model
    train_flow_matching(config, config_ae)