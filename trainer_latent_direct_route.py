import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import wandb
from conflictfree.utils import get_gradient_vector
from conflictfree.grad_operator import ConFIGOperator
import math

from dataset import IsotropicTurbulenceDataset, BigIsotropicTurbulenceDataset, BigSpectralIsotropicTurbulenceDataset, SupervisedSpectralTurbulenceDataset
import utils
from my_config_length import UniProjectionLength
from model_vqvae import VQVAE, VAE, AE
from src.core.models.box.pdedit import PDEDiT3D_S, PDEDiT3D_B, PDEDiT3D_L

wandb.login(key="f4a726b2fe7929990149e82fb88da423cfa74e46")

wandb.init(project="Latent fm DP v")

def fm_standard_step(model, xt, t, target, optimizer, config):
    # Forward pass
    pred = model(xt, t)
    pred = pred.sample
    loss = ((target - pred) ** 2).mean()
    total_loss = loss
    
    # Backward pass and optimization
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return total_loss, loss

def fm_PINN_step(dataset, model, xt, t, target, optimizer, config):
    # Forward pass
    pred = model(xt, t)
    pred = pred.sample
    loss = ((target - pred) ** 2).mean()
    
    x1_pred = xt + (1 - t[:, None, None, None, None]) * pred

    # Compute the divergence-free loss
    divergence = utils.compute_divergence(dataset.Y_scaler.inverse(x1_pred[:, :3, :, :, :]), 2*math.pi/config.Data.grid_size)
    divergence_loss = torch.mean(torch.abs(divergence))

    # Combine the flow matching loss and the divergence-free loss
    total_loss = loss + config.Training.divergence_loss_weight * divergence_loss
    
    # Backward pass and optimization
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return total_loss, loss

def fm_PINN_dyn_step(dataset, model, xt, t, target, optimizer, config):
    # Forward pass
    pred = model(xt, t)
    pred = pred.sample
    loss = ((target - pred) ** 2).mean()
    
    x1_pred = xt + (1 - t[:, None, None, None, None]) * pred

    # Compute the divergence-free loss
    divergence = utils.compute_divergence(dataset.Y_scaler.inverse(x1_pred[:, :3, :, :, :]), 2*math.pi/config.Data.grid_size)
    divergence_loss = torch.mean(torch.abs(divergence))

    # Combine the flow matching loss and the divergence-free loss
    coef = loss / divergence_loss
    total_loss = loss + coef * divergence_loss
    
    # Backward pass and optimization
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return total_loss, loss

def fm_ConFIG_step(dataset, model, xt, t, target, optimizer, config, operator):
    # Forward pass
    pred = model(xt, t)
    pred = pred.sample
    loss = ((target - pred) ** 2).mean()
    
    x1_pred = xt + (1 - t[:, None, None, None, None]) * pred

    # Compute the divergence-free loss
    divergence = utils.compute_divergence(dataset.Y_scaler.inverse(x1_pred[:, :3, :, :, :]), 2*math.pi/config.Data.grid_size)
    divergence_loss = torch.mean(torch.abs(divergence))
    
    # ConFIG
    loss_physics_unscaled = divergence_loss.clone()
    loss.backward(retain_graph=True)
    grads_1 = get_gradient_vector(model, none_grad_mode="skip")
    optimizer.zero_grad()
    divergence_loss.backward()
    grads_2 = get_gradient_vector(model, none_grad_mode="skip")

    operator.update_gradient(model, [grads_1, grads_2])
    optimizer.step()
    
    total_loss = loss + config.Training.divergence_loss_weight * divergence_loss
    
    return total_loss, loss

# Define the training function
def train_flow_matching(config, config_ae):
    # Load the dataset
    #dataset = IsotropicTurbulenceDataset(dt=config_ae.Data.dt, grid_size=config_ae.Data.grid_size, crop=config_ae.Data.crop, seed=config_ae.Data.seed, size=config_ae.Data.size, batch_size=config.Training.batch_size)
    #dataset = BigIsotropicTurbulenceDataset("/mnt/data4/pbdl-datasets-local/3d_jhtdb/isotropic1024coarse.hdf5", sim_group='sim0', norm=True, size=None, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, batch_size=config.Training.batch_size)
    dataset = SupervisedSpectralTurbulenceDataset(grid_size=config.Data.grid_size,
                                                    norm=config.Data.norm,
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
    
    # Load the pre-trained autoencoder
    #model = VQVAE(input_size=config.Model.in_channels, hidden_size=config.Model.hidden_size, depth=config.Model.depth, num_res_block=config.Model.num_res_block, res_size=config.Model.res_size, embedding_size=config.Model.embedding_size,
    #             num_embedding=config.Model.num_embedding, device=config.device).to(config.device)
    #model = AE(input_size=config.Model.in_channels, image_size=config.Data.grid_size, hidden_size=config.Model.hidden_size, depth=config.Model.depth, num_res_block=config.Model.num_res_block, res_size=config.Model.res_size, embedding_size=config.Model.embedding_size,
    #            device=config.device, z_dim=config.Model.z_dim).to(config.device)
    ae = VAE(input_size=config_ae.Model.in_channels, hidden_size=config_ae.Model.hidden_size, depth=config_ae.Model.depth, num_res_block=config_ae.Model.num_res_block, res_size=config_ae.Model.res_size, embedding_size=config_ae.Model.embedding_size,
                device=config_ae.device).to(config.device)
    ae.load_state_dict(torch.load(config_ae.Model.ae_path, map_location=config.device))
    ae.eval()
    for param in ae.parameters():
        param.requires_grad = False

    # Convert learning_rate and divergence_loss_weight to float if they are strings
    if isinstance(config.Training.learning_rate, str):
        config.Training.learning_rate = float(config.Training.learning_rate)
    if isinstance(config.Training.divergence_loss_weight, str):
        config.Training.divergence_loss_weight = float(config.Training.divergence_loss_weight)
    if isinstance(config.Training.sigma_min, str):
        config.Training.sigma_min = float(config.Training.sigma_min)
    if isinstance(config.Training.gamma, str):
        config.Training.gamma = float(config.Training.gamma)
    if isinstance(config.Training.last_lr, str):
        config.Training.last_lr = float(config.Training.last_lr)

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.Training.learning_rate)

    # Find the next run directory
    runs_dir = "runs"
    existing = [d for d in os.listdir(runs_dir) if d.isdigit()]
    if existing:
        next_run = f"{max([int(d) for d in existing])+1:03d}"
    else:
        next_run = "001"
    run_dir = os.path.join(runs_dir, next_run)
    os.makedirs(run_dir, exist_ok=True)

    # Training loop with validation loss
    print("Starting training...")
    mse_losses = []
    val_losses = []
    for epoch in range(config.Training.epochs):
        model.train()
        epoch_loss = 0.0
        mse_loss = 0.0

        # Get the next batch from the train_loader
        batch_idx = 0
        for batch_X, batch_Y in train_loader:
            batch_idx += 1
            #print(f"Batch {batch_idx}/{len(train_loader)}")

            # Ensure all elements in the batch are tensors
            x1 = torch.tensor(batch_Y) if isinstance(batch_Y, np.ndarray) else batch_Y
            x0 = torch.tensor(batch_X) if isinstance(batch_X, np.ndarray) else batch_X
            
            # Apply encoder to x0 and x1
            mu1, logvar1 = ae.encode(x1)
            mu0, logvar0 = ae.encode(x0)
            z1 = ae.reparameterize(mu1, logvar1)
            z0 = ae.reparameterize(mu0, logvar0)
            
            target = z1 - (1 - config.Training.sigma_min) * z0
            target = target.to(config.device)

            # Sample random time steps
            t = torch.rand(x1.size(0), device=config.device)

            # Interpolate between x0 and x1
            zt = (1 - (1 - config.Training.sigma_min) * t[:, None, None, None, None]) * z0 + t[:, None, None, None, None] * z1
            
            # Perform the training step
            total_loss, loss = fm_standard_step(model, zt, t, target, optimizer, config)
                
            epoch_loss += total_loss.item()
            mse_loss += loss.item()
            
        mse_loss /= len(train_loader)
        mse_losses.append(mse_loss)

        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                x1 = torch.tensor(batch_Y) if isinstance(batch_Y, np.ndarray) else batch_Y
                x0 = torch.tensor(batch_X) if isinstance(batch_X, np.ndarray) else batch_X
                
                # Apply encoder to x0 and x1
                mu1, logvar1 = ae.encode(x1)
                mu0, logvar0 = ae.encode(x0)
                z1 = ae.reparameterize(mu1, logvar1)
                z0 = ae.reparameterize(mu0, logvar0)
                
                target = z1 - (1 - config.Training.sigma_min) * z0
                target = target.to(config.device)

                t = torch.rand(x1.size(0), device=config.device)
                zt = (1 - t[:, None, None, None, None]) * z0 + t[:, None, None, None, None] * z1

                pred = model(zt, t)
                val_loss += ((target - pred) ** 2).mean().item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        wandb.log({
            "epoch": epoch+1,
            "train_loss": mse_loss,
            "validation_loss": val_loss
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
    with open("configs/config_lfm.yml", "r") as f:
        config = yaml.safe_load(f)
    config = utils.dict2namespace(config)
    print(config.device)
    
    print("Loading config...")
    with open("configs/config_vqvae.yml", "r") as f:
        config_ae = yaml.safe_load(f)
    config_ae = utils.dict2namespace(config_ae)

    # Train the model
    train_flow_matching(config, config_ae)