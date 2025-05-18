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

from dataset import IsotropicTurbulenceDataset
import utils
from model_latent import Model_base
from my_config_length import UniProjectionLength
from model_latent import LatentModel
from model_ae import CVAE_3D_II

wandb.login(key="f4a726b2fe7929990149e82fb88da423cfa74e46")

wandb.init(project="Latent fm")

def fm_standard_step(model, xt, t, target, optimizer, config):
    # Forward pass
    pred = model(xt, t)
    loss = ((target - pred) ** 2).mean()
    total_loss = loss
    
    # Backward pass and optimization
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return total_loss, loss

# Define the training function
def train_flow_matching(config):
    # Load the dataset
    print("Loading dataset...")
    dataset = IsotropicTurbulenceDataset(dt=config.Data.dt, grid_size=config.Data.grid_size, crop=config.Data.crop, seed=config.Data.seed, size=config.Data.size)
    velocity = dataset.velocity

    # Define the dataset split ratios
    train_ratio = 0.8
    val_ratio = 0.1

    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    # Split the dataset randomly with config.Data.seed
    indices = np.arange(total_size)
    np.random.seed(config.Data.seed)
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    train_dataset = torch.utils.data.Subset(velocity, train_indices)
    val_dataset = torch.utils.data.Subset(velocity, val_indices)
    test_dataset = torch.utils.data.Subset(velocity, test_indices)

    # Update the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.Training.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=config.Training.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.Training.batch_size, shuffle=False)

    # Initialize the model
    model = LatentModel(config)
    model = model.to(config.device)

    # Convert learning_rate and divergence_loss_weight to float if they are strings
    if isinstance(config.Training.learning_rate, str):
        config.Training.learning_rate = float(config.Training.learning_rate)
    if isinstance(config.Training.divergence_loss_weight, str):
        config.Training.divergence_loss_weight = float(config.Training.divergence_loss_weight)
    if isinstance(config.Training.sigma_min, str):
        config.Training.sigma_min = float(config.Training.divergence_loss_weight)
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

    # Load the pre-trained autoencoder
    ae = CVAE_3D_II(image_channels=config.Model.in_channels, h_dim=config.Model.h_dim, z_dim=config.Model.z_dim, input_shape=(config.Model.in_channels, config.Data.grid_size, config.Data.grid_size, config.Data.grid_size))
    ae.load_state_dict(torch.load(config.Model.ae_path, map_location=config.device))
    ae = ae.to(config.device)
    ae.eval()
    for param in ae.parameters():
        param.requires_grad = False

    # Training loop with validation loss
    print("Starting training...")
    mse_losses = []
    val_losses = []
    for epoch in range(config.Training.epochs):
        model.train()
        epoch_loss = 0.0
        mse_loss = 0.0

        for batch_idx, x1 in enumerate(train_loader):
            print(f"Batch {batch_idx+1}/{len(train_loader)}")
            x1 = torch.tensor(x1) if isinstance(x1, np.ndarray) else x1
            x0 = torch.randn_like(x1)
            x1 = x1.to(config.device)
            x0 = x0.to(config.device)

            # Encode to latent space
            with torch.no_grad():
                z1, _, _ = ae.encode(x1)
                z0, _, _ = ae.encode(x0)

            target = z1 - (1 - config.Training.sigma_min) * z0
            t = torch.rand(z1.size(0), device=config.device)
            xt = (1 - (1 - config.Training.sigma_min) * t[:, None]) * z0 + t[:, None] * z1

            # Flow matching in latent space
            total_loss, loss = fm_standard_step(model, xt, t, target, optimizer, config)
            epoch_loss += total_loss.item()
            mse_loss += loss.item()
            
        mse_loss /= len(train_loader)
        mse_losses.append(mse_loss)

        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_batch in val_loader:
                x1 = val_batch
                x0 = torch.randn_like(x1)
                x1 = x1.to(config.device)
                x0 = x0.to(config.device)
                z1, _, _ = ae.encode(x1)
                z0, _, _ = ae.encode(x0)
                target = z1 - (1 - config.Training.sigma_min) * z0
                t = torch.rand(z1.size(0), device=config.device)
                xt = (1 - (1 - config.Training.sigma_min) * t[:, None]) * z0 + t[:, None] * z1
                pred = model(xt, t)
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
        if (epoch + 1) % 10 == 0:
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
    with open("configs/config_generative.yml", "r") as f:
        config = yaml.safe_load(f)
    config = utils.dict2namespace(config)
    print(config.device)

    # Train the model
    train_flow_matching(config)