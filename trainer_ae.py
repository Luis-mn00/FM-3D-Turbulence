import torch
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import wandb
from conflictfree.utils import get_gradient_vector
from conflictfree.grad_operator import ConFIGOperator

from model_ae import CVAE_3D, CVAE_3D_II
from loss import schedule_KL_annealing
from dataset import IsotropicTurbulenceDataset
import utils
from my_config_length import UniProjectionLength

wandb.login(key="f4a726b2fe7929990149e82fb88da423cfa74e46")

wandb.init(project="Fluid AE")

def standard_step(model, batch, kl_weight, optimizer, config):
    # move data into GPU tensors
    batch = batch.to(config.device)

    # call CVAE model
    # feeding 3D volume to Conv3D: https://discuss.pytorch.org/t/feeding-3d-volumes-to-conv3d/32378/6
    recon_batch, mu, logvar, _ = model(batch)

    # compute batch losses
    mse_loss = torch.nn.MSELoss(reduction='sum')(recon_batch, batch)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = mse_loss + kl_weight * kl_loss

    # compute gradients and update weights
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss, mse_loss, kl_loss, 0

def PINN_step(model, batch, kl_weight, optimizer, config):
    # move data into GPU tensors
    batch = batch.to(config.device)

    recon_batch, mu, logvar, _ = model(batch)

    # compute batch losses
    mse_loss = torch.nn.MSELoss(reduction='sum')(recon_batch, batch)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    ae_loss = mse_loss + kl_weight * kl_loss
    
    # Compute the divergence-free loss
    divergence = utils.compute_divergence(recon_batch)
    divergence_loss = torch.mean(divergence ** 2)

    # Combine the flow matching loss and the divergence-free loss
    total_loss = ae_loss + config.Training.divergence_loss_weight * divergence_loss
    
    # Backward pass and optimization
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return total_loss, mse_loss, kl_loss, divergence_loss

def PINN_dyn_step(model, batch, kl_weight, optimizer, config):
    # move data into GPU tensors
    batch = batch.to(config.device)

    recon_batch, mu, logvar, _ = model(batch)

    # compute batch losses
    mse_loss = torch.nn.MSELoss(reduction='sum')(recon_batch, batch)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    ae_loss = mse_loss + kl_weight * kl_loss
    
    # Compute the divergence-free loss
    divergence = utils.compute_divergence(recon_batch)
    divergence_loss = torch.mean(divergence ** 2)

    # Combine the flow matching loss and the divergence-free loss
    coef = ae_loss / divergence_loss
    total_loss = ae_loss + coef * divergence_loss
    
    # Backward pass and optimization
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return total_loss, mse_loss, kl_loss, divergence_loss

def ConFIG_step(model, batch, kl_weight, optimizer, config, operator):
    # move data into GPU tensors
    batch = batch.to(config.device)

    recon_batch, mu, logvar, _ = model(batch)

    # compute batch losses
    mse_loss = torch.nn.MSELoss(reduction='sum')(recon_batch, batch)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    ae_loss = mse_loss + kl_weight * kl_loss
    
    # Compute the divergence-free loss
    divergence = utils.compute_divergence(recon_batch)
    divergence_loss = torch.mean(divergence ** 2)
    
    # ConFIG
    loss_physics_unscaled = divergence_loss.clone()
    ae_loss.backward(retain_graph=True)
    grads_1 = get_gradient_vector(model, none_grad_mode="skip")
    optimizer.zero_grad()
    divergence_loss.backward()
    grads_2 = get_gradient_vector(model, none_grad_mode="skip")

    operator.update_gradient(model, [grads_1, grads_2])
    optimizer.step()
    
    total_loss = ae_loss + config.Training.divergence_loss_weight * divergence_loss
    
    return total_loss, mse_loss, kl_loss, divergence_loss

def train_ae(config):
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

    # instantiate model and initialize network weights
    model = CVAE_3D_II(image_channels=config.Model.in_channels, h_dim=config.Model.h_dim, z_dim=config.Model.z_dim).to(device=config.device, dtype=torch.float)
    model.apply(utils.init_weights) # xavier initialization
    
    # Convert learning_rate and divergence_loss_weight to float if they are strings
    if isinstance(config.Training.learning_rate, str):
        config.Training.learning_rate = float(config.Training.learning_rate)
    if isinstance(config.Training.divergence_loss_weight, str):
        config.Training.divergence_loss_weight = float(config.Training.divergence_loss_weight)
    if isinstance(config.Training.gamma, str):
        config.Training.gamma = float(config.Training.gamma)
    if isinstance(config.Training.last_lr, str):
        config.Training.last_lr = float(config.Training.last_lr)
    
    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.Training.learning_rate) # 1e-4 0 KLD, 1e-3 works, 1e-1 & 1e-2 gives NaN
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader))

    # Find the next run directory
    runs_dir = "runs"
    existing = [d for d in os.listdir(runs_dir) if d.isdigit()]
    if existing:
        next_run = f"{max([int(d) for d in existing])+1:03d}"
    else:
        next_run = "001"
    run_dir = os.path.join(runs_dir, next_run)
    os.makedirs(run_dir, exist_ok=True)

    # schedule KL annealing
    kl_weights = schedule_KL_annealing(0.0, 1.0, config.Training.epochs, 4) # cyclical annealing
    kl_weight = 0
    
    # Training loop with validation loss
    print("Starting training...")
    train_losses = []
    val_losses = []
    for epoch in range(config.Training.epochs):
        model.train()
        epoch_total_loss = 0
        epoch_BCE_loss = 0
        epoch_KLD_loss = 0
        epoch_div_loss = 0
        
        # update KL weight at every epoch
        kl_weight = kl_weights[epoch]
        print("current KL weight:", kl_weight)

        # Get the next batch from the train_loader
        for batch_idx, x1 in enumerate(train_loader):
            print(f"Batch {batch_idx+1}/{len(train_loader)}")

            # Perform the training step
            if config.Training.method == "std":
                train_total_loss, train_BCE_loss, train_KLD_loss, train_div_loss = standard_step(model, x1, kl_weight, optimizer, config)
            
            elif config.Training.method == "PINN":
                train_total_loss, train_BCE_loss, train_KLD_loss, train_div_loss = PINN_step(model, x1, kl_weight, optimizer, config)
                
            elif config.Training.method == "PINN_dyn":
                train_total_loss, train_BCE_loss, train_KLD_loss, train_div_loss = PINN_dyn_step(model, x1, kl_weight, optimizer, config)
                
            elif config.Training.method == "ConFIG":
                operator = ConFIGOperator(length_model=UniProjectionLength())
                train_total_loss, train_BCE_loss, train_KLD_loss, train_div_loss = ConFIG_step(model, x1, kl_weight, optimizer, config, operator)
            
            epoch_total_loss += train_total_loss.item()
            epoch_BCE_loss += train_BCE_loss.item()
            epoch_KLD_loss += train_KLD_loss.item()
            epoch_div_loss += train_div_loss.item()
            
        epoch_BCE_loss /= len(train_loader)
        train_losses.append(epoch_BCE_loss)

        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_batch in val_loader:
                # move data into GPU tensors
                val_batch = val_batch.to(config.device)
                recon_batch, mu, logvar, _ = model(val_batch)
                mse_loss = torch.nn.MSELoss(reduction='sum')(recon_batch, val_batch)
                val_loss += mse_loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        wandb.log({
            "epoch": epoch+1,
            "train_loss": epoch_total_loss,
            "validation_loss": val_loss
        })
        
        # Custom LR scheduler: multiply by gamma, but do not go below last_lr
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
            new_lr = max(current_lr * config.Training.gamma, config.Training.last_lr)
            param_group['lr'] = new_lr

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(run_dir, f"epoch_{epoch+1}_{epoch_total_loss:.4f}_{val_loss:.4f}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

        # Log the epoch loss and validation loss
        print(f"Epoch [{epoch + 1}/{config.Training.epochs}], Loss: {epoch_total_loss:.4f}, Validation Loss: {val_loss:.4f}")
        
    wandb.finish()

    # Plot losses after training
    plt.figure()
    plt.plot(range(1, config.Training.epochs + 1), train_losses, label='Train MSE Loss')
    plt.plot(range(1, config.Training.epochs + 1), val_losses, label='Validation Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    image_path = os.path.join(run_dir, "history.png")
    plt.savefig(image_path)
        
        
if __name__ == '__main__':
    # Load the configuration
    print("Loading config...")
    with open("configs/config_ae.yml", "r") as f:
        config = yaml.safe_load(f)
    config = utils.dict2namespace(config)
    print(config.device)
    
    # Train the model
    train_ae(config)