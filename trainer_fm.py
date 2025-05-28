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

from dataset import IsotropicTurbulenceDataset, BigIsotropicTurbulenceDataset
import utils
from model_simple import Model_base
from src.core.models.box.pdedit import PDEDiT3D_S
from my_config_length import UniProjectionLength

wandb.login(key="f4a726b2fe7929990149e82fb88da423cfa74e46")

wandb.init(project="fm")

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

def fm_PINN_step(model, xt, t, target, optimizer, config):
    # Forward pass
    pred = model(xt, t)
    loss = ((target - pred) ** 2).mean()
    
    x1_pred = xt + (1 - t[:, None, None, None, None]) * pred

    # Compute the divergence-free loss
    divergence = utils.compute_divergence(x1_pred[:, :3, :, :, :])
    #divergence_loss = torch.mean(divergence ** 2)
    divergence_loss = torch.sqrt(torch.sum(divergence ** 2))

    # Combine the flow matching loss and the divergence-free loss
    total_loss = loss + config.Training.divergence_loss_weight * divergence_loss
    
    # Backward pass and optimization
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return total_loss, loss

def fm_PINN_dyn_step(model, xt, t, target, optimizer, config):
    # Forward pass
    pred = model(xt, t)
    loss = ((target - pred) ** 2).mean()
    
    x1_pred = xt + (1 - t[:, None, None, None, None]) * pred

    # Compute the divergence-free loss
    divergence = utils.compute_divergence(x1_pred[:, :3, :, :, :])
    divergence_loss = torch.mean(divergence ** 2)

    # Combine the flow matching loss and the divergence-free loss
    coef = loss / divergence_loss
    total_loss = loss + coef * divergence_loss
    
    # Backward pass and optimization
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return total_loss, loss

def fm_ConFIG_step(model, xt, t, target, optimizer, config, operator):
    # Forward pass
    pred = model(xt, t)
    loss = ((target - pred) ** 2).mean()
    
    x1_pred = xt + (1 - t[:, None, None, None, None]) * pred

    # Compute the divergence-free loss
    divergence = utils.compute_divergence(x1_pred[:, :3, :, :, :])
    divergence_loss = torch.mean(divergence ** 2)
    
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
def train_flow_matching(config):
    # Load the dataset
    print("Loading dataset...")
    #dataset = IsotropicTurbulenceDataset(dt=config.Data.dt, grid_size=config.Data.grid_size, crop=config.Data.crop, seed=config.Data.seed, size=config.Data.size, batch_size=config.Training.batch_size, field=None)
    dataset = BigIsotropicTurbulenceDataset("/mnt/data4/pbdl-datasets-local/3d_jhtdb/isotropic1024coarse.hdf5", sim_group='sim0', norm=True, size=config.Data.size, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, batch_size=config.Training.batch_size, num_samples=10, grid_size=config.Data.grid_size)

    # Update the dataloaders
    train_loader = dataset.train_loader
    val_loader = dataset.val_loader
    test_loader = dataset.test_loader

    # Initialize the model
    model = PDEDiT3D_S(
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
        for batch_idx, x1 in enumerate(train_loader):
            print(f"Batch {batch_idx+1}/{len(train_loader)}")
            
            # Ensure all elements in the batch are tensors
            x1 = torch.tensor(x1) if isinstance(x1, np.ndarray) else x1
            x0 = torch.randn_like(x1)
            target = x1 - (1 - config.Training.sigma_min) * x0

            x1 = x1.to(config.device)
            x0 = x0.to(config.device)
            target = target.to(config.device)

            # Sample random time steps
            t = torch.rand(x1.size(0), device=config.device)

            # Interpolate between x0 and x1
            xt = (1 - (1 - config.Training.sigma_min) * t[:, None, None, None, None]) * x0 + t[:, None, None, None, None] * x1
            xt = xt.float()
            print(xt.shape)
            
            # Perform the training step
            if config.Training.method == "std":
                total_loss, loss = fm_standard_step(model, xt, t, target, optimizer, config)
                
            elif config.Training.method == "PINN":
                total_loss, loss = fm_PINN_step(model, xt, t, target, optimizer, config)
                
            elif config.Training.method == "PINN_dyn":
                total_loss, loss = fm_PINN_dyn_step(model, xt, t, target, optimizer, config)
                
            elif config.Training.method == "ConFIG":
                operator = ConFIGOperator(length_model=UniProjectionLength())
                total_loss, loss = fm_ConFIG_step(model, xt, t, target, optimizer, config, operator)
                
            else:
                raise ValueError(f"Unknown training method: {config.Training.method}")
            

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
                target = x1 - x0

                # Ensure all tensors are on the same device
                x1 = x1.to(config.device)
                x0 = x0.to(config.device)
                target = target.to(config.device)

                t = torch.rand(x1.size(0), device=config.device)
                xt = (1 - t[:, None, None, None, None]) * x0 + t[:, None, None, None, None] * x1

                pred = model(xt, t)
                pred = pred.sample
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
    with open("configs/config_fm.yml", "r") as f:
        config = yaml.safe_load(f)
    config = utils.dict2namespace(config)
    print(config.device)
    if torch.cuda.is_available():
        print(f"CUDA current device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    # Train the model
    train_flow_matching(config)