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

from dataset import IsotropicTurbulenceDataset, BigIsotropicTurbulenceDataset
import utils
from model_regression import Model_base
from my_config_length import UniProjectionLength

wandb.login(key="f4a726b2fe7929990149e82fb88da423cfa74e46")

wandb.init(project="fm")

def standard_step(model, x, target, optimizer, config):
    # Forward pass
    pred = model(x)
    loss = ((target - pred) ** 2).mean()
    total_loss = loss
    
    # Backward pass and optimization
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return total_loss, loss

def PINN_step(model, x, target, optimizer, config):
    # Forward pass
    pred = model(x)
    loss = ((target - pred) ** 2).mean()

    # Compute the divergence-free loss
    divergence = utils.compute_divergence(pred[:, :3, :, :, :])
    divergence_loss = torch.mean(divergence ** 2)

    # Combine the flow matching loss and the divergence-free loss
    total_loss = loss + config.Training.divergence_loss_weight * divergence_loss
    
    # Backward pass and optimization
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return total_loss, loss

def PINN_dyn_step(model, x, target, optimizer, config):
    # Forward pass
    pred = model(x)
    loss = ((target - pred) ** 2).mean()

    # Compute the divergence-free loss
    divergence = utils.compute_divergence(pred[:, :3, :, :, :])
    divergence_loss = torch.mean(divergence ** 2)

    # Combine the flow matching loss and the divergence-free loss
    coef = loss / divergence_loss
    total_loss = loss + coef * divergence_loss
    
    # Backward pass and optimization
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return total_loss, loss

def ConFIG_step(model, x, target, optimizer, config, operator):
    # Forward pass
    pred = model(x)
    loss = ((target - pred) ** 2).mean()

    # Compute the divergence-free loss
    divergence = utils.compute_divergence(pred[:, :3, :, :, :])
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

class RegressionDataset(Dataset):
    def __init__(self, low_res_images, high_res_images):
        self.low_res_images = low_res_images
        self.high_res_images = high_res_images

    def __len__(self):
        return len(self.low_res_images)

    def __getitem__(self, idx):
        low_res = self.low_res_images[idx]
        high_res = self.high_res_images[idx]

        return low_res, high_res

def create_dataloader(low_res_images, high_res_images, batch_size):
    dataset = RegressionDataset(low_res_images, high_res_images)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Define the training function
def train_flow_matching(config):
    # Load the dataset
    print("Loading dataset...")
    dataset = IsotropicTurbulenceDataset(dt=config.Data.dt, grid_size=config.Data.grid_size, crop=config.Data.crop, seed=config.Data.seed, size=config.Data.size, batch_size=config.Training.batch_size)
    #dataset = BigIsotropicTurbulenceDataset("/mnt/data4/pbdl-datasets-local/3d_jhtdb/isotropic1024coarse.hdf5", sim_group='sim0', norm=True, size=None, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, batch_size=config.Training.batch_size)

    # Update the dataloaders
    train_loader = dataset.train_loader
    val_loader = dataset.val_loader
    test_loader = dataset.test_loader

    # Initialize the model
    model = Model_base(config)
    model = model.to(config.device)

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
        for batch_Y in train_loader:
            batch_idx += 1
            print(f"Batch {batch_idx}/{len(train_loader)}")

            # Ensure all elements in the batch are tensors
            batch_X, samples_ids = utils.interpolate_dataset(batch_Y, config.Data.perc / 100)
            y = torch.tensor(batch_Y) if isinstance(batch_Y, np.ndarray) else batch_Y
            x = torch.tensor(batch_X) if isinstance(batch_X, np.ndarray) else batch_X
            y = y.to(config.device)
            x = x.to(config.device)
            
            # Perform the training step
            if config.Training.method == "std":
                total_loss, loss = standard_step(model, x, y, optimizer, config)
                
            elif config.Training.method == "PINN":
                total_loss, loss = PINN_step(model, x, y, optimizer, config)
                
            elif config.Training.method == "PINN_dyn":
                total_loss, loss = PINN_dyn_step(model, x, y, optimizer, config)
                
            elif config.Training.method == "ConFIG":
                operator = ConFIGOperator(length_model=UniProjectionLength())
                total_loss, loss = ConFIG_step(model, x, y, optimizer, config, operator)
                
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
            for batch_Y in val_loader:
                batch_X, samples_ids = utils.interpolate_dataset(batch_Y, config.Data.perc / 100)
                y = torch.tensor(batch_Y) if isinstance(batch_Y, np.ndarray) else batch_Y
                x = torch.tensor(batch_X) if isinstance(batch_X, np.ndarray) else batch_X
                y = y.to(config.device)
                x = x.to(config.device)

                pred = model(x)
                val_loss += ((y - pred) ** 2).mean().item()

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
    with open("configs/config_regression.yml", "r") as f:
        config = yaml.safe_load(f)
    config = utils.dict2namespace(config)
    print(config.device)

    # Train the model
    train_flow_matching(config)