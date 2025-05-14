import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import yaml

from dataset import IsotropicTurbulenceDataset
import utils
from model_simple import Model_base

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
    model = Model_base(config)
    model = model.to(config.device)

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
    for epoch in range(config.Training.epochs):
        model.train()
        epoch_loss = 0.0
        mse_loss = 0.0

        # Get the next batch from the train_loader
        for batch_idx, x1 in enumerate(train_loader):
            # Ensure all elements in the batch are tensors
            x1 = torch.tensor(x1) if isinstance(x1, np.ndarray) else x1

            print(f"Batch {batch_idx+1}/{len(train_loader)}")
            
            x0 = torch.randn_like(x1)
            target = x1 - x0

            # Ensure all tensors are on the same device
            x1 = x1.to(config.device)
            x0 = x0.to(config.device)
            target = target.to(config.device)

            # Sample random time steps
            t = torch.rand(x1.size(0), device=config.device)

            # Interpolate between x0 and x1
            xt = (1 - t[:, None, None, None, None]) * x0 + t[:, None, None, None, None] * x1

            # Forward pass
            pred = model(xt, t)
            x1_pred = xt + (1 - t[:, None, None, None, None]) * pred

            # Compute the loss
            loss = ((target - pred) ** 2).mean()

            # Compute the divergence-free loss
            divergence = utils.compute_divergence(x1_pred)
            divergence_loss = torch.mean(divergence ** 2)

            # Combine the flow matching loss and the divergence-free loss
            total_loss = loss + config.Training.divergence_loss_weight * divergence_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            mse_loss += loss.item()
            
        mse_loss /= len(train_loader)

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
                val_loss += ((target - pred) ** 2).mean().item()

        val_loss /= len(val_loader)

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(run_dir, f"epoch_{epoch+1}_{mse_loss:.4f}_{val_loss:.4f}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

        # Log the epoch loss and validation loss
        print(f"Epoch [{epoch + 1}/{config.Training.epochs}], Loss: {mse_loss:.4f}, Validation Loss: {val_loss:.4f}")

if __name__ == "__main__":
    # Load the configuration
    print("Loading config...")
    with open("configs/config_file.yml", "r") as f:
        config = yaml.safe_load(f)
    config = utils.dict2namespace(config)
    print(config.device)

    # Train the model
    train_flow_matching(config)