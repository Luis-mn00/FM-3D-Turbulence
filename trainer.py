import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from model_simple import Model_base
import numpy as np

# Define a simple dataset class
class FlowDataset(Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path, weights_only=False)
        # Only consider the first 3 channels
        self.data = self.data[:, :3, :, :, :]
        print(self.data.shape)  # Print the shape of the loaded data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

# Define a configuration class
class Config:
    class Data:
        path = 'data_0.1_64.pt'
        image_size = 64  # Set the resolution of the input data

    class Model:
        ch = 64
        out_ch = 3
        ch_mult = [1, 2, 4]
        num_res_blocks = 2
        attn_resolutions = []
        dropout = 0.1
        in_channels = 3
        save_path = 'model_pinn.pth'
        resamp_with_conv = True

    class Training:
        epochs = 100
        batch_size = 5
        learning_rate = 1e-4
        divergence_loss_weight = 0.1

    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = "cpu"
    
def physical_loss(x1_pred): # CAMBIAR dim
    vx, vy, vz = x1_pred[:, 0:1], x1_pred[:, 1:2], x1_pred[:, 2:3]
    dvx_dx = torch.gradient(vx, dim=2)[0]
    dvy_dy = torch.gradient(vy, dim=3)[0]
    dvz_dz = torch.gradient(vz, dim=4)[0]  
    divergence = dvx_dx + dvy_dy + dvz_dz
    divergence_loss = torch.mean(divergence ** 2)
    
    return divergence_loss

# Define the training function
def train_flow_matching(config):
    # Load the dataset
    dataset = FlowDataset(config.Data.path)

    # Define the dataset split ratios
    train_ratio = 0.8
    val_ratio = 0.1

    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Update the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.Training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.Training.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.Training.batch_size, shuffle=False)

    # Initialize the model
    model = Model_base(config)
    model = model.to(config.device)

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.Training.learning_rate)

    # Training loop with validation loss
    for epoch in range(config.Training.epochs):
        model.train()
        epoch_loss = 0.0

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
            divergence_loss = physical_loss(x1_pred)

            # Combine the flow matching loss and the divergence-free loss
            total_loss = loss + config.Training.divergence_loss_weight * divergence_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

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

        # Log the epoch loss and validation loss
        print(f"Epoch [{epoch + 1}/{config.Training.epochs}], Loss: {epoch_loss / len(train_loader):.4f}, Validation Loss: {val_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), config.Model.save_path)
    print(f"Model saved to {config.Model.save_path}")

if __name__ == "__main__":
    # Load the configuration
    config = Config()

    # Train the model
    train_flow_matching(config)