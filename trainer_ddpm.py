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
from my_config_length import UniProjectionLength
from diffusion import Diffusion

wandb.login(key="f4a726b2fe7929990149e82fb88da423cfa74e46")

wandb.init(project="fm")

def ddpm_standard_step(model, diffusion, y, optimizer, config):
    batch_size = y.shape[0]
    t = torch.randint(0, diffusion.num_timesteps, size=(batch_size,), device=y.device)

    x_t, noise = diffusion.forward(y, t)
    e_pred = model(x_t, t)
    mse_loss = (noise - e_pred).square().mean()

    mse_loss.backward()
    optimizer.step()

    # Compute res_loss for metrics comparison
    with torch.no_grad():
        a_b = diffusion.alphas_b[t].view(batch_size, 1, 1, 1)
        a_b = a_b.view(-1, 1, 1, 1, 1)
        x0_pred = (x_t - (1 - a_b).sqrt() * e_pred) / a_b.sqrt()
        eq_residual = utils.compute_divergence(x0_pred[:, :3, :, :, :])
        eq_res_m = torch.mean(eq_residual ** 2)

    return mse_loss, eq_res_m

def ddpm_PINN_step(model, diffusion, y, optimizer, config):
    batch_size = y.shape[0]
    t = torch.randint(0, diffusion.num_timesteps, size=(batch_size,), device=y.device)

    x_t, noise = diffusion.forward(y, t)
    e_pred = model(x_t, t)
    mse_loss = (noise - e_pred).square().mean()

    # Compute res_loss for metrics comparison
    with torch.no_grad():
        a_b = diffusion.alphas_b[t].view(batch_size, 1, 1, 1)
        a_b = a_b.view(-1, 1, 1, 1, 1)
        x0_pred = (x_t - (1 - a_b).sqrt() * e_pred) / a_b.sqrt()
        eq_residual = utils.compute_divergence(x0_pred[:, :3, :, :, :])
        eq_res_m = torch.mean(eq_residual ** 2)
        
    total_loss = mse_loss + config.Training.ddpm_loss_weight * eq_res_m
    total_loss.backward()
    optimizer.step()

    return mse_loss, eq_res_m

def ddpm_PINN_dyn_step(model, diffusion, y, optimizer, config):
    batch_size = y.shape[0]
    t = torch.randint(0, diffusion.num_timesteps, size=(batch_size,), device=y.device)

    x_t, noise = diffusion.forward(y, t)
    e_pred = model(x_t, t)
    mse_loss = (noise - e_pred).square().mean()

    # Compute res_loss for metrics comparison
    with torch.no_grad():
        a_b = diffusion.alphas_b[t].view(batch_size, 1, 1, 1)
        a_b = a_b.view(-1, 1, 1, 1, 1)
        x0_pred = (x_t - (1 - a_b).sqrt() * e_pred) / a_b.sqrt()
        eq_residual = utils.compute_divergence(x0_pred[:, :3, :, :, :])
        eq_res_m = torch.mean(eq_residual ** 2)
        
    coef = mse_loss / eq_res_m
        
    total_loss = mse_loss + coef * eq_res_m
    total_loss.backward()
    optimizer.step()

    return mse_loss, eq_res_m

def ddpm_ConFIG_step(model, diffusion, y, optimizer, config, operator):
    batch_size = y.shape[0]
    t = torch.randint(0, diffusion.num_timesteps, size=(batch_size,), device=y.device)

    x_t, noise = diffusion.forward(y, t)
    e_pred = model(x_t, t)
    mse_loss = (noise - e_pred).square().mean()

    # Compute res_loss for metrics comparison
    with torch.no_grad():
        a_b = diffusion.alphas_b[t].view(batch_size, 1, 1, 1)
        a_b = a_b.view(-1, 1, 1, 1, 1)
        x0_pred = (x_t - (1 - a_b).sqrt() * e_pred) / a_b.sqrt()
        eq_residual = utils.compute_divergence(x0_pred[:, :3, :, :, :])
        eq_res_m = torch.mean(eq_residual ** 2)
    
    # ConFIG
    loss_physics_unscaled = eq_res_m.clone()
    mse_loss.backward(retain_graph=True)
    grads_1 = get_gradient_vector(model, none_grad_mode="skip")
    optimizer.zero_grad()
    eq_res_m.backward()
    grads_2 = get_gradient_vector(model, none_grad_mode="skip")

    operator.update_gradient(model, [grads_1, grads_2])
    optimizer.step()
    
    return mse_loss, loss_physics_unscaled

# Define the training function
def train_ddpm(config):
    # Load the dataset
    print("Loading dataset...")
    dataset = IsotropicTurbulenceDataset(dt=config.Data.dt, grid_size=config.Data.grid_size, crop=config.Data.crop, seed=config.Data.seed, size=config.Data.size, batch_size=config.Training.batch_size)
    #dataset = BigIsotropicTurbulenceDataset("/mnt/data4/pbdl-datasets-local/3d_jhtdb/isotropic1024coarse.hdf5", sim_group='sim0', norm=True, size=config.Data.size, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, batch_size=config.Training.batch_size)

    # Update the dataloaders
    train_loader = dataset.train_loader
    val_loader = dataset.val_loader
    test_loader = dataset.test_loader
    
    # Diffusion parameters
    diffusion = Diffusion(config)

    # Initialize the model
    model = Model_base(config)
    model = model.to(config.device)

    # Convert learning_rate and divergence_loss_weight to float if they are strings
    if isinstance(config.Training.learning_rate, str):
        config.Training.learning_rate = float(config.Training.learning_rate)
    if isinstance(config.Training.ddpm_loss_weight, str):
        config.Training.ddpm_loss_weight = float(config.Training.ddpm_loss_weight)
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
        mse_loss = 0.0

        # Get the next batch from the train_loader
        for batch_idx, x1 in enumerate(train_loader):
            #print(f"Batch {batch_idx+1}/{len(train_loader)}")
            optimizer.zero_grad()
            
            # Ensure all elements in the batch are tensors
            x1 = torch.tensor(x1) if isinstance(x1, np.ndarray) else x1
            x1 = x1.to(config.device)
            
            # Perform the training step
            if config.Training.method == "std":
                mse_loss, physics_loss = ddpm_standard_step(model, diffusion, x1, optimizer, config)
                
            elif config.Training.method == "PINN":
                mse_loss, physics_loss = ddpm_PINN_step(model, diffusion, x1, optimizer, config)
                
            elif config.Training.method == "PINN_dyn":
                mse_loss, physics_loss = ddpm_PINN_dyn_step(model, diffusion, x1, optimizer, config)
                
            elif config.Training.method == "ConFIG":
                operator = ConFIGOperator(length_model=UniProjectionLength())
                mse_loss, physics_loss = ddpm_ConFIG_step(model, diffusion, x1, optimizer, config, operator)
                
            else:
                raise ValueError(f"Unknown training method: {config.Training.method}")
            
            mse_loss += mse_loss.item()
            
        mse_loss /= len(train_loader)
        mse_losses.append(mse_loss)

        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_batch in val_loader:
                val_batch = val_batch.to(config.device)  # Ensure val_batch is on the correct device
                batch_size = val_batch.shape[0]
                t = torch.randint(0, diffusion.num_timesteps, size=(batch_size,), device=val_batch.device)
                x_t, noise = diffusion.forward(val_batch, t)
                e_pred = model(x_t, t)
                val_loss += (noise - e_pred).square().mean()
                
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
    mse_losses_np = [x.detach().cpu().numpy() if torch.is_tensor(x) else x for x in mse_losses]
    val_losses_np = [x.detach().cpu().numpy() if torch.is_tensor(x) else x for x in val_losses]
    plt.plot(range(1, config.Training.epochs + 1), mse_losses_np, label='Train MSE Loss')
    plt.plot(range(1, config.Training.epochs + 1), val_losses_np, label='Validation Loss')
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
    with open("configs/config_ddpm.yml", "r") as f:
        config = yaml.safe_load(f)
    config = utils.dict2namespace(config)
    print(config.device)

    # Train the model
    train_ddpm(config)