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

from model_vqvae import VQVAE, VAE, AE
from loss import schedule_KL_annealing
from dataset import IsotropicTurbulenceDataset, BigIsotropicTurbulenceDataset, BigSpectralIsotropicTurbulenceDataset
import utils
from my_config_length import UniProjectionLength

wandb.login(key="f4a726b2fe7929990149e82fb88da423cfa74e46")

wandb.init(project="Fluid AE")

def train_ae(config):
    print("Loading dataset...")
    #dataset = IsotropicTurbulenceDataset(dt=config.Data.dt, grid_size=config.Data.grid_size, crop=config.Data.crop, seed=config.Data.seed, size=config.Data.size, batch_size=config.Training.batch_size)
    #dataset = BigIsotropicTurbulenceDataset("/mnt/data4/pbdl-datasets-local/3d_jhtdb/isotropic1024coarse.hdf5", sim_group='sim0', norm=True, size=None, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, batch_size=config.Training.batch_size, grid_size=config.Data.grid_size)
    dataset = BigSpectralIsotropicTurbulenceDataset(grid_size=config.Data.grid_size,
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
    
    # instantiate model and initialize network weights
    
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
    model = VAE(input_size=config.Model.in_channels,
               image_size=config.Data.grid_size,
               hidden_size=config.Model.hidden_size,
               depth=config.Model.depth,
               num_res_block=config.Model.num_res_block,
               res_size=config.Model.res_size,
               device=config.device,
               z_dim=config.Model.z_dim).to(config.device)
    
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
    train_losses = []
    val_losses = []
    for epoch in range(config.Training.epochs):
        model.train()
        total_loss = 0

        # Get the next batch from the train_loader
        for batch_idx, x1 in enumerate(train_loader):
            #print(f"Batch {batch_idx+1}/{len(train_loader)}")
            #print(x1.shape)

            # Perform the training step
            x1 = x1.to(config.device)
            input = {'uvw': x1, 'duvw': utils.spectral_derivative_3d(x1)}
            optimizer.zero_grad()
            output = model(input, Epoch=epoch)
            total_loss_batch = output['loss'].mean()
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            
            total_loss += total_loss_batch.item()
            
        total_loss /= len(train_loader)
        train_losses.append(total_loss)

        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_batch in val_loader:
                # move data into GPU tensors
                val_batch = val_batch.to(config.device)
                input = {'uvw': val_batch, 'duvw': utils.spectral_derivative_3d(val_batch)}
                output = model(input)
                val_loss_batch = output['loss'].mean()
                val_loss += val_loss_batch.item()
                
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        wandb.log({
            "epoch": epoch+1,
            "train_loss": total_loss,
            "validation_loss": val_loss
        })
        
        # Custom LR scheduler: multiply by gamma, but do not go below last_lr
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
            new_lr = max(current_lr * config.Training.gamma, config.Training.last_lr)
            param_group['lr'] = new_lr

        # Save checkpoint every 10 epochs
        #if (epoch + 1) % 10 == 0:
        #    checkpoint_path = os.path.join(run_dir, f"epoch_{epoch+1}_{epoch_BCE_loss:.4f}_{val_loss:.4f}.pth")
        #    torch.save(model.state_dict(), checkpoint_path)
        #    print(f"Saved checkpoint: {checkpoint_path}")

        # Log the epoch loss and validation loss
        print(f"Epoch [{epoch + 1}/{config.Training.epochs}], Loss: {total_loss:.4f}, Validation Loss: {val_loss:.4f}")
        
    wandb.finish()
    
    checkpoint_path = os.path.join(run_dir, f"epoch_{epoch+1}_{total_loss:.4f}_{val_loss:.4f}.pth")
    torch.save(model.state_dict(), checkpoint_path)

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
    with open("configs/config_vqvae.yml", "r") as f:
        config = yaml.safe_load(f)
    config = utils.dict2namespace(config)
    print(config.device)
    
    # Train the model
    train_ae(config)