import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import wandb
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.strategies import DDPStrategy

from conflictfree.utils import get_gradient_vector
from conflictfree.grad_operator import ConFIGOperator

from dataset import IsotropicTurbulenceDataset
import utils
from model_simple import Model_base
from my_config_length import UniProjectionLength

wandb.login(key="f4a726b2fe7929990149e82fb88da423cfa74e46")
wandb.init(project="fm")

class RegressionDataset(Dataset):
    def __init__(self, low_res_images, high_res_images):
        self.low_res_images = low_res_images
        self.high_res_images = high_res_images

    def __len__(self):
        return len(self.low_res_images)

    def __getitem__(self, idx):
        return self.low_res_images[idx], self.high_res_images[idx]

def create_dataloader(low_res_images, high_res_images, batch_size):
    dataset = RegressionDataset(low_res_images, high_res_images)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

class FlowMatchingModule(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = Model_base(config)
        self.operator = ConFIGOperator(length_model=UniProjectionLength()) if config.Training.method == "ConFIG" else None
        self.loss_fn = torch.nn.MSELoss()

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.config.Training.learning_rate)

    def forward(self, xt, t):
        return self.model(xt, t)

    def training_step(self, batch, batch_idx):
        x0, x1 = batch
        t = torch.rand(x0.size(0), device=self.device)
        target = x1 - (1 - self.config.Training.sigma_min) * x0
        xt = (1 - (1 - self.config.Training.sigma_min) * t[:, None, None, None, None]) * x0 + t[:, None, None, None, None] * x1

        pred = self(xt, t)
        loss = ((target - pred) ** 2).mean()

        if self.config.Training.method == "std":
            total_loss = loss

        elif self.config.Training.method == "PINN":
            x1_pred = xt + (1 - t[:, None, None, None, None]) * pred
            divergence = utils.compute_divergence(x1_pred)
            divergence_loss = torch.mean(divergence ** 2)
            total_loss = loss + self.config.Training.divergence_loss_weight * divergence_loss

        elif self.config.Training.method == "PINN_dyn":
            x1_pred = xt + (1 - t[:, None, None, None, None]) * pred
            divergence = utils.compute_divergence(x1_pred)
            divergence_loss = torch.mean(divergence ** 2)
            coef = loss / divergence_loss
            total_loss = loss + coef * divergence_loss

        elif self.config.Training.method == "ConFIG":
            x1_pred = xt + (1 - t[:, None, None, None, None]) * pred
            divergence = utils.compute_divergence(x1_pred)
            divergence_loss = torch.mean(divergence ** 2)
            loss.backward(retain_graph=True)
            grads_1 = get_gradient_vector(self.model, none_grad_mode="skip")
            self.optimizers().zero_grad()
            divergence_loss.backward()
            grads_2 = get_gradient_vector(self.model, none_grad_mode="skip")
            self.operator.update_gradient(self.model, [grads_1, grads_2])
            total_loss = loss + self.config.Training.divergence_loss_weight * divergence_loss
        
        self.log("train_loss", loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x0, x1 = batch
        t = torch.rand(x0.size(0), device=self.device)
        target = x1 - (1 - self.config.Training.sigma_min) * x0
        xt = (1 - t[:, None, None, None, None]) * x0 + t[:, None, None, None, None] * x1

        pred = self(xt, t)
        val_loss = ((target - pred) ** 2).mean()
        self.log("val_loss", val_loss, prog_bar=True)


def train_flow_matching(config):
    print("Loading dataset...")
    dataset_hr = IsotropicTurbulenceDataset(dt=config.Data.dt, grid_size=config.Data.grid_size,
                                            crop=config.Data.crop, seed=config.Data.seed, size=config.Data.size)
    velocity_hr = dataset_hr.velocity
    velocity_lr = utils.interpolate_dataset(velocity_hr, config.Data.perc / 100)

    total_size = len(dataset_hr)
    indices = np.arange(total_size)
    np.random.seed(config.Data.seed)
    np.random.shuffle(indices)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    train_idx, val_idx, _ = indices[:train_size], indices[train_size:train_size+val_size], indices[train_size+val_size:]

    train_lr, val_lr = torch.utils.data.Subset(velocity_lr, train_idx), torch.utils.data.Subset(velocity_lr, val_idx)
    train_hr, val_hr = torch.utils.data.Subset(velocity_hr, train_idx), torch.utils.data.Subset(velocity_hr, val_idx)

    train_loader = create_dataloader(train_lr, train_hr, config.Training.batch_size)
    val_loader = create_dataloader(val_lr, val_hr, config.Training.batch_size)

    model = FlowMatchingModule(config)

    trainer = Trainer(
        max_epochs=config.Training.epochs,
        accelerator="gpu",
        devices=-1,  # Use all available GPUs
        strategy=DDPStrategy(find_unused_parameters=False),
        log_every_n_steps=1
    )

    trainer.fit(model, train_loader, val_loader)