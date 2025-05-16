import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import wandb
import pytorch_lightning as pl

from conflictfree.utils import get_gradient_vector
from conflictfree.grad_operator import ConFIGOperator
from dataset import IsotropicTurbulenceDataset
import utils
from model_simple import Model_base
from my_config_length import UniProjectionLength

wandb.login(key="f4a726b2fe7929990149e82fb88da423cfa74e46")

class FlowMatchingLightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(ignore=['config'])  # Don't log config object itself
        self.config = config
        self.model = Model_base(config)
        self.loss_fn = nn.MSELoss()
        self.operator = ConFIGOperator(length_model=UniProjectionLength()) if config.Training.method == "ConFIG" else None
        self.mse_losses = []
        self.val_losses = []

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=float(self.config.Training.learning_rate))

    def training_step(self, batch, batch_idx):
        x1 = batch
        x0 = torch.randn_like(x1)
        sigma_min = float(self.config.Training.sigma_min)
        target = x1 - (1 - sigma_min) * x0

        t = torch.rand(x1.size(0), device=self.device)
        xt = (1 - (1 - sigma_min) * t[:, None, None, None, None]) * x0 + t[:, None, None, None, None] * x1

        pred = self.model(xt, t)
        loss = ((target - pred) ** 2).mean()

        if self.config.Training.method == "std":
            total_loss = loss

        elif self.config.Training.method == "PINN":
            x1_pred = xt + (1 - t[:, None, None, None, None]) * pred
            divergence_loss = utils.compute_divergence(x1_pred).pow(2).mean()
            total_loss = loss + float(self.config.Training.divergence_loss_weight) * divergence_loss

        elif self.config.Training.method == "PINN_dyn":
            x1_pred = xt + (1 - t[:, None, None, None, None]) * pred
            divergence_loss = utils.compute_divergence(x1_pred).pow(2).mean()
            coef = loss / divergence_loss
            total_loss = loss + coef * divergence_loss

        elif self.config.Training.method == "ConFIG":
            x1_pred = xt + (1 - t[:, None, None, None, None]) * pred
            divergence_loss = utils.compute_divergence(x1_pred).pow(2).mean()
            loss.backward(retain_graph=True)
            grads_1 = get_gradient_vector(self.model, none_grad_mode="skip")
            self.optimizers().zero_grad()
            divergence_loss.backward()
            grads_2 = get_gradient_vector(self.model, none_grad_mode="skip")
            self.operator.update_gradient(self.model, [grads_1, grads_2])
            self.optimizers().step()
            total_loss = loss + float(self.config.Training.divergence_loss_weight) * divergence_loss
            return {"loss": total_loss, "mse": loss}

        self.log("train_loss", loss)
        return {"loss": total_loss, "mse": loss}

    def validation_step(self, batch, batch_idx):
        x1 = batch
        x0 = torch.randn_like(x1)
        target = x1 - x0
        t = torch.rand(x1.size(0), device=self.device)
        xt = (1 - t[:, None, None, None, None]) * x0 + t[:, None, None, None, None] * x1
        pred = self.model(xt, t)
        val_loss = ((target - pred) ** 2).mean()
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

def create_dataloaders(config):
    dataset = IsotropicTurbulenceDataset(dt=config.Data.dt, grid_size=config.Data.grid_size,
                                         crop=config.Data.crop, seed=config.Data.seed, size=config.Data.size)
    velocity = dataset.velocity
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    indices = np.arange(total_size)
    np.random.seed(config.Data.seed)
    np.random.shuffle(indices)
    train_dataset = torch.utils.data.Subset(velocity, indices[:train_size])
    val_dataset = torch.utils.data.Subset(velocity, indices[train_size:train_size + val_size])
    test_dataset = torch.utils.data.Subset(velocity, indices[train_size + val_size:])

    train_loader = DataLoader(train_dataset, batch_size=config.Training.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.Training.batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader

def train(config):
    model = FlowMatchingLightningModule(config)
    train_loader, val_loader = create_dataloaders(config)

    wandb.init(project="fm", config=config)

    trainer = pl.Trainer(
        max_epochs=config.Training.epochs,
        accelerator="gpu",
        devices=-1,  # use all available GPUs
        strategy="ddp",  # DistributedDataParallel
        log_every_n_steps=10
    )

    trainer.fit(model, train_loader, val_loader)
    wandb.finish()

if __name__ == "__main__":
    with open("configs/config_generative.yml", "r") as f:
        config = yaml.safe_load(f)
    config = utils.dict2namespace(config)
    train(config)
