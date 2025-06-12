import pytorch_lightning as pl
import torch
import torch.optim as optim
import wandb
import math
import matplotlib.pyplot as plt
import os

from conflictfree.utils import get_gradient_vector
from conflictfree.grad_operator import ConFIGOperator
from diffusion import Diffusion
import utils
from src.core.models.box.pdedit import PDEDiT3D_B
from my_config_length import UniProjectionLength

# Your Dataset import assumed

class DDPMTrainer(pl.LightningModule):
    def __init__(self, config, dataset):
        super().__init__()
        self.save_hyperparameters()  # saves config automatically
        self.config = config
        self.dataset = dataset

        self.model = PDEDiT3D_B(
            channel_size=config.Model.channel_size,
            channel_size_out=config.Model.channel_size_out,
            drop_class_labels=config.Model.drop_class_labels,
            partition_size=config.Model.partition_size,
            mending=False
        )

        self.diffusion = Diffusion(config)

        # For ConFIG method, initialize operator
        if config.Training.method == "ConFIG":
            self.operator = ConFIGOperator(length_model=UniProjectionLength())
        else:
            self.operator = None

        # Convert learning_rate and weights from strings if needed
        for attr in ["learning_rate", "ddpm_loss_weight", "gamma", "last_lr"]:
            val = getattr(config.Training, attr)
            if isinstance(val, str):
                setattr(config.Training, attr, float(val))

    def training_step(self, batch, batch_idx):
        y = batch.to(self.device)
        batch_size = y.shape[0]
        t = torch.randint(0, self.diffusion.num_timesteps, size=(batch_size,), device=y.device)

        x_t, noise = self.diffusion.forward(y, t)
        e_pred = self.model(x_t, t).sample

        mse_loss = (noise - e_pred).square().mean()

        a_b = self.diffusion.alphas_b[t].view(batch_size, 1, 1, 1).view(-1, 1, 1, 1, 1)
        x0_pred = (x_t - (1 - a_b).sqrt() * e_pred) / a_b.sqrt()
        eq_residual = utils.compute_divergence(self.dataset.data_scaler.inverse(x0_pred[:, :3, :, :, :]),
                                               2 * math.pi / self.config.Data.grid_size)
        eq_res_m = torch.mean(torch.abs(eq_residual))

        method = self.config.Training.method

        if method == "std":
            loss = mse_loss

        elif method == "PINN":
            loss = mse_loss + self.config.Training.ddpm_loss_weight * eq_res_m

        elif method == "PINN_dyn":
            coef = mse_loss / (eq_res_m + 1e-8)  # avoid div by zero
            loss = mse_loss + coef * eq_res_m

        elif method == "ConFIG":
            loss = mse_loss  # initial loss for backward

            # Backprop mse_loss first
            mse_loss.backward(retain_graph=True)

            grads_1 = get_gradient_vector(self.model, none_grad_mode="skip")
            
            self.optimizer.zero_grad()

            # Backprop physics residual loss
            eq_res_m.backward()

            grads_2 = get_gradient_vector(self.model, none_grad_mode="skip")

            self.operator.update_gradient(self.model, [grads_1, grads_2])

            # Apply optimizer step manually here, skip Lightning optimizer step for ConFIG
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Log losses and return
            self.log('train_mse_loss', mse_loss, on_step=True, on_epoch=True)
            self.log('train_eq_res_m', eq_res_m, on_step=True, on_epoch=True)
            return loss

        else:
            raise ValueError(f"Unknown training method: {method}")

        self.log('train_mse_loss', mse_loss, on_step=True, on_epoch=True)
        self.log('train_eq_res_m', eq_res_m, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        y = batch.to(self.device)
        batch_size = y.shape[0]
        t = torch.randint(0, self.diffusion.num_timesteps, size=(batch_size,), device=y.device)

        x_t, noise = self.diffusion.forward(y, t)
        e_pred = self.model(x_t, t).sample
        val_loss = (noise - e_pred).square().mean()

        self.log('val_loss', val_loss, on_step=False, on_epoch=True)

        return val_loss

    def configure_optimizers(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.Training.learning_rate)
        return self.optimizer

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                       optimizer_closure=None, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        # Skip optimizer step for ConFIG since done manually
        if self.config.Training.method == "ConFIG":
            pass
        else:
            optimizer.step()
            optimizer.zero_grad()

    def on_train_epoch_end(self):
        # Custom LR scheduler: decay lr by gamma, no lower than last_lr
        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']
            new_lr = max(current_lr * self.config.Training.gamma, self.config.Training.last_lr)
            param_group['lr'] = new_lr

# Optional: You can define a LightningDataModule for your dataset if you want cleaner data handling

# Then the training script:

if __name__ == "__main__":
    import yaml
    from dataset import BigSpectralIsotropicTurbulenceDataset

    print("Loading config...")
    with open("configs/config_ddpm.yml", "r") as f:
        config = yaml.safe_load(f)
    config = utils.dict2namespace(config)

    print("Loading dataset...")
    dataset = BigSpectralIsotropicTurbulenceDataset(
        grid_size=config.Data.grid_size,
        norm=config.Data.norm,
        size=config.Data.size,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        batch_size=config.Training.batch_size,
        num_samples=10
    )

    model = DDPMTrainer(config=config, dataset=dataset)

    wandb_logger = pl.loggers.WandbLogger(project="ddpm")
    
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

    trainer = pl.Trainer(
        max_epochs=config.Training.epochs,
        accelerator="gpu",
        devices=2,
        strategy="ddp" ,
        logger=wandb_logger,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        # Save checkpoints automatically
        default_root_dir="runs"
    )

    trainer.fit(model, train_dataloaders=dataset.train_loader, val_dataloaders=dataset.val_loader)
