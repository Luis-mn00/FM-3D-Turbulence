import torch
from model_simple import Model_base
import matplotlib.pyplot as plt
import os
import yaml

import utils

# Create a folder to save plots
plot_folder = "generated_plots"
os.makedirs(plot_folder, exist_ok=True)

# Load the trained model
def load_model(config, model_path):
    model = Model_base(config)
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model = model.to(config.device)
    model.eval()
    return model

# Integrate ODE and generate samples
def integrate_ode_and_sample(config, model, num_samples=1, steps=10):
    torch.manual_seed(42)
    model.eval().requires_grad_(False)

    samples = []
    for _ in range(num_samples):
        # Initialize random sample
        xt = torch.randn((1, config.Model.in_channels, config.Data.grid_size, config.Data.grid_size, config.Data.grid_size), device=config.device)

        for i, t in enumerate(torch.linspace(0, 1, steps, device=config.device), start=1):
            print(f"Step {i}/{steps}")
            # Predict the flow
            pred = model(xt, t.expand(xt.size(0)))

            # Update xt using the ODE integration step
            xt = xt + (1 / steps) * pred

        # Only store the final generated sample
        samples.append(xt.cpu().detach())
        
    return samples

# Linear Beta Schedule (from beta_min to beta_max over the T timesteps)
def get_linear_beta_schedule(T, beta_min=1e-4, beta_max=0.02):
    betas = torch.linspace(beta_min, beta_max, T)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas_cumprod

# Convert velocity to noise epsilon (based on flow matching)
def velocity_to_epsilon(v, x, t, alpha_cum_t):
    alpha_t = alpha_cum_t.sqrt ()
    sigma_t = (1 - alpha_cum_t).sqrt()
    delta = 1e-12
    eps = (v + x / (alpha_t + delta)) / (delta + 1 + sigma_t / (alpha_t + delta))
    return eps
    
# DDIM sampling (using reverse diffusion with flow matching)
def ddim(x, model, t_start, reverse_steps, betas, alphas_cumprod):
    seq = list(range(t_start, 0, -t_start // reverse_steps))
    next_seq = [-1] + seq[:-1]
    n = x.size(0)  # Batch size

    for i, j in zip(reversed(seq), reversed(next_seq)):
        t = torch.full((n,), i / t_start, dtype=torch.float, device=x.device)  # Normalize time to [1, 0]
        t = 1 - t  # Invert to match FM
        print(f"Step {i}/{t_start}, Time: {t[0].item():.4f}")

        alpha_bar_t = alphas_cumprod[i] if i < len(alphas_cumprod) else alphas_cumprod[-1]
        alpha_bar_next = alphas_cumprod[j] if 0 <= j < len(alphas_cumprod) else alpha_bar_t
        
        # Predict velocity v_theta(x_t, t) using the model
        v = model(x, t)

        # Convert velocity to noise epsilon
        e = velocity_to_epsilon(v, x, t, alpha_bar_t)

        # Classic DDIM x0 prediction and update
        x0_pred = (x - e * (1 - alpha_bar_t).sqrt()) / alpha_bar_t.sqrt()
        x = alpha_bar_next.sqrt() * x0_pred + (1 - alpha_bar_next).sqrt() * e

        # Free memory of intermediate tensors
        del v, e, x0_pred
        torch.cuda.empty_cache()

    return x

# Generate samples using the denoising model
def generate_samples_with_denoiser(model, num_samples, t_start, reverse_steps, T):
    # Get the linear beta schedule
    betas, alphas_cumprod = get_linear_beta_schedule(T)

    samples = []
    for _ in range(num_samples):
        # Ensure the input tensor is in float format to match the model's parameters
        x = torch.randn((1, model.config.model.in_channels, model.config.data.image_size, model.config.data.image_size), device=model.config.device).float()
        
        # Perform denoising using DDIM
        sample = ddim(x, model, t_start, reverse_steps, betas, alphas_cumprod)
        samples.append(sample.cpu().detach())
        
    return samples

if __name__ == "__main__":
    # Load the configuration
    print("Loading config...")
    with open("configs/config_generative.yml", "r") as f:
        config = yaml.safe_load(f)
    config = utils.dict2namespace(config)
    print(config.device)

    # Load the trained model
    print("Loading model...")
    model = load_model(config, config.Model.save_path)

    # Generate samples using ODE integration
    print("Generating samples...")
    num_samples = 1
    samples = integrate_ode_and_sample(config, model, num_samples=num_samples)
    
    for i, sample in enumerate(samples):
        utils.plot_slice(sample, 0, 1, 63, f"generated_sample_{i}")
        
    # Generate samples using the denoising model
    samples = generate_samples_with_denoiser(model, num_samples, t_start=1000, reverse_steps=100, T=1000)
    
    for i, sample in enumerate(samples):
        utils.plot_slice(sample, 0, 1, 63, f"generated_sample_diff_{i}")