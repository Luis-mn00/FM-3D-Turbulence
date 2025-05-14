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
def integrate_ode_and_sample(config, model, num_samples=1, steps=100):
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

    # Save the final generated sample as a plot
    for i, sample in enumerate(samples):
        plt.figure(figsize=(6, 6))
        plt.imshow(sample[0, 0, :, :, sample.size(4) // 2].numpy(), cmap="viridis")
        plt.title(f"Generated Sample {i}")
        plt.colorbar()
        plt.savefig(f"generated_sample_{i}.png")
        plt.close()

    model.train().requires_grad_(True)
    print("Done Sampling")
    return samples

if __name__ == "__main__":
    # Load the configuration
    print("Loading config...")
    with open("configs/config_file.yml", "r") as f:
        config = yaml.safe_load(f)
    config = utils.dict2namespace(config)
    print(config.device)

    # Load the trained model
    print("Loading model...")
    model = load_model(config, config.Model.save_path)

    # Generate samples using ODE integration
    print("Generating samples...")
    num_samples = 3
    samples = integrate_ode_and_sample(config, model, num_samples=num_samples)