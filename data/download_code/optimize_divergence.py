import numpy as np
import torch 
import torch.optim as optim
from torchfsm.operator import Div
from torchfsm.mesh import MeshGrid
from torchfsm.plot import plot_3D_field
import matplotlib.pyplot as plt

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset = torch.load(f'data/data_spectral_128.pt', weights_only=False)
print(dataset.shape)

if isinstance(dataset, np.ndarray):
    dataset = torch.from_numpy(dataset)
    
optimized_samples = torch.zeros((500, 3, 128, 128, 128), device=device)

mesh_grid=MeshGrid([(0, 2*torch.pi, 128),(0, 2*torch.pi, 128), (0, 2*torch.pi, 128)], device=device)
div=Div()

# Iterate over all samples in the dataset
num_iterations = int(2e+4)
for idx in range(dataset.shape[0]):
    print(f"Processing sample {idx+1}/{dataset.shape[0]}...")
    
    # Extract the sample and prepare it for optimization
    sample = dataset[idx, :3, :, :, :].unsqueeze(0).to(device).requires_grad_(True)
    optimizer = optim.Adam([sample], lr=5e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.6)
    
    # Perform optimization
    for i in range(num_iterations):
        optimizer.zero_grad()  
        divergence = div(sample, mesh=mesh_grid)
        loss = torch.mean(torch.abs(divergence))
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        #if (i + 1) % 1000 == 0:
        #    print(f"Iteration {i+1}/{num_iterations}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    
    # Store the optimized sample
    optimized_samples[idx] = sample.detach().squeeze(0)

# Save the optimized tensor
torch.save(optimized_samples, "data/data_spectral_128_mindiv.pt")
print("Optimized samples saved as optimized_samples.pt")
        