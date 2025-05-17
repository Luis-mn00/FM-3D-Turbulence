import matplotlib.pyplot as plt
import torch
import numpy as np
import argparse
import random
from scipy.interpolate import griddata
from LSIM.distance_model import DistanceModel
from typing import Optional
import ot as pot
from functools import partial
import math

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

class StdScaler(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        shape = [1, -1] + [1] * (x.ndim - 2)
        mean = self.mean.reshape(*shape)
        std = self.std.reshape(*shape)
        return (x - mean) / std

    def inverse(self, x):
        shape = [1, -1] + [1] * (x.ndim - 2)
        mean = self.mean.reshape(*shape)
        std = self.std.reshape(*shape)
        return x * std + mean

    def scale(self):
        return self.std
    
def compute_statistics(data):
    mean = data.mean(axis=(0,2,3,4))
    std = data.std(axis=(0,2,3,4))
    return mean, std
    
def plot_slice(data, snapshot_idx, channel_idx, slice_idx, name=None):
    """
    Plot a slice of the data at a specific snapshot and channel.
    
    Parameters:
    - data: The dataset containing the flow field.
    - snapshot_idx: Index of the time snapshot to plot.
    - channel_idx: Index of the channel to plot.
    - slice_idx: Index of the slice along the z-axis.
    """
    # Extract the specific snapshot, channel, and slice
    snapshot = data[snapshot_idx]  # Shape: (channels, Nx, Ny, Nz)
    channel_data = snapshot[channel_idx]  # Shape: (Nx, Ny, Nz)
    slice_data = channel_data[:, :, slice_idx]  # Shape: (Nx, Ny)

    # Plot the selected slice
    plt.figure(figsize=(8, 6))
    plt.imshow(slice_data, cmap='viridis', origin='lower')
    plt.colorbar(label='Value')
    plt.title(f'Snapshot {snapshot_idx}, Channel {channel_idx}, Slice {slice_idx}')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # Save the plot instead of showing it
    if name is None:
        output_file = f'generated_plots/snapshot_{snapshot_idx}_channel_{channel_idx}_slice_{slice_idx}.png'
    else:
        output_file = f'generated_plots/{name}.png'
    plt.savefig(output_file)
    print(f'Plot saved as {output_file}')
    
def plot_2d_comparison(low_res, high_res, gt, filename):
    plt.figure(figsize=(10, 10))

    plt.subplot(1, 3, 1)
    plt.title("Low-Resolution Input")
    plt.imshow(low_res, cmap="inferno")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Super-Resolved Output")
    plt.imshow(high_res, cmap="inferno")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Ground truth")
    plt.imshow(gt, cmap="inferno")
    plt.axis('off')

    # Save the plot as a PNG file
    plt.tight_layout()
    filename = f"generated_plots/{filename}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved in {filename}")

def compute_divergence(velocity):
    assert velocity.shape[1] == 3, "Velocity must have 3 channels (vx, vy, vz)"
    # Pad for central differences (replicate boundary)
    def central_diff(f, dim):
        # f: (batch, Nx, Ny, Nz)
        pad = [0, 0, 0, 0, 0, 0]  # [z0, z1, y0, y1, x0, x1]
        pad[2 * (2 - dim) + 1] = 1  # after
        pad[2 * (2 - dim)] = 1      # before
        f_pad = torch.nn.functional.pad(f, pad, mode='replicate')
        # Central difference
        slices_before = [slice(None)] * 4
        slices_after = [slice(None)] * 4
        slices_before[dim+1] = slice(0, -2)
        slices_after[dim+1] = slice(2, None)
        return (f_pad[tuple(slices_after)] - f_pad[tuple(slices_before)]) / 2.0

    vx = velocity[:, 0]
    vy = velocity[:, 1]
    vz = velocity[:, 2]

    dvx_dx = central_diff(vx, 0)
    dvy_dy = central_diff(vy, 1)
    dvz_dz = central_diff(vz, 2)

    divergence = dvx_dx + dvy_dy + dvz_dz
    return divergence

def upsample(data_lr, factor=2):
    data_lr = data_lr.float()
    N_time, N_channels, D, H, W = data_lr.shape
    data_lr_reshaped = data_lr.view(N_time * N_channels, 1, D, H, W)
    data_lr_upsampled = torch.nn.functional.interpolate(data_lr_reshaped, size=(D*factor, H*factor, W*factor), mode='nearest')
    velocity_lr_to_hr = data_lr_upsampled.view(N_time, N_channels, D*factor, H*factor, W*factor)
    print(f"velocity_lr upsampled to: {velocity_lr_to_hr.shape}")
    
    return velocity_lr_to_hr

def interpolate_points(image, perc=0, ids=None, method="nearest"):
    # Support both 2D and 3D images
    if image.ndim == 2:
        Nx, Ny = image.shape
        if ids is None:
            sampled_ids = random.sample(range(Nx * Ny), int(Nx * Ny * perc))
        else:
            sampled_ids = ids
        vals = np.tile(image.reshape(Nx * Ny)[sampled_ids], 9)
        ids = [[(x // Ny), (x % Ny)] for x in sampled_ids] + \
              [[(x // Ny), Ny + (x % Ny)] for x in sampled_ids] + \
              [[(x // Ny), 2*Ny + (x % Ny)] for x in sampled_ids] + \
              [[Nx + (x // Ny), (x % Ny)] for x in sampled_ids] + \
              [[Nx + (x // Ny), Ny + (x % Ny)] for x in sampled_ids] + \
              [[Nx + (x // Ny), 2*Ny + (x % Ny)] for x in sampled_ids] + \
              [[2*Nx + (x // Ny), (x % Ny)] for x in sampled_ids] + \
              [[2*Nx + (x // Ny), Ny + (x % Ny)] for x in sampled_ids] + \
              [[2*Nx + (x // Ny), 2*Ny + (x % Ny)] for x in sampled_ids]
        grid_x, grid_y = np.mgrid[0:Nx*3, 0:Ny*3]
        grid_z = griddata(ids, vals, (grid_x, grid_y), method=method, fill_value=0)
        return torch.tensor(grid_z[Nx:Nx*2, Ny:Ny*2])
    elif image.ndim == 3:
        Nx, Ny, Nz = image.shape
        if ids is None:
            sampled_ids = random.sample(range(Nx * Ny * Nz), int(Nx * Ny * Nz * perc))
        else:
            sampled_ids = ids
        # Get the (x, y, z) coordinates for sampled indices
        coords = np.array([(idx // (Ny*Nz), (idx % (Ny*Nz)) // Nz, idx % Nz) for idx in sampled_ids])
        vals = image.reshape(-1)[sampled_ids]
        # Interpolate onto a dense grid
        grid_x, grid_y, grid_z = np.mgrid[0:Nx, 0:Ny, 0:Nz]
        interp = griddata(coords, vals, (grid_x, grid_y, grid_z), method=method, fill_value=0)
        return torch.tensor(interp)
    else:
        raise ValueError("Input image must be 2D or 3D.")

def interpolate_dataset(dataset, perc, method="nearest"):
    X_vals = dataset.cpu().clone() if type(dataset) is torch.Tensor else dataset.copy()
    n_samples = dataset.shape[0]
    n_channels = dataset.shape[1]
    dims = dataset.shape[2:]
    n_points = int(np.prod(dims) * perc)
    sampled_ids = np.zeros((n_samples, n_points), dtype=np.int32)

    for i in range(n_samples):
        if i % 100 == 0:
            print(f"Interpolating sample {i} of {n_samples}")
        sampled_ids[i] = np.array(random.sample(range(np.prod(dims)), n_points))
        for c in range(n_channels):
            X_vals[i, c] = interpolate_points(X_vals[i, c], perc=perc, ids=sampled_ids[i], method=method)
    return X_vals, sampled_ids


lsim_model = DistanceModel(baseType="lsim", isTrain=False, useGPU=False)
lsim_model.load("LSIM/LSiM.pth")
def LSiM_distance(A, B):
    # https://github.com/tum-pbs/LSIM
    # Expected input sizes: [1, 3, 256, 256], [3, 256, 256]  or [256,256]
    assert len(A.shape) == len(B.shape)
    global lsim_model

    # Handle 5D input: (batch, channels, Nx, Ny, Nz)
    # Still not okay. LSiM is not prepared for 3D data
    if len(A.shape) == 5:
        # Assume batch size 1 for generative inference
        assert A.shape[0] == 1 and B.shape[0] == 1, "Batch size > 1 not supported for 3D LSiM_distance."
        n_channels = A.shape[1]
        Nx, Ny, Nz = A.shape[2:]
        total_dist = 0.0
        for z in range(Nz):
            # For each z-slice, shape (channels, Nx, Ny)
            A_slice = A[0, :, :, :, z]
            B_slice = B[0, :, :, :, z]
            total_dist += LSiM_distance(A_slice, B_slice)
        return total_dist

    if len(A.shape) == 4:
        A = A[0]
        B = B[0]

    if A.shape[0] == 3:
        return np.mean([
            LSiM_distance(A[0], B[0]),
            LSiM_distance(A[1], B[1]),
            LSiM_distance(A[2], B[2])
        ])

    if len(A.shape) == 2:
        A = A.unsqueeze(-1)
    if len(B.shape) == 2:
        B = B.unsqueeze(-1)
    A = A.cpu() if type(A) is torch.Tensor else A
    B = B.cpu() if type(B) is torch.Tensor else B
    dist = lsim_model.computeDistance(A, B)
    return dist[0]

def diffuse_mask(value_ids, A=1, sig=0.044, search_dist=-1, N=256, Nx=256, Ny=256, Nz=None, tol=1e-6):
    """
    Create a 2D or 3D diffuse mask with Gaussian spread around value_ids.
    If Nz is None, defaults to 2D (Nx, Ny). If Nz is given, mask is 3D (Nx, Ny, Nz).
    """
    L = 2 * np.pi
    dx = L / Nx
    dy = L / Ny
    if Nz is not None:
        dz = L / Nz
        grid = np.zeros((Nx, Ny, Nz))
        # Set boundaries to 1
        grid[0, :, :] = 1
        grid[-1, :, :] = 1
        grid[:, 0, :] = 1
        grid[:, -1, :] = 1
        grid[:, :, 0] = 1
        grid[:, :, -1] = 1

        def gauss3d(x0, y0, z0, x, y, z):
            return A * np.exp(-((x0 - x)**2 + (y0 - y)**2 + (z0 - z)**2) / (2 * sig**2))

        if search_dist < 0:
            min_search_steps = 0
            while gauss3d(0, 0, 0, dx*min_search_steps, 0, 0) > tol:
                min_search_steps += 1
            search_dist = min_search_steps

        S = search_dist * 2 + 1
        gaussian = np.zeros((S, S, S))
        x0 = y0 = z0 = search_dist * dx
        for i in range(S):
            for j in range(S):
                for k in range(S):
                    gaussian[i, j, k] = gauss3d(x0, y0, z0, i*dx, j*dy, k*dz)

        for sid in value_ids:
            i = sid // (Ny * Nz)
            j = (sid % (Ny * Nz)) // Nz
            k = sid % Nz

            ilb = max(0, i - search_dist)
            iub = min(Nx, i + search_dist + 1)
            jlb = max(0, j - search_dist)
            jub = min(Ny, j + search_dist + 1)
            klb = max(0, k - search_dist)
            kub = min(Nz, k + search_dist + 1)

            gilb = max(0, search_dist - i)
            giub = S - max(0, i + search_dist - (Nx - 1))
            gjlb = max(0, search_dist - j)
            gjub = S - max(0, j + search_dist - (Ny - 1))
            gklb = max(0, search_dist - k)
            gkub = S - max(0, k + search_dist - (Nz - 1))

            grid[ilb:iub, jlb:jub, klb:kub] = np.fmax(
                gaussian[gilb:giub, gjlb:gjub, gklb:gkub],
                grid[ilb:iub, jlb:jub, klb:kub]
            )
        return grid
    else:
        # ...existing 2D code...
        # (leave your current 2D implementation here)
        pass
    
# https://github.com/atong01/conditional-flow-matching/blob/c25e1918a80dfacbe9475c055d61ac997f28262a/torchcfm/optimal_transport.py#L218
def wasserstein(
    x0: torch.Tensor,
    x1: torch.Tensor,
    method: Optional[str] = None,
    reg: float = 0.05,
    power: int = 2,
    **kwargs,
) -> float:
    """Compute the Wasserstein (1 or 2) distance (wrt Euclidean cost) between a source and a target
    distributions.

    Parameters
    ----------
    x0 : Tensor, shape (bs, *dim)
        represents the source minibatch
    x1 : Tensor, shape (bs, *dim)
        represents the source minibatch
    method : str (default : None)
        Use exact Wasserstein or an entropic regularization
    reg : float (default : 0.05)
        Entropic regularization coefficients
    power : int (default : 2)
        power of the Wasserstein distance (1 or 2)
    Returns
    -------
    ret : float
        Wasserstein distance
    """
    assert power == 1 or power == 2
    # ot_fn should take (a, b, M) as arguments where a, b are marginals and
    # M is a cost matrix
    if method == "exact" or method is None:
        ot_fn = pot.emd2
    elif method == "sinkhorn":
        ot_fn = partial(pot.sinkhorn2, reg=reg)
    else:
        raise ValueError(f"Unknown method: {method}")

    a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
    if x0.dim() > 2:
        x0 = x0.reshape(x0.shape[0], -1)
    if x1.dim() > 2:
        x1 = x1.reshape(x1.shape[0], -1)
    M = torch.cdist(x0, x1)
    if power == 2:
        M = M**2
    ret = ot_fn(a, b, M.detach().cpu().numpy(), numItermax=int(1e7))
    if power == 2:
        ret = math.sqrt(ret)
    return ret

def init_weights(model):
    """
    Set weight initialization for Conv3D in network.
    Based on: https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073/24
    """
    if isinstance(model, torch.nn.Conv3d):
        torch.nn.init.xavier_uniform_(model.weight)
        torch.nn.init.constant_(model.bias, 0)
        # torch.nn.init.zeros_(model.bias)