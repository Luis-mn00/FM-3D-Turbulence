import matplotlib.pyplot as plt
import torch
import numpy as np
import argparse
import random
from scipy.interpolate import griddata
#from LSIM.distance_model import DistanceModel
from typing import Optional
import ot as pot
from functools import partial
import math
import torch.nn as nn
from LSIM_3D.src.volsim.distance_model import *
import vedo
from torchfsm.operator import Div
from torchfsm.mesh import MeshGrid
from scipy.ndimage import zoom
from scipy.ndimage import laplace
import h5py

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
        mean = self.mean.reshape(*shape).to(x.device)
        std = self.std.reshape(*shape).to(x.device)
        return (x - mean) / std

    def inverse(self, x):
        shape = [1, -1] + [1] * (x.ndim - 2)
        mean = self.mean.reshape(*shape).to(x.device)
        std = self.std.reshape(*shape).to(x.device)
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
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Plot saved in {filename}")
    
def compute_blurriness(tensor):
    """
    Compute variance of 3D Laplacian for each channel.
    tensor: numpy.ndarray of shape (C, Lx, Ly, Lz)
    Returns: numpy.ndarray of shape (C,) with Laplacian variances per channel
    """
    C = tensor.shape[0]
    variances = np.zeros(C)
    for c in range(C):
        lap = laplace(tensor[c])
        variances[c] = lap.var()
    return np.mean(variances)

def compute_divergence(velocity, h):
    assert velocity.shape[1] == 3, "Velocity must have 3 channels (vx, vy, vz)"
    
    """
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
        return (f_pad[tuple(slices_after)] - f_pad[tuple(slices_before)]) / (2 * h)

    vx = velocity[:, 0]
    vy = velocity[:, 1]
    vz = velocity[:, 2]

    dvx_dx = central_diff(vx, 0)
    dvy_dy = central_diff(vy, 1)
    dvz_dz = central_diff(vz, 2)

    divergence = dvx_dx + dvy_dy + dvz_dz
    """
    
    mesh_grid=MeshGrid([(0, 2*torch.pi, 128),(0, 2*torch.pi, 128), (0, 2*torch.pi, 128)])
    velocity = velocity.to(mesh_grid.device)
    div=Div()
    divergence = div(velocity, mesh=mesh_grid)
    
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
    
    random.seed(1234)

    for i in range(n_samples):
        #print(f"sample {i+1}/{n_samples}")
        sampled_ids[i] = np.array(random.sample(range(np.prod(dims)), n_points))
        for c in range(n_channels):
            X_vals[i, c] = interpolate_points(X_vals[i, c], perc=perc, ids=sampled_ids[i], method=method)
    return X_vals, sampled_ids


def downscale_data(high_res, scale_factor):
    channels = len(high_res.shape) == 5  # (N, C, Lx, Ly, Lz)

    if channels:
        N, C, Lx, Ly, Lz = high_res.shape
        high_res = high_res.reshape(N * C, Lx, Ly, Lz)
    else:
        N, Lx, Ly, Lz = high_res.shape

    _high_res = high_res.numpy() if isinstance(high_res, torch.Tensor) else high_res

    Lx_small = int(Lx / scale_factor)
    Ly_small = int(Ly / scale_factor)
    Lz_small = int(Lz / scale_factor)
    NN = _high_res.shape[0]

    X_small = np.zeros((NN, Lx_small, Ly_small, Lz_small), dtype=np.float32)
    X_upscaled = np.zeros((NN, Lx, Ly, Lz), dtype=np.float32)

    for i in range(NN):
        #print(f"sample {i}/{NN}")
        # Downscale
        X_small[i] = zoom(_high_res[i], zoom=(Lx_small / Lx, Ly_small / Ly, Lz_small / Lz), order=0)
        # Upscale
        X_upscaled[i] = zoom(X_small[i], zoom=(Lx / Lx_small, Ly / Ly_small, Lz / Lz_small), order=0)

    if channels:
        X_upscaled = X_upscaled.reshape(N, C, Lx, Ly, Lz)

    return torch.Tensor(X_upscaled)


#lsim_model = DistanceModel(baseType="lsim", isTrain=False, useGPU=False)
#lsim_model.load("LSIM/LSiM.pth")
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
            A_slice = A[0, :3, :, :, z]
            B_slice = B[0, :3, :, :, z]
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

def LSiM_distance_3D(A, B):
    A = A.squeeze(0)
    A = A.permute(1, 2, 3, 0)
    A = A.cpu().numpy()
    B = B.squeeze(0)
    B = B.permute(1, 2, 3, 0)
    B = B.cpu().numpy()
    model_3d = DistanceModel.load("LSIM_3D/models/VolSiM.pth", useGPU=False)
    dist = model_3d.computeDistance(A, B, normalize=True, interpolate=False)
    
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
        
def spectral_derivative_3d(V):
    N, C, H, W, D = V.size()

    # Generate frequency grids
    h = np.fft.fftfreq(H, 1. / H)
    w = np.fft.fftfreq(W, 1. / W)
    d = np.fft.fftfreq(D, 1. / D)
    mesh_h, mesh_w, mesh_d = np.meshgrid(h, w, d, indexing='ij')

    # Convert to torch tensors
    mesh_h = torch.tensor(mesh_h, device=V.device, dtype=V.dtype)
    mesh_w = torch.tensor(mesh_w, device=V.device, dtype=V.dtype)
    mesh_d = torch.tensor(mesh_d, device=V.device, dtype=V.dtype)

    # Perform FFT
    V_fft = torch.fft.fftn(V, dim=(-3, -2, -1))

    # Multiply by i * k to compute derivative in Fourier domain
    I = 1j
    dV_dh_fft = I * mesh_h * V_fft
    dV_dw_fft = I * mesh_w * V_fft
    dV_dd_fft = I * mesh_d * V_fft

    # Inverse FFT to get real-valued derivatives
    dV_dh = torch.fft.ifftn(dV_dh_fft, dim=(-3, -2, -1)).real
    dV_dw = torch.fft.ifftn(dV_dw_fft, dim=(-3, -2, -1)).real
    dV_dd = torch.fft.ifftn(dV_dd_fft, dim=(-3, -2, -1)).real

    # Stack derivatives along new dimension
    dV = torch.stack([dV_dh, dV_dw, dV_dd], dim=2)

    return dV


def physics(A_model, A_target):
    # continuity = [None, None]
    S_ijS_ij_m = [None, None]
    R_ijR_ij_m = [None, None]
    SijSkjSji_m = [None, None]
    VortexStret_m = [None, None]
    A_model = A_model[:, :3, :, :, :]
    A_target = A_target[:, :3, :, :, :]
    for i, A in enumerate([A_model, A_target]):
        A11, A22, A33 = A[:, 0, 0], A[:, 1, 1], A[:, 2, 2]
        # continuity[i] = (A11 + A22 + A33).mean()
        S = 0.5 * (A + A.transpose(1, 2))
        R = 0.5 * (A - A.transpose(1, 2))
        S_ijS_ij = (S * S).sum(dim=[1, 2])
        R_ijR_ij = (R * R).sum(dim=[1, 2])
        S_ijS_ij_m[i] = S_ijS_ij.mean()
        R_ijR_ij_m[i] = R_ijR_ij.mean()

        S = S.permute(0, 3, 4, 5, 1, 2).reshape(-1, 3, 3)
        R = R.permute(0, 3, 4, 5, 1, 2).reshape(-1, 3, 3)
        SijSkjSji = torch.sum(torch.matmul(S, S) * S, axis=(1, 2))
        Omega = torch.empty((*R.size()[:-1], 1), device=R.device)
        Omega[:, 0, 0] = 2 * R[:, 2, 1]
        Omega[:, 1, 0] = 2 * R[:, 0, 2]
        Omega[:, 2, 0] = 2 * R[:, 1, 0]
        VS_3d = torch.matmul(S, Omega)
        VortexStret = torch.matmul(Omega.transpose(1, 2), VS_3d)
        SijSkjSji_m[i] = SijSkjSji.mean()
        VortexStret_m[i] = (-3 / 4) * VortexStret.mean()

    weight = torch.tensor([1, 1, 1, 1], device=A_model.device)
    output = 0
    for i, item in enumerate([S_ijS_ij_m, R_ijR_ij_m, SijSkjSji_m, VortexStret_m]):
        output += (item[1] - item[0]).abs() * weight[i]

    return output


def weighted_mse_loss(input, target, weight=(2. * torch.ones(3, 3)).fill_diagonal_(1)):
    loss_ = nn.functional.mse_loss(input, target, reduction='none')
    return sum([(weight[i, j] * loss_[:, i, j]).mean() for i in range(3) for j in range(3)])

def visualize_3d_cloud_volume(
    volume_data: torch.Tensor | np.ndarray,
    title: str = "3D Cloud Volume Rendering",
    bg_color: str = 'black',
    scalars_min: float = None,
    scalars_max: float = None
):
    """
    Visualizes a 3D scalar volumetric field as a cloud-like volume rendering
    using vedo. The color and density of the cloud indicate the data values.

    Args:
        volume_data (torch.Tensor or np.ndarray):
            The 3D scalar field data. Expected shape (D, H, W) or (1, D, H, W).
            If a PyTorch tensor, it will be moved to CPU and converted to NumPy.
        title (str): Title for the visualization window.
        bg_color (str): Background color of the plot ('black', 'white', etc.).
        scalars_min (float, optional): Optional minimum value for mapping data
            to the transfer function. If None, uses data_np.min().
        scalars_max (float, optional): Optional maximum value for mapping data
            to the transfer function. If None, uses data_np.max().
    """
    # Ensure vedo is installed
    try:
        import vedo
    except ImportError:
        print("Error: vedo library not found. Please install it using: pip install vedo")
        return

    # Convert input to a 3D NumPy array (D, H, W)
    if isinstance(volume_data, torch.Tensor):
        # Remove batch dimension if present and move to CPU, then convert to NumPy
        if volume_data.ndim == 4 and volume_data.shape[0] == 1:
            data_np = volume_data.squeeze(0).detach().cpu().numpy()
        elif volume_data.ndim == 3:
            data_np = volume_data.detach().cpu().numpy()
        else:
            raise ValueError(f"Input tensor has unsupported shape: {volume_data.shape}. Expected (D, H, W) or (1, D, H, W).")
    elif isinstance(volume_data, np.ndarray):
        if volume_data.ndim == 4 and volume_data.shape[0] == 1:
            data_np = volume_data.squeeze(0)
        elif volume_data.ndim == 3:
            data_np = volume_data
        else:
            raise ValueError(f"Input numpy array has unsupported shape: {volume_data.shape}. Expected (D, H, W) or (1, D, H, W).")
    else:
        raise TypeError("Input data must be a torch.Tensor or numpy.ndarray.")

    # Determine data range for normalization
    data_min = scalars_min if scalars_min is not None else data_np.min()
    data_max = scalars_max if scalars_max is not None else data_np.max()

    if data_max == data_min:
        print("Warning: All values in the volume data are identical. Cannot create meaningful visualization.")
        return

    # Normalize data to [0, 1] for consistent transfer function mapping
    # This step is crucial if your data's actual range can vary significantly.
    normalized_data_np = (data_np - data_min) / (data_max - data_min)

    # Create a vedo Volume object
    vol = vedo.Volume(normalized_data_np)

    # Define the custom transfer function for "cloud-like" rendering
    # These scalar values correspond to the normalized data range [0, 1]
    scalars = [0.0, 0.05, 0.15, 0.4, 0.7, 1.0]

    # Opacities (0.0 = fully transparent, 1.0 = fully opaque)
    # Creates the cloud effect: fuzzy at edges, denser in core
    opacities = [0.0, 0.005, 0.05, 0.2, 0.5, 0.8]

    # Colors (define the colormap from cool to hot)
    colors = [
        'lightskyblue', # Lowest values (most transparent blue)
        'cyan',
        'lime',
        'yellow',
        'orange',
        'red'           # Highest values (most opaque red)
    ]

    # Build the Lookup Table (LUT)
    transfer_function = vedo.build_lut(scalars, opacities, colors)

    # Apply the transfer function to the volume
    vol.cmap(transfer_function)
    vol.mode('composite') # Ensure composite rendering

    # Set up the plotter and visualize
    plotter = vedo.Plotter(size=(900, 900), bg=bg_color)

    # Add scalar bar for reference
    # Use the original data range for the scalar bar labels
    vol.add_scalarbar(f"Field Value ({data_min:.2f} to {data_max:.2f})", c='white' if bg_color == 'black' else 'black')

    # Add lighting for better depth perception
    vol.lighting('plastic')

    # Add the volume to the plotter
    plotter.add(vol)

    # Set an initial camera position. You can adjust these values
    # The default view often works well, but this provides a consistent starting point.
    center = np.array(data_np.shape) / 2
    plotter.camera.SetPosition([center[0]*2.5, center[1]*2.5, center[2]*2.5]) # View from outside
    plotter.camera.SetFocalPoint(center) # Look at the center of the volume
    plotter.camera.SetViewUp([0, 1, 0]) # Keep Y-axis up

    # Show the plot
    plotter.show(
        title,
        interactor_style=1, # 1 for TrackballCamera, allows easy navigation
    )

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def compute_energy_spectrum(velocity, name, smooth=True):
    lx = ly = lz = 2.0*math.pi
    
    # Convert to numpy arrays
    if isinstance(velocity, torch.Tensor):
        velocity = velocity.cpu().numpy()
        
    N, _, nx, ny, nz = velocity.shape
    nt = nx * ny * nz
    n = nx
    
    k0x = 2.0*math.pi/lx
    k0y = 2.0*math.pi/ly
    k0z = 2.0*math.pi/lz
    knorm = (k0x + k0y + k0z)/3.0
    kxmax = nx/2
    kymax = ny/2
    kzmax = nz/2
    wave_numbers = knorm*np.arange(0,n)
    
    spectra = []
    
    for i in range(N):
        u = velocity[i, 0]
        v = velocity[i, 1]
        w = velocity[i, 2]

        uh = np.fft.fftn(u) / nt
        vh = np.fft.fftn(v) / nt
        wh = np.fft.fftn(w) / nt

        tkeh = np.zeros((nx,ny,nz))
        tkeh = 0.5 * (uh * np.conj(uh) + vh * np.conj(vh) + wh * np.conj(wh)).real
        
        tke_spectrum = np.zeros(len(wave_numbers))
        for kx in range(nx):
            rkx = kx if kx <= kxmax else kx - nx
            for ky in range(ny):
                rky = ky if ky <= kymax else ky - ny
                for kz in range(nz):
                    rkz = kz if kz <= kzmax else kz - nz
                    rk = np.sqrt(rkx**2 + rky**2 + rkz**2)
                    k = int(np.round(rk))
                    tke_spectrum[k] += tkeh[kx, ky, kz]

        tke_spectrum /= knorm
        tke_spectrum = tke_spectrum[1:]

        if smooth:
            smoothed = movingaverage(tke_spectrum, 5)
            smoothed[0:4] = tke_spectrum[0:4]
            tke_spectrum = smoothed

        spectra.append(tke_spectrum)
        
    # Average over all N snapshots
    tke_spectrum_avg = np.mean(spectra, axis=0)   
    wave_numbers = wave_numbers[1:] 
    
    # Analytical line
    C = 1.6          
    eps = 0.103        
    E_k_analytic = C * (eps ** (2/3)) * (wave_numbers ** (-5/3))

    
    # Plot
    plt.figure(figsize=(8, 5))
    plt.loglog(wave_numbers, tke_spectrum_avg + 1e-20)
    plt.loglog(wave_numbers, E_k_analytic, 'k--', label=r"$1.6 \, \varepsilon^{2/3} \, k^{-5/3}$")
    plt.xlabel("Wavenumber $k$")
    plt.ylabel("Energy $E(k)$")
    plt.ylim(1e-7, 1)
    plt.title("Energy Spectrum")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.legend()
    plt.tight_layout()
    output_file = f"generated_plots/{name}.png"
    plt.savefig(output_file)
    
    return tke_spectrum_avg
    
def compute_energy_spectrum_original(file_path: str, name: str, smooth=True):
    lx = ly = lz = 2.0 * math.pi

    with h5py.File(file_path, 'r') as f:
        keys = list(f['sims']['sim0'].keys())[:5]  
        print(f"Found {len(keys)} snapshots.")
        
        spectra = []
        for i, key in enumerate(keys):
            print(f"\n📦 Processing snapshot {i+1}/{len(keys)}: {key}")
            sample = f['sims']['sim0'][key]
            velocity = np.transpose(sample, (3, 0, 1, 2))[:3]  # (3, Nx, Ny, Nz)

            # Add batch dimension: (1, 3, Nx, Ny, Nz)
            velocity = velocity[np.newaxis, ...]

            N, _, nx, ny, nz = velocity.shape
            nt = nx * ny * nz
            n = nx
            
            k0x = 2.0 * math.pi / lx
            k0y = 2.0 * math.pi / ly
            k0z = 2.0 * math.pi / lz
            knorm = (k0x + k0y + k0z) / 3.0
            kxmax = nx / 2
            kymax = ny / 2
            kzmax = nz / 2
            wave_numbers = knorm * np.arange(0, n)

            u, v, w = velocity[0, 0], velocity[0, 1], velocity[0, 2]

            uh = np.fft.fftn(u) / nt
            vh = np.fft.fftn(v) / nt
            wh = np.fft.fftn(w) / nt

            tkeh = 0.5 * (uh * np.conj(uh) + vh * np.conj(vh) + wh * np.conj(wh)).real

            tke_spectrum = np.zeros(len(wave_numbers))
            for kx in range(nx):
                rkx = kx if kx <= kxmax else kx - nx
                for ky in range(ny):
                    rky = ky if ky <= kymax else ky - ny
                    for kz in range(nz):
                        rkz = kz if kz <= kzmax else kz - nz
                        rk = np.sqrt(rkx**2 + rky**2 + rkz**2)
                        k = int(np.round(rk))
                        if k < len(tke_spectrum):
                            tke_spectrum[k] += tkeh[kx, ky, kz]

            tke_spectrum /= knorm
            tke_spectrum = tke_spectrum[1:]  # remove k=0
            if smooth:
                smoothed = movingaverage(tke_spectrum, 5)
                smoothed[0:4] = tke_spectrum[0:4]
                tke_spectrum = smoothed

            spectra.append(tke_spectrum)

    # Average over all snapshots
    tke_spectrum_avg = np.mean(spectra, axis=0)
    wave_numbers = wave_numbers[1:]

    # Kolmogorov analytical line
    C = 1.6          
    eps = 0.103 
    E_k_analytic = C * (eps ** (2 / 3)) * (wave_numbers ** (-5 / 3))

    # Plot
    plt.figure(figsize=(8, 5))
    plt.loglog(wave_numbers, tke_spectrum_avg + 1e-20, label="Averaged TKE Spectrum")
    plt.loglog(wave_numbers, E_k_analytic, 'k--', label=r"$1.6 \, \varepsilon^{2/3} \, k^{-5/3}$")
    plt.xlabel("Wavenumber $k$")
    plt.ylabel("Energy $E(k)$")
    plt.ylim(1e-7, 1)
    plt.title("Energy Spectrum")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.legend()
    plt.tight_layout()

    os.makedirs("generated_plots", exist_ok=True)
    output_file = f"generated_plots/{name}.png"
    plt.savefig(output_file)

    return tke_spectrum_avg

            