import torch
import numpy as np
import utils 
from torch.utils.data import DataLoader, Dataset, TensorDataset
import h5py
from numpy.fft import fftn, ifftn, fftshift, ifftshift

class BigSpectralIsotropicTurbulenceDataset:
    def __init__(self, grid_size=128, norm=True, size=None, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, batch_size=32, num_samples=10):
        self.grid_size = grid_size
        self.norm = norm
        self.size = size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.num_samples = num_samples
        
        self.data = torch.load(f'data/data_spectral_{grid_size}_mindiv.pt', weights_only=False, map_location='cpu')
        if isinstance(self.data, np.ndarray):
            self.data = torch.from_numpy(self.data)
        self.N_time, self.N_channels, self.Nx, self.Ny, self.Nz = self.data.shape
        print(f"N_time: {self.N_time}, N_channels: {self.N_channels}, Nx: {self.Nx}, Ny: {self.Ny}, Nz: {self.Nz}")

        self.velocity = self.data[:, :3, :, :, :]
        
        mean_data, std_data = utils.compute_statistics(self.data)
        self.data_scaler = utils.StdScaler(mean_data, std_data)
        
        if self.norm:
            self.data = self.data_scaler(self.data)
                
        if self.size is not None:
            self.N_time = self.size
            self.data = self.data[:self.size]
            
        indices = torch.randperm(self.size, generator=torch.Generator().manual_seed(1234))
                
        train_size = int(self.train_ratio * self.N_time)
        val_size = int(self.val_ratio * self.N_time)
        test_size = self.N_time - train_size - val_size
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        train_dataset = torch.utils.data.Subset(self.data, train_indices)
        val_dataset = torch.utils.data.Subset(self.data, val_indices)
        test_dataset = torch.utils.data.Subset(self.data, test_indices)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        self.test_dataset = test_dataset[:self.num_samples]
    
    def __len__(self):
        return self.size
    
class SupervisedSpectralTurbulenceDataset:
    def __init__(self, grid_size=128, norm=True, size=None, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, batch_size=32, num_samples=10):
        self.grid_size = grid_size
        self.norm = norm
        self.size = size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.num_samples = num_samples

        # Load X (inputs) and Y (ground truth outputs)
        self.X = torch.load(f'data/data_spectral_{grid_size}_mindiv_5.pt', weights_only=False)
        #self.X = torch.load(f'data/data_spectral_{grid_size}_mindiv_down4.pt', weights_only=False) 
        self.Y = torch.load(f'data/data_spectral_{grid_size}_mindiv.pt', weights_only=False)

        if isinstance(self.X, np.ndarray):
            self.X = torch.from_numpy(self.X)
        if isinstance(self.Y, np.ndarray):
            self.Y = torch.from_numpy(self.Y)
            
        self.X = self.X.to('cpu')
        self.Y = self.Y.to('cpu')
        
        self.Y = self.Y[:, :3, :, :, :]

        self.N_time, self.N_channels, self.Nx, self.Ny, self.Nz = self.Y.shape
        print(f"N_time: {self.N_time}, N_channels: {self.N_channels}, Nx: {self.Nx}, Ny: {self.Ny}, Nz: {self.Nz}")

        mean_Y, std_Y = utils.compute_statistics(self.Y)
        self.Y_scaler = utils.StdScaler(mean_Y, std_Y)

        mean_X, std_X = utils.compute_statistics(self.X)
        self.X_scaler = utils.StdScaler(mean_X, std_X)

        if self.norm:
            self.X = self.X_scaler(self.X)
            self.Y = self.Y_scaler(self.Y)

        if self.size is not None:
            self.X = self.X[:self.size]
            self.Y = self.Y[:self.size]
            self.N_time = self.size
        else:
            self.size = self.N_time

        # Create consistent random split
        indices = torch.randperm(self.size, generator=torch.Generator().manual_seed(1234))
        train_size = int(self.train_ratio * self.size)
        val_size = int(self.val_ratio * self.size)
        test_size = self.size - train_size - val_size

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        # Create paired (X, Y) datasets
        train_dataset = TensorDataset(self.X[train_indices], self.Y[train_indices])
        val_dataset = TensorDataset(self.X[val_indices], self.Y[val_indices])
        test_dataset = TensorDataset(self.X[test_indices], self.Y[test_indices])

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # Sampled test set for visualization or evaluation
        self.test_dataset = test_dataset[:self.num_samples]

    def __len__(self):
        return self.size
    
    
    

class IsotropicTurbulenceDataset:
    def __init__(self, dt=0.1, grid_size=128, crop="", field="vorticity", norm=True, shuffle=True, seed=1234, size=None, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, batch_size=32, num_samples=10):
        self.dt = dt
        self.grid_size = grid_size
        self.norm = norm
        self.shuffle = shuffle
        self.seed = seed
        self.size = size
        self.field = field
        self.crop = crop
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.num_samples = num_samples
        
        self.data = torch.load(f'data/data_{self.crop}_{dt}_{grid_size}.pt', weights_only=False)
        if isinstance(self.data, np.ndarray):
            self.data = torch.from_numpy(self.data)
        self.N_time, self.N_channels, self.Nx, self.Ny, self.Nz = self.data.shape
        print(f"N_time: {self.N_time}, N_channels: {self.N_channels}, Nx: {self.Nx}, Ny: {self.Ny}, Nz: {self.Nz}")

        self.velocity = self.data[:, :3, :, :, :]
        self.pressure = self.data[:, 3, :, :, :].unsqueeze(1)
        self.vorticity = None
        self.vorticity_magnitude = None
        
        mean_data, std_data = utils.compute_statistics(self.data)
        print(f"Data mean: {mean_data}, std: {std_data}")
        self.data_scaler = utils.StdScaler(mean_data, std_data)
        
        if field == "vorticity":
            dvx_dy = np.gradient(self.velocity[:, 0, :, :, :], axis=2)
            dvx_dz = np.gradient(self.velocity[:, 0, :, :, :], axis=3)
            dvy_dx = np.gradient(self.velocity[:, 1, :, :, :], axis=1)
            dvy_dz = np.gradient(self.velocity[:, 1, :, :, :], axis=3)
            dvz_dx = np.gradient(self.velocity[:, 2, :, :, :], axis=1)
            dvz_dy = np.gradient(self.velocity[:, 2, :, :, :], axis=2)

            curl_x = dvz_dy - dvy_dz
            curl_y = dvx_dz - dvz_dx
            curl_z = dvy_dx - dvx_dy

            self.vorticity = torch.tensor(np.stack((curl_x, curl_y, curl_z), axis=1), dtype=torch.float32)
            self.vorticity_magnitude = torch.sqrt(self.vorticity[:, 0, :, :, :]**2 + self.vorticity[:, 1, :, :, :]**2 + self.vorticity[:, 2, :, :, :]**2).unsqueeze(1)

        print(f"Velocity shape: {self.velocity.shape}")
        print(f"Pressure shape: {self.pressure.shape}")
        print(f"Vorticity shape: {self.vorticity.shape if self.vorticity is not None else 'N/A'}")
        print(f"Vorticity magnitude shape: {self.vorticity_magnitude.shape if self.vorticity_magnitude is not None else 'N/A'}")
        
        if self.norm:
            mean_velocity, std_velocity = utils.compute_statistics(self.velocity)
            print(f"Velocity mean: {mean_velocity}, std: {std_velocity}")
            self.velocity_scaler = utils.StdScaler(mean_velocity, std_velocity)
            self.velocity = self.velocity_scaler(self.velocity)
            
            mean_pressure, std_pressure = utils.compute_statistics(self.pressure)
            print(f"Pressure mean: {mean_pressure}, std: {std_pressure}")
            self.pressure_scaler = utils.StdScaler(mean_pressure, std_pressure)
            self.pressure = self.pressure_scaler(self.pressure)
            
            if self.vorticity is not None:
                mean_vorticity, std_vorticity = utils.compute_statistics(self.vorticity)
                print(f"Vorticity mean: {mean_vorticity}, std: {std_vorticity}")
                self.vorticity_scaler = utils.StdScaler(mean_vorticity, std_vorticity)
                self.vorticity = self.vorticity_scaler(self.vorticity)
                
            if self.vorticity_magnitude is not None:
                mean_vorticity_magnitude, std_vorticity_magnitude = utils.compute_statistics(self.vorticity_magnitude)
                print(f"Vorticity magnitude mean: {mean_vorticity_magnitude}, std: {std_vorticity_magnitude}")
                self.vorticity_magnitude_scaler = utils.StdScaler(mean_vorticity_magnitude, std_vorticity_magnitude)
                self.vorticity_magnitude = self.vorticity_magnitude_scaler(self.vorticity_magnitude)
                
        if self.shuffle:
            indices = torch.randperm(self.N_time, generator=torch.Generator().manual_seed(self.seed))
            self.velocity = self.velocity[indices]
            self.pressure = self.pressure[indices]
            if self.vorticity is not None:
                self.vorticity = self.vorticity[indices]
            if self.vorticity_magnitude is not None:
                self.vorticity_magnitude = self.vorticity_magnitude[indices]
                
        if self.size is not None:
            self.N_time = self.size
            self.velocity = self.velocity[:self.size]
            self.pressure = self.pressure[:self.size]
            if self.vorticity is not None:
                self.vorticity = self.vorticity[:self.size]
            if self.vorticity_magnitude is not None:
                self.vorticity_magnitude = self.vorticity_magnitude[:self.size]
            indices = torch.arange(self.N_time)
                
        train_size = int(self.train_ratio * self.N_time)
        val_size = int(self.val_ratio * self.N_time)
        test_size = self.N_time - train_size - val_size
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        #self.data = torch.cat((self.velocity, self.pressure), dim=1)
        self.data = self.velocity
        
        train_dataset = torch.utils.data.Subset(self.data, train_indices)
        val_dataset = torch.utils.data.Subset(self.data, val_indices)
        test_dataset = torch.utils.data.Subset(self.data, test_indices)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        self.train_dataset = train_dataset[:]
        self.test_dataset = test_dataset[:self.num_samples]
    
    def __len__(self):
        return self.size

"""
class BigIsotropicTurbulenceDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, sim_group='sim0', norm=True, size=None, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, batch_size=32, num_samples=10, test=False, grid_size=128):
        self.file_path = file_path
        self.sim_group = sim_group
        self.norm = norm
        self.size = size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.grid_size = grid_size
        
        self.N_time, self.N_channels, self.Nx, self.Ny, self.Nz = 500, 4, grid_size, grid_size, grid_size

        # Open file once to get keys and normalization constants
        with h5py.File(self.file_path, 'r') as f:
            self.indices = list(f['sims'][self.sim_group].keys())
            self.fields_max = f['norm_fields_sca_max'][:]
            self.fields_min = f['norm_fields_sca_min'][:]
            self.fields_mean = f['norm_fields_sca_mean'][:]
            self.fields_std = f['norm_fields_sca_std'][:]
        if self.size is not None:
            self.indices = self.indices[:self.size]
              
        self.data_scaler = utils.StdScaler(self.fields_mean[:3], self.fields_std[:3])

        # Split indices for train/val/test
        N = len(self.indices)
        train_size = int(self.train_ratio * N)
        val_size = int(self.val_ratio * N)
        test_size = N - train_size - val_size
        train_indices = self.indices[:train_size]
        val_indices = self.indices[train_size:train_size + val_size]
        test_indices = self.indices[train_size + val_size:]

        # Define inner dataset class for on-the-fly loading
        class HDF5SampleDataset(torch.utils.data.Dataset):
            def __init__(self, file_path, sim_group, indices, norm, fields_mean, fields_std, grid_size):
                self.file_path = file_path
                self.sim_group = sim_group
                self.indices = indices
                self.norm = norm
                self.fields_mean = fields_mean
                self.fields_std = fields_std
                self.grid_size = grid_size
            def __len__(self):
                return len(self.indices)
            def spectral_resize_3d(self, img, target_size):
                # img: (D, H, W), target_size: int
                original_shape = np.array(img.shape)
                scale_factor = np.prod(original_shape) / (target_size ** 3)  # normalize energy

                F = fftn(img)
                F_shifted = fftshift(F)
                center = original_shape // 2
                half_size = target_size // 2

                # Crop the central part of the spectrum
                cropped = F_shifted[
                    center[0] - half_size:center[0] + half_size,
                    center[1] - half_size:center[1] + half_size,
                    center[2] - half_size:center[2] + half_size
                ]

                cropped_unshifted = ifftshift(cropped)
                resized = ifftn(cropped_unshifted)
                resized = np.real(resized) / scale_factor  # apply normalization

                return resized
            def __getitem__(self, idx):
                with h5py.File(self.file_path, 'r') as f:
                    sample = f['sims'][self.sim_group][self.indices[idx]][:]
                    # Change shape from (512, 512, 512, C) to (C, 512, 512, 512)
                    sample = np.transpose(sample, (3, 0, 1, 2))
                    if self.norm:
                        sample = (sample - self.fields_mean) / self.fields_std
                        
                    sample = sample[:3, :, :, :]
                    # Spectral downsampling for each channel
                    sample_plot = torch.from_numpy(sample)
                    sample_plot = sample_plot.unsqueeze(0)
                    #utils.plot_slice(sample_plot, 0, 0, int(512/2), name="original_sample_before")
                    c, d, h, w = sample.shape
                    gs = self.grid_size
                    sample_ds = np.zeros((c, gs, gs, gs), dtype=np.float32)
                    for ch in range(c):
                        sample_ds[ch] = self.spectral_resize_3d(sample[ch], gs)
                    sample = torch.tensor(sample_ds, dtype=torch.float32)
                return sample

        # General dataloaders
        self.train_loader = DataLoader(HDF5SampleDataset(self.file_path, self.sim_group, train_indices, self.norm, self.fields_mean, self.fields_std, self.grid_size), batch_size=self.batch_size, shuffle=False)
        self.val_loader = DataLoader(HDF5SampleDataset(self.file_path, self.sim_group, val_indices, self.norm, self.fields_mean, self.fields_std, self.grid_size), batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(HDF5SampleDataset(self.file_path, self.sim_group, test_indices, self.norm, self.fields_mean, self.fields_std, self.grid_size), batch_size=self.batch_size, shuffle=False)

        # Define train_dataset and test_dataset as lists of tensors (if you want to load them into memory)
        n_train = int(len(self.indices) * self.train_ratio)
        n_test = int(len(self.indices) * self.test_ratio)
        self.train_dataset = None
        self.test_dataset = None
        if test:
            with h5py.File(self.file_path, 'r') as f:
                test_dataset = []
                for i in range(len(self.indices) - n_test, len(self.indices) - n_test + self.num_samples):
                    print(i)
                    sample = f['sims'][self.sim_group][self.indices[i]][:]
                    sample = np.transpose(sample, (3, 0, 1, 2)).astype(np.float32)
                    # Spectral downsampling for each channel
                    c, d, h, w = sample.shape
                    gs = self.grid_size
                    sample_ds = np.zeros((c, gs, gs, gs), dtype=np.float32)
                    for ch in range(c):
                        F = fftn(sample[ch])
                        F_shifted = fftshift(F)
                        center = np.array(F_shifted.shape) // 2
                        half_size = gs // 2
                        cropped = F_shifted[
                            center[0]-half_size:center[0]+half_size,
                            center[1]-half_size:center[1]+half_size,
                            center[2]-half_size:center[2]+half_size
                        ]
                        cropped_unshifted = ifftshift(cropped)
                        resized = ifftn(cropped_unshifted)
                        sample_ds[ch] = np.real(resized)
                    test_dataset.append(sample_ds)
                self.test_dataset = torch.tensor(np.stack(test_dataset, axis=0), dtype=torch.float32)


"""
class BigIsotropicTurbulenceDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, sim_group='sim0', norm=True, size=None, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, batch_size=32, num_samples=10, test=False, grid_size=128):
        self.file_path = file_path
        self.sim_group = sim_group
        self.norm = norm
        self.size = size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.grid_size = grid_size
        
        self.N_time, self.N_channels, self.Nx, self.Ny, self.Nz = 500, 4, grid_size, grid_size, grid_size

        # Open file once to get keys and normalization constants
        with h5py.File(self.file_path, 'r') as f:
            self.indices = list(f['sims'][self.sim_group].keys())
            self.fields_max = f['norm_fields_sca_max'][:]
            self.fields_min = f['norm_fields_sca_min'][:]
            self.fields_mean = f['norm_fields_sca_mean'][:]
            self.fields_std = f['norm_fields_sca_std'][:]
        if self.size is not None:
            self.indices = self.indices[:self.size]
            
        self.data_scaler = utils.StdScaler(self.fields_mean, self.fields_std)

        # Split indices for train/val/test
        N = len(self.indices)
        train_size = int(self.train_ratio * N)
        val_size = int(self.val_ratio * N)
        test_size = N - train_size - val_size
        train_indices = self.indices[:train_size]
        val_indices = self.indices[train_size:train_size + val_size]
        test_indices = self.indices[train_size + val_size:]

        # Define inner dataset class for on-the-fly loading
        class HDF5SampleDataset(torch.utils.data.Dataset):
            def __init__(self, file_path, sim_group, indices, norm, fields_mean, fields_std, grid_size):
                self.file_path = file_path
                self.sim_group = sim_group
                self.indices = indices
                self.norm = norm
                self.fields_mean = fields_mean
                self.fields_std = fields_std
                self.grid_size = grid_size
            def __len__(self):
                return len(self.indices)
            def __getitem__(self, idx):
                with h5py.File(self.file_path, 'r') as f:
                    sample = f['sims'][self.sim_group][self.indices[idx]][:]
                    # Change shape from (512, 512, 512, C) to (C, 512, 512, 512)
                    sample = np.transpose(sample, (3, 0, 1, 2))
                    #sample = sample[:3, :, :, :]
                    if self.norm:
                        sample = (sample - self.fields_mean) / self.fields_std
                        #sample = (sample - self.fields_mean[:3, :, :, :]) / self.fields_std[:3, :, :, :]
                    # Crop a centered window of size grid_size x grid_size x grid_size
                    #c, d, h, w = sample.shape
                    #gs = self.grid_size
                    #start_d = (d - gs) // 2
                    #start_h = (h - gs) // 2
                    #start_w = (w - gs) // 2
                    #sample = sample[:, start_d:start_d+gs, start_h:start_h+gs, start_w:start_w+gs]
                    #sample = torch.tensor(sample, dtype=torch.float32)
                return sample

        # General dataloaders
        self.train_loader = DataLoader(HDF5SampleDataset(self.file_path, self.sim_group, train_indices, self.norm, self.fields_mean, self.fields_std, self.grid_size), batch_size=self.batch_size, shuffle=False)
        self.val_loader = DataLoader(HDF5SampleDataset(self.file_path, self.sim_group, val_indices, self.norm, self.fields_mean, self.fields_std, self.grid_size), batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(HDF5SampleDataset(self.file_path, self.sim_group, test_indices, self.norm, self.fields_mean, self.fields_std, self.grid_size), batch_size=self.batch_size, shuffle=False)

        # Define train_dataset and test_dataset as lists of tensors (if you want to load them into memory)
        n_train = int(len(self.indices) * self.train_ratio)
        n_test = int(len(self.indices) * self.test_ratio)
        self.train_dataset = None
        self.test_dataset = None
        if test:
            with h5py.File(self.file_path, 'r') as f:
                test_dataset = []
                for i in range(len(self.indices) - n_test, len(self.indices)):
                    print(i)
                    sample = f['sims'][self.sim_group][self.indices[i]][:]
                    sample = np.transpose(sample, (3, 0, 1, 2)).astype(np.float16)
                    #sample = sample[:3, :, :, :]
                    # Crop a centered window of size grid_size x grid_size x grid_size
                    c, d, h, w = sample.shape
                    gs = self.grid_size
                    start_d = (d - gs) // 2
                    start_h = (h - gs) // 2
                    start_w = (w - gs) // 2
                    sample = sample[:, start_d:start_d+gs, start_h:start_h+gs, start_w:start_w+gs]
                    test_dataset.append(sample)
                self.test_dataset = torch.tensor(np.stack(test_dataset, axis=0), dtype=torch.float16)
                self.test_dataset = self.test_dataset[:self.num_samples]
