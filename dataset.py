import torch
import numpy as np
import utils 

class IsotropicTurbulenceDataset:
    def __init__(self, dt=0.1, grid_size=128, crop="", field="vorticity", norm=True, shuffle=True, seed=1234, size=None):
        self.dt = dt
        self.grid_size = grid_size
        self.norm = norm
        self.shuffle = shuffle
        self.seed = seed
        self.size = size
        self.field = field
        self.crop = crop
        
        self.data = torch.load(f'data/data_{self.crop}_{dt}_{grid_size}.pt', weights_only=False)
        if isinstance(self.data, np.ndarray):
            self.data = torch.from_numpy(self.data)
        self.N_time, self.N_channels, self.Nx, self.Ny, self.Nz = self.data.shape
        print(f"N_time: {self.N_time}, N_channels: {self.N_channels}, Nx: {self.Nx}, Ny: {self.Ny}, Nz: {self.Nz}")

        self.velocity = self.data[:, :3, :, :, :]
        self.pressure = self.data[:, 3, :, :, :].unsqueeze(1)
        self.vorticity = None
        self.vorticity_magnitude = None
        
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

    def __len__(self):
        return self.N_time




