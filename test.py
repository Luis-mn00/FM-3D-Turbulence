from dataset import IsotropicTurbulenceDataset
import utils

dataset = IsotropicTurbulenceDataset(crop="crop")

utils.plot_slice(dataset.vorticity_magnitude, 20, 0, 63)