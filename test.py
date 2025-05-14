from dataset import IsotropicTurbulenceDataset
import utils

dataset = IsotropicTurbulenceDataset(crop="crop")

utils.plot_slice(dataset.vorticity_magnitude, 20, 0, 63)

residual = utils.compute_divergence(dataset.velocity[20].unsqueeze(0))
residual = residual.unsqueeze(0)
utils.plot_slice(residual, 0, 0, 63, "residual")