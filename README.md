This repository contains code for Flow Matching Super-Resolution in 3D turbulent data. We can use it to generate new 3D fluid samples or to perform super-resolution.

## Data

The simulation data directly in format `.pt` should be stored inside the data folder `data`. It can be downloaded from this [link](https://syncandshare.lrz.de/getlink/fiC6ZnSbgVPwsmpPE8mQqi/). The file `data_spectral_128.pt` contains the original data from the isotropic turbulent dataset in the [JHTDB](https://turbulence.pha.jhu.edu/Forced_isotropic_turbulence.aspx) downscaled to resolution $128^3$ using spectral interpolation. Since the velocity in this compressed dataset is not divergence free, we modified it slightly to minimize the velocity divergence, which is now stored in `data_spectral_128_mindiv.pt`. Finally, `data_spectral_128_mindiv_5.pt` contains the synthetic sparse fields with only 5% of data, and `data_spectral_128_mindiv_down4.pt` contains the low-resolution data in a grid of size $32^3$ upsampled to $128^3$. 

## Model

Everything related to the used model is inside the folder `src` using the hybrid CNN-Transformer model P3D by ... The model for the autoencoder used in latent versions is implemented in `model_vqvae.py`.

## Usage

All important scripts to train and run experiments are located in the main folder, and its main parameters are listed in cofiguration files inside the folder `configs`. The training scripts are:

- `trainer_ddpm.py`: Train classical diffusion generative models.
- `trainer_direct_route.py`: Train flow matching model to directly model the transition from low-resolution to high-resolution.
- `trainer_fm.py`: Train classical flow matching generative models.
- `trainer_latent_direct_route.py`: Train the latent version of the direct route flow matching.
- `trainer_latent_fm.py`: Train the latent version of classical flow matching.
- `trainer_regression.py`: Train the backbone model with classical regression to learn the map from low-resolution to high-resolution.
- `trainer_vqvae.py`: Train the autoencoder to be used in latent models.

The inference or experiment scripts are:
- `inference_generative_ddpm.py`: Experiments to generate new samples using diffusion.
- `inference_generative.py`: Experiments to generate new samples using flow matching, both with FM sampling and DDIM sampling.
- `inference_latent_direct_route.py`: Experiments to perform super-resolution using the direct route flow matching in the latent version.
- `inference_latent_generative.py`: Experiments to generate new samples using latent flow matching.
- `inference_latent_super.py`: Experiments to perform super-resolution using latent flow matching.
- `inference_regression.py`: Experiments to perform super-resolution using the regression in the backbone model.
- `inference_super_ddpm.py`: Experiments to perform super-resolution using diffusion and different masking strategies.
- `inference_super_direct_route.py`: Experiments to perform super-resolution using the direct route flow matching.
- `inference_super.py`: Experiments to perform super-resolution using the classical flow matching and different masking strategies.

## Extra

Finally, the are some additional files and folders to store the runs, the generated plots, or additional code.