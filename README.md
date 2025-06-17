# denoising
Denoising STXM data
This Python function applies unsupervised denoising to a 3D hyperspectral data cube using a shallow autoencoder implemented in PyTorch. The cube is assumed to have dimensions (X, Y, E), where X and Y are spatial coordinates and E is the number of spectral channels (energies).

The denoising approach uses the following steps:

Preprocessing:
The 3D cube is reshaped into a 2D array of shape (pixels, energy), and normalized to the 99th percentile to reduce sensitivity to outliers.

Autoencoder Architecture:
A simple fully-connected neural network is defined, consisting of:

An encoder that compresses the spectral dimension to a user-defined bottleneck size.

A decoder that reconstructs the full spectrum from the compressed representation.

Noise Injection and Training:
Gaussian noise is added to the inputs during training (controlled by the noise_factor parameter) to encourage robustness. The model is trained to reconstruct the clean input spectra using mean squared error (MSE) loss.

Denoising:
After training for a specified number of epochs (num_epochs), the trained autoencoder is used to generate denoised spectra, which are then reshaped back to the original (X, Y, E) cube format.

Parameters:
cube: Input hyperspectral data cube (np.ndarray) of shape (X, Y, E).

bottleneck: Number of neurons in the autoencoder’s bottleneck layer (default: 4).

num_epochs: Number of training epochs (default: 300).

noise_factor: Standard deviation of Gaussian noise added during training (default: 0.05).

Returns:
A denoised hyperspectral cube with the same shape as the input.

