# \# Denoising STXM Hyperspectral Data with Autoencoders

# 

# This repository provides a PyTorch-based unsupervised denoising autoencoder for 3D STXM hyperspectral image cubes.

# 

# \## Input format

# 

# The input cube has shape:

# 

# (X, Y, E)

# 

# where:

# 

# \- X = spatial dimension

# \- Y = spatial dimension

# \- E = spectral / energy channels

# 

# \## Method

# 

# This workflow performs:

# 

# 1\. Cube reshaping into spectral vectors

# 2\. Percentile normalization

# 3\. Autoencoder training with Gaussian noise injection

# 4\. Spectral reconstruction

# 5\. Reshaping back into image cube

# 

# \## Parameters

# 

# \- cube → input hyperspectral cube

# \- bottleneck → latent dimension

# \- num\_epochs → training epochs

# \- noise\_factor → Gaussian noise level

# 

# \## Output

# 

# Returns a denoised cube with the same shape as the input.

# 

# \## Scientific note

# 

# This method may smooth weak spectral features.

# Always compare denoised spectra with raw spectra and reference spectra.

