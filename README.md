# STXM Autoencoder Denoising

PyTorch implementation of an unsupervised autoencoder for denoising 3D STXM hyperspectral datasets.

## Why?

Scanning Transmission X-ray Microscopy (STXM) datasets often suffer from:

- low photon statistics
- detector noise
- weak spectral contrast
- limited acquisition time

This project uses a shallow neural autoencoder to recover cleaner spectra while preserving spectral features.

## Features

✓ Unsupervised learning  
✓ No labelled data needed  
✓ Spectral denoising of hyperspectral cubes  
✓ Designed for X-ray spectroscopy / STXM workflows  

## Input

Hyperspectral cube:

```python
(X, Y, E)
```

where:

- `X, Y` = spatial dimensions
- `E` = energy channels

## Quick Start

```python
from denoising import denoise_cube_simple

denoised = denoise_cube_simple(
    cube,
    bottleneck=4,
    num_epochs=300,
    noise_factor=0.05
)
```

## Method

Pipeline:

```text
Raw cube → normalization → noise injection → autoencoder → denoised cube
```

## Applications

- STXM
- XANES imaging
- Synchrotron microscopy
- Hyperspectral imaging

## Scientific caution

Autoencoders may smooth weak spectral signatures.

Always validate denoised spectra against:

- raw spectra
- reference spectra
- known absorption features

## Dependencies

```bash
pip install -r requirements.txt
```

## Author

Faidra Amargianou

