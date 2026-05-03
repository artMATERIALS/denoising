import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def denoise_cube_simple(cube, bottleneck=4, num_epochs=300, noise_factor=0.05):
    """
    Apply unsupervised autoencoder-based denoising to 3D hyperspectral cube.

    Args:
        cube (np.ndarray): Input data with shape (X, Y, E)
        bottleneck (int): Bottleneck size for autoencoder
        num_epochs (int): Number of training epochs
        noise_factor (float): Gaussian noise injection factor

    Returns:
        np.ndarray: Denoised data cube with same shape as input
    """
    X, Y, E = cube.shape
    pixels = X * Y
    cube_2d = cube.reshape((pixels, E)).astype(np.float32)

    percentile_99 = np.percentile(cube_2d, 99)
    cube_norm = cube_2d / percentile_99

    inputs = torch.tensor(cube_norm).float()
    targets = torch.tensor(cube_norm).float()

    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    class SpectralAutoencoder(nn.Module):
        def __init__(self, input_dim, bottleneck):
            super(SpectralAutoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, bottleneck),
                nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Linear(bottleneck, 64),
                nn.ReLU(),
                nn.Linear(64, input_dim)
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    model = SpectralAutoencoder(input_dim=E, bottleneck=bottleneck)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        total_loss = 0.0
        for noisy_batch, clean_batch in dataloader:
            noisy_input = noisy_batch + noise_factor * torch.randn_like(noisy_batch)
            output = model(noisy_input)

            loss = criterion(output, clean_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss = {total_loss/len(dataloader):.6f}")

    with torch.no_grad():
        denoised = model(inputs).numpy()

    denoised_rescaled = denoised * percentile_99
    return denoised_rescaled.reshape((X, Y, E))
