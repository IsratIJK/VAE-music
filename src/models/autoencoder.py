"""
Deterministic Autoencoder (AE) model definition.

Unlike the VAE variants there is no KL regularisation: the encoder maps
directly to a deterministic bottleneck code.  This serves as a baseline
to show whether probabilistic regularisation helps clustering.

Classes
-------
Autoencoder: Deterministic MLP Autoencoder.
"""

import torch
import torch.nn as nn

from .base_vae import make_mlp


class Autoencoder(nn.Module):
    """Deterministic MLP Autoencoder (no KL term).

    The encoder maps x -> z deterministically (no mean / log-var heads).
    The decoder maps z -> x̂.  Training uses plain MSE reconstruction loss.

    Parameters
    ----------
    in_dim: Input feature dimensionality.
    z_dim: Bottleneck (latent) dimensionality.
    h: Tuple of hidden layer widths for encoder and decoder.
    """

    def __init__(self, in_dim: int, z_dim: int, h: 
                 tuple[int, ...] = (256, 128, 64)) -> None:
        super().__init__()
        enc_dims = [in_dim] + list(h) + [z_dim]
        dec_dims = [z_dim] + list(reversed(h)) + [in_dim]
        self.encoder = make_mlp(enc_dims)
        self.decoder = make_mlp(dec_dims)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, bottleneck_code)."""
        z = self.encoder(x)
        return self.decoder(z), z

    def enc(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Uniform latent-extraction interface.

        Returns (z, None) - the None placeholder keeps the API consistent
        with VAE models that return (mu, log_var).
        """
        return self.encoder(x), None
