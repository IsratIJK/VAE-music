"""
MLP-VAE and Beta-VAE model definitions.

Classes
-------
MLPVAE: Standard fully-connected Variational Autoencoder.
BetaVAE: Identical architecture to MLPVAE; β is passed at training time.
"""

import torch
import torch.nn as nn

from .base_vae import make_mlp


class MLPVAE(nn.Module):
    """Fully-connected (MLP) Variational Autoencoder.

    Encoder: Linear stack → (μ, log σ²)
    Reparameterisation trick: z = μ + ε·σ  (ε ~ N(0,I))
    Decoder: Linear stack → reconstruction

    Parameters
    ----------
    in_dim: Dimensionality of the input feature vector.
    z_dim: Latent space dimensionality.
    h: Tuple of hidden layer widths shared by encoder and decoder.
    """

    def __init__(self, in_dim: int, z_dim: int, 
                 h: tuple[int, ...] = (256, 128, 64)) -> None:
        super().__init__()

        # Encoder: input -> hidden layers (no final activation)
        enc_dims = [in_dim] + list(h)
        self.enc_net = make_mlp(enc_dims)

        # Latent distribution heads
        self.mu_fc = nn.Linear(h[-1], z_dim)
        self.lv_fc = nn.Linear(h[-1], z_dim)

        # Decoder: z -> hidden layers (reversed) -> reconstruction
        dec_dims = [z_dim] + list(reversed(h)) + [in_dim]
        self.dec_net = make_mlp(dec_dims)

    # ------------------------------------------------------------------
    # Core forward methods
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to (μ, log σ²)."""
        h = self.enc_net(x)
        return self.mu_fc(h), self.lv_fc(h)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample z via the reparameterisation trick (training only)."""
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)
        return mu  # deterministic at eval time

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent code z to reconstructed input."""
        return self.dec_net(z)

    def forward(self, x: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass.

        Returns
        -------
        (reconstruction, mu, log_var, z)
        """
        mu, lv = self.encode(x)
        z = self.reparameterize(mu, lv)
        return self.decode(z), mu, lv, z

    # Uniform API used by extract_latent
    def enc(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Alias for encode - provides a uniform interface across all models."""
        return self.encode(x)


class BetaVAE(MLPVAE):
    """Beta-VAE: identical architecture to MLPVAE.

    The β penalty on the KL term (β > 1) is applied at training time via
    ``vae_loss_fn(..., beta=BETA_VAE_B)``, not inside the model itself.
    This class exists as a distinct type so experiment tracking can clearly
    label it as "Beta-VAE".
    """
    pass
