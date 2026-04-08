"""
Conditional VAE (CVAE) model definition.

The genre one-hot label vector is concatenated to both encoder input and
decoder input, allowing the model to learn genre-conditioned latent
representations.

At clustering time we use ''encode_unconditional'' (zero condition vector)
so that cluster structure reflects the data geometry rather than the label.

Classes
-------
CVAE: Conditional Variational Autoencoder.
"""

import torch
import torch.nn as nn

from .base_vae import make_mlp


class CVAE(nn.Module):
    """Conditional Variational Autoencoder.

    Architecture
    ------------
    Encoder: [x ‖ c] -> hidden -> (μ, log σ²)
    Decoder: [z ‖ c] -> hidden -> reconstruction

    where c is a one-hot genre label of shape (batch, n_class).

    Parameters
    ----------
    in_dim: Input feature dimensionality.
    n_class: Number of genre classes (length of one-hot condition vector).
    z_dim: Latent space dimensionality.
    h: Tuple of hidden layer widths for encoder / decoder.
    """

    def __init__(self, in_dim: int, n_class: int, z_dim: int, 
                 h: tuple[int, ...] = (256, 128, 64)) -> None:
        super().__init__()
        self.n_class = n_class

        # Encoder receives (audio features || one-hot genre)
        enc_in = in_dim + n_class
        enc_dims = [enc_in] + list(h)
        self.enc_net = make_mlp(enc_dims)
        self.mu_fc = nn.Linear(h[-1], z_dim)
        self.lv_fc = nn.Linear(h[-1], z_dim)

        # Decoder receives (latent z || one-hot genre)
        dec_in = z_dim + n_class
        dec_dims = [dec_in] + list(reversed(h)) + [in_dim]
        self.dec_net = make_mlp(dec_dims)

    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode conditioned on c -> (μ, log σ²)."""
        xc = torch.cat([x, c], dim=1)
        h  = self.enc_net(xc)
        return self.mu_fc(h), self.lv_fc(h)

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Decode z conditioned on c -> reconstruction."""
        return self.dec_net(torch.cat([z, c], dim=1))

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)
        return mu

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full conditional forward pass.

        Returns
        -------
        (reconstruction, mu, log_var, z)
        """
        mu, lv = self.encode(x, c)
        z = self.reparameterize(mu, lv)
        return self.decode(z, c), mu, lv, z

    def encode_unconditional(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode with zero condition vector (unknown genre).

        Used during clustering so that the latent space reflects
        audio structure, not the label.
        """
        c = torch.zeros(x.size(0), self.n_class, device=x.device)
        return self.encode(x, c)

    def enc(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Uniform latent-extraction interface (unconditional)."""
        return self.encode_unconditional(x)
