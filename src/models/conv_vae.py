"""
Conv1D-VAE model definition.

Treats a 1-D feature vector as a 1-D signal (channels=1) and applies a
stack of Conv1D layers to capture local correlations between adjacent
MFCC / spectral feature bins.

Classes
-------
ConvVAE: 1-D Convolutional Variational Autoencoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    """Conv1D Variational Autoencoder.

    The feature vector of shape (batch, feat_dim) is unsqueezed to
    (batch, 1, feat_dim) so that Conv1d layers treat it as a 1-channel
    1-D signal. After the final TransposeConv layer an adaptive-avg-pool
    restores the exact original length.

    Parameters
    ----------
    in_dim: Input feature dimensionality.
    z_dim: Latent space dimensionality.
    channels: Tuple of channel widths for each Conv1D block.
    """

    def __init__(self, in_dim: int, z_dim: int, 
                 channels: tuple[int, ...] = (32, 64, 128)) -> None:
        super().__init__()
        self.in_dim = in_dim

        # -- Convolutional Encoder ----------------------------------
        enc_layers: list[nn.Module] = []
        prev = 1
        for ch in channels:
            enc_layers += [
                nn.Conv1d(prev, ch, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(ch),
                nn.LeakyReLU(0.2),
            ]
            prev = ch
        self.conv_enc = nn.Sequential(*enc_layers)

        # Compute flattened size after all conv layers
        dummy = self.conv_enc(torch.zeros(1, 1, in_dim))
        self.flat = int(dummy.view(1, -1).shape[1])

        self.mu_fc = nn.Linear(self.flat, z_dim)
        self.lv_fc = nn.Linear(self.flat, z_dim)

        # -- Convolutional Decoder ---------------------------------
        self.ch0  = channels[-1]
        self.seq0 = self.flat // channels[-1]

        self.fc_dec = nn.Linear(z_dim, self.flat)

        dec_layers: list[nn.Module] = []
        prev = channels[-1]
        for ch in reversed(channels[:-1]):
            dec_layers += [
                nn.ConvTranspose1d(prev, ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(ch),
                nn.LeakyReLU(0.2),
            ]
            prev = ch
        # Final layer maps back to 1 channel (the original signal)
        dec_layers.append(
            nn.ConvTranspose1d(prev, 1, kernel_size=4, stride=2, padding=1)
        )
        self.conv_dec = nn.Sequential(*dec_layers)

    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode (batch, feat_dim) -> (μ, log σ²)."""
        h = self.conv_enc(x.unsqueeze(1)).view(x.size(0), -1)
        return self.mu_fc(h), self.lv_fc(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent z -> reconstruction (batch, feat_dim)."""
        h = self.fc_dec(z).view(z.size(0), self.ch0, self.seq0)
        out = self.conv_dec(h)
        return F.adaptive_avg_pool1d(out, self.in_dim).squeeze(1)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)
        return mu

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, 
                                                torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, mu, log_var, z)."""
        mu, lv = self.encode(x)
        z = self.reparameterize(mu, lv)
        return self.decode(z), mu, lv, z

    def enc(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Uniform latent-extraction interface."""
        return self.encode(x)
