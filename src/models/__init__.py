"""Model definitions for VAE-music."""

from .base_vae import make_mlp, vae_loss_fn
from .mlp_vae import MLPVAE, BetaVAE
from .conv_vae import ConvVAE
from .cvae import CVAE
from .autoencoder import Autoencoder

__all__ = [
    "make_mlp",
    "vae_loss_fn",
    "MLPVAE",
    "BetaVAE",
    "ConvVAE",
    "CVAE",
    "Autoencoder",
]
