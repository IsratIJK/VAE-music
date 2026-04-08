"""Model definitions for VAE-music."""

from .base_vae import make_mlp, vae_loss_fn
from .mlp_vae import MLPVAE, BetaVAE
from .conv_vae import ConvVAE
from .cvae import CVAE
from .autoencoder import Autoencoder
from .gmvae import GMVAE, train_gmvae
from .contrastive_vae import ContrastiveVAE, infonce_loss, make_contrastive_pairs, train_contrastive_vae
from .dann_vae import GradReverse, DANNVAE, dann_lambda_schedule, build_dann_dataset, train_dann_vae

__all__ = [
    "make_mlp",
    "vae_loss_fn",
    "MLPVAE",
    "BetaVAE",
    "ConvVAE",
    "CVAE",
    "Autoencoder",
    "GMVAE",
    "train_gmvae",
    "ContrastiveVAE",
    "infonce_loss",
    "make_contrastive_pairs",
    "train_contrastive_vae",
    "GradReverse",
    "DANNVAE",
    "dann_lambda_schedule",
    "build_dann_dataset",
    "train_dann_vae",
]
