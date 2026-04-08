"""
Shared building blocks used by all VAE model variants.

Functions
---------
make_mlp: Build a sequential MLP from a list of layer widths.
vae_loss_fn: β-VAE ELBO loss (reconstruction MSE + β x KL divergence).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_mlp(dims: list[int], activation: type = nn.LeakyReLU, dropout: float = 0.2, 
             bn: bool = True) -> nn.Sequential:
    """Build a sequential MLP from a list of widths.

    BatchNorm and Dropout are applied between all hidden layers but NOT on the
    final output layer, preserving the raw linear activation there.

    Parameters
    ----------
    dims: List of layer widths, e.g. [512, 256, 128, 32].
                 First element is input dim, last is output dim.
    activation: Activation class (default: LeakyReLU with slope 0.2).
    dropout: Dropout probability between hidden layers (0 = disabled).
    bn: Whether to insert BatchNorm1d after each linear layer.

    Returns
    -------
    nn.Sequential of Linear -> [BN] -> Activation -> [Dropout] blocks.
    """
    layers: list[nn.Module] = []
    prev = dims[0]
    for d in dims[1:]:
        layers.append(nn.Linear(prev, d))
        is_last = d == dims[-1]
        if bn and not is_last:
            layers.append(nn.BatchNorm1d(d))
        if not is_last:
            # LeakyReLU requires the negative_slope arg
            if activation is nn.LeakyReLU:
                layers.append(activation(0.2))
            else:
                layers.append(activation())
            if dropout:
                layers.append(nn.Dropout(dropout))
        prev = d
    return nn.Sequential(*layers)


def vae_loss_fn(recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor, 
                beta: float = 1.0) -> tuple[torch.Tensor, float, float]:
    """Compute the β-VAE ELBO loss.

    Loss = Reconstruction (MSE) + β x KL-divergence

    The KL term encourages the posterior q(z|x) to stay close to the
    unit Gaussian prior p(z) = N(0, I).  β > 1 increases disentanglement
    pressure (Beta-VAE).

    Parameters
    ----------
    recon: Reconstructed input tensor (batch_size, input_dim).
    x: Original input tensor (batch_size, input_dim).
    mu: Latent mean (batch_size, latent_dim).
    log_var: Latent log-variance (batch_size, latent_dim).
    beta: Weight on the KL term (1.0 = standard VAE).

    Returns
    -------
    (total_loss, recon_loss_scalar, kl_loss_scalar)
    """
    recon_loss = F.mse_loss(recon, x, reduction="sum") / x.size(0)
    kl_loss = (
        -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)
    )
    total = recon_loss + beta * kl_loss
    return total, recon_loss.item(), kl_loss.item()
