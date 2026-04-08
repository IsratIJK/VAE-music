"""
Gaussian Mixture VAE (GMVAE).

Replaces the standard N(0, I) prior with a K-component Gaussian Mixture.
Learned mixture components serve as genre prototypes, enabling joint
learning of cluster assignments and latent representation.

Classes
-------
GMVAE: Gaussian Mixture VAE model.

Functions
---------
train_gmvae: Train a GMVAE on scaled feature matrix.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class GMVAE(nn.Module):
    """Gaussian Mixture VAE with K-component learned prior.

    Architecture
    ------------
    Encoder: Linear stack (in_dim -> h) -> (μ, log σ², component weights)
    Prior: Learnable mixture means and log-variances (K, z_dim)
    Decoder: Linear stack (z_dim -> reversed(h) -> in_dim)

    The ELBO uses an upper-bound KL approximation:
        KL ≈ Σ_k q(y_k|x) · KL(q(z|x) ∥ p(z|y_k))
    plus an entropy term to encourage balanced component usage.

    Parameters
    ----------
    in_dim: Input feature dimensionality.
    z_dim: Latent code dimensionality.
    n_components: Number of Gaussian mixture components (= n_genres).
    h: Tuple of hidden layer sizes.
    """

    def __init__(self, in_dim: int, z_dim: int, n_components: int, 
                 h: tuple[int, ...] = (256, 128, 64)) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.K = n_components

        # -- Encoder -------------------------------------------------------
        prev = in_dim
        enc_layers: list[nn.Module] = []
        for hd in h:
            enc_layers += [
                nn.Linear(prev, hd), nn.BatchNorm1d(hd),
                nn.LeakyReLU(0.2), nn.Dropout(0.2)
            ]
            prev = hd
        self.enc_net = nn.Sequential(*enc_layers)
        self.mu_fc = nn.Linear(prev, z_dim)
        self.lv_fc = nn.Linear(prev, z_dim)

        # -- Soft component assignment head -------------------------------------------
        self.qy_net = nn.Sequential(
            nn.Linear(prev, 128), nn.ReLU(),
            nn.Linear(128, n_components), nn.Softmax(dim=1)
        )

        # -- Learnable mixture prior parameters ----------------------------------------
        self.mu_prior = nn.Parameter(torch.randn(n_components, z_dim) * 0.5)
        self.lv_prior = nn.Parameter(torch.zeros(n_components, z_dim))

        # -- Decoder -------------------------------------------------------
        prev = z_dim
        dec_layers: list[nn.Module] = []
        for hd in reversed(h):
            dec_layers += [
                nn.Linear(prev, hd), nn.BatchNorm1d(hd),
                nn.LeakyReLU(0.2), nn.Dropout(0.2)
            ]
            prev = hd
        dec_layers.append(nn.Linear(prev, in_dim))
        self.dec_net = nn.Sequential(*dec_layers)

    # -- Forward helpers -------------------------------------------------------

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode input to (μ, log σ², component weights)."""
        h = self.enc_net(x)
        mu = self.mu_fc(h)
        lv = self.lv_fc(h)
        qy = self.qy_net(h)   # (B, K) soft component probabilities
        return mu, lv, qy

    def reparameterize(self, mu: torch.Tensor, lv: torch.Tensor) -> torch.Tensor:
        """Reparameterisation trick; deterministic at inference."""
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec_net(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, 
                                                torch.Tensor, torch.Tensor, 
                                                torch.Tensor]:
        mu, lv, qy = self.encode(x)
        z = self.reparameterize(mu, lv)
        return self.decode(z), mu, lv, qy, z

    def enc(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Latent extraction interface (compatible with extract_latent)."""
        mu, _, _ = self.encode(x)
        return mu, None

    # -- Loss -------------------------------------------------------

    def gmvae_loss(self, recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, 
                   lv: torch.Tensor, qy: torch.Tensor, beta: float = 1.0) -> tuple[torch.Tensor, 
                                                                                   float, float]:
        """GMVAE ELBO loss.

        Returns
        -------
        (loss, recon_loss_item, kl_item)
        """
        # Reconstruction loss
        rl = F.mse_loss(recon, x, reduction="sum") / x.size(0)

        # KL(q(z|x) ∥ Σ_k q(y_k) N(μ_k, σ_k²)) - upper-bound via:
        # KL ≈ Σ_k q(y_k) · KL(N(μ_q, σ_q) ∥ N(μ_k, σ_k))
        mu_pr = self.mu_prior.unsqueeze(0)   # (1, K, z)
        lv_pr = self.lv_prior.unsqueeze(0)   # (1, K, z)
        mu_q = mu.unsqueeze(1)   # (B, 1, z)
        lv_q = lv.unsqueeze(1)   # (B, 1, z)

        kl_k = 0.5 * torch.sum(
            lv_pr - lv_q
            + (lv_q.exp() + (mu_q - mu_pr).pow(2)) / (lv_pr.exp() + 1e-8)
            - 1,
            dim=2,
        )  # (B, K)

        kl = (qy * kl_k).sum(dim=1).mean()

        # Entropy regulariser: encourages balanced component usage
        ent = -(qy * (qy + 1e-8).log()).sum(dim=1).mean()

        loss = rl + beta * kl - 0.1 * ent
        return loss, float(rl.item()), float(kl.item())


def train_gmvae(X_sc: np.ndarray, n_components: int, latent_dim: int, 
                h: tuple[int, ...] = (256, 128, 64), epochs: int = 100, 
                lr: float = 1e-3, beta: float = 1.0, batch_size: int = 256, 
                device: torch.device | None = None) -> tuple[GMVAE, list[float], float]:
    """Train GMVAE on a scaled feature matrix.

    Parameters
    ----------
    X_sc: Scaled feature matrix (N, feat_dim).
    n_components: Number of Gaussian mixture components.
    latent_dim: Latent code dimensionality.
    h: Encoder/decoder hidden layer sizes.
    epochs: Training epochs.
    lr: AdamW learning rate.
    beta: β weight on KL term.
    batch_size: Mini-batch size.
    device: Target device.

    Returns
    -------
    (model, history, best_loss)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GMVAE(X_sc.shape[1], latent_dim, n_components, h=h).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_sc)),
        batch_size=batch_size, shuffle=True, drop_last=False
    )

    best_loss: float = float("inf")
    best_state: dict | None = None
    history: list[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_total = 0.0
        n_batches = 0

        for (bx,) in loader:
            bx = bx.to(device)
            opt.zero_grad()
            recon, mu, lv, qy, _ = model(bx)
            loss, _, _ = model.gmvae_loss(recon, bx, mu, lv, qy, beta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_total += loss.item()
            n_batches += 1

        sched.step()
        avg = epoch_total / n_batches
        history.append(avg)

        if avg < best_loss:
            best_loss = avg
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 25 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{epochs}  loss={avg:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, best_loss
