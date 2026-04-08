"""
Contrastive VAE — InfoNCE + β-VAE.

Combines a standard VAE ELBO with an InfoNCE (NT-Xent / SimCLR) contrastive
loss applied through a separate projection head, so the contrastive signal
tightens genre clusters without distorting the latent space used downstream.

Reference: Chen et al. "A Simple Framework for Contrastive Learning" (2020).

Classes
-------
ContrastiveVAE: VAE with projection head for InfoNCE loss.

Functions
---------
infonce_loss: NT-Xent contrastive loss on projected pairs.
make_contrastive_pairs: Build DataLoader of same-genre positive pairs.
train_contrastive_vae: Train ContrastiveVAE with λ-weighted InfoNCE.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from src.models.base_vae import vae_loss_fn


class ContrastiveVAE(nn.Module):
    """VAE with a SimCLR-style projection head for contrastive learning.

    Parameters
    ----------
    in_dim: Input feature dimensionality.
    z_dim: Latent code dimensionality.
    h: Encoder/decoder hidden layer sizes.
    proj_dim: Projection head output dimension for InfoNCE.
    temperature: InfoNCE temperature τ (lower -> tighter clusters).
    """

    def __init__(self, in_dim: int, z_dim: int, h: tuple[int, ...] = (256, 128, 64), 
                 proj_dim: int = 64, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

        # -- Encoder ----------------------------------------------
        prev = in_dim
        enc_layers: list[nn.Module] = []
        for hd in h:
            enc_layers += [
                nn.Linear(prev, hd), nn.BatchNorm1d(hd),
                nn.LeakyReLU(0.2), nn.Dropout(0.2),
            ]
            prev = hd
        self.enc_net = nn.Sequential(*enc_layers)
        self.mu_fc = nn.Linear(prev, z_dim)
        self.lv_fc = nn.Linear(prev, z_dim)

        # -- Projection head (for InfoNCE only; not used for downstream tasks)
        self.proj = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, proj_dim)
        )

        # -- Decoder -------------------------------------------------
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

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc_net(x)
        return self.mu_fc(h), self.lv_fc(h)

    def reparameterize(self, mu: torch.Tensor, lv: torch.Tensor) -> torch.Tensor:
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec_net(z)

    def project(self, z: torch.Tensor) -> torch.Tensor:
        """L2-normalised projection — used for InfoNCE only."""
        return F.normalize(self.proj(z), dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, 
                                                torch.Tensor, torch.Tensor, 
                                                torch.Tensor]:
        mu, lv = self.encode(x)
        z = self.reparameterize(mu, lv)
        return self.decode(z), mu, lv, z, self.project(z)

    def enc(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Latent extraction interface (compatible with extract_latent)."""
        return self.encode(x)


# ------------------------------------------------------------------------------
# InfoNCE (NT-Xent) loss
# ------------------------------------------------------------------------------

def infonce_loss(p_i: torch.Tensor, p_j: torch.Tensor, 
                 temperature: float = 0.07) -> torch.Tensor:
    """NT-Xent contrastive loss (SimCLR).

    Positive pairs: (p_i[k], p_j[k]) — two augmented views of sample k.
    Negative pairs: all other samples in the batch.

    L = -log[ exp(sim(i,j)/τ) / Σ_{k≠i} exp(sim(i,k)/τ) ]

    Parameters
    ----------
    p_i, p_j: (B, proj_dim) L2-normalised projection vectors.
    temperature: Softmax temperature τ.
    """
    B = p_i.size(0)
    p = torch.cat([p_i, p_j], dim=0)  # (2B, proj_dim)
    sim = torch.mm(p, p.T) / temperature   # (2B, 2B)

    # Mask self-similarity
    mask = torch.eye(2 * B, dtype=torch.bool, device=p.device)
    sim.masked_fill_(mask, -1e9)

    # Positive pair targets: i <-> i+B
    pos_i = torch.arange(B, 2 * B, device=p.device)
    pos_j = torch.arange(0, B, device=p.device)
    targets = torch.cat([pos_i, pos_j], dim=0)   # (2B,)

    return F.cross_entropy(sim, targets)


# ------------------------------------------------------------------------------
# Positive-pair DataLoader builder
# ------------------------------------------------------------------------------

def make_contrastive_pairs(X_sc: np.ndarray, y_labels: np.ndarray, 
                           batch_size: int = 256,) -> tuple[DataLoader, 
                                                            torch.Tensor, 
                                                            LabelEncoder]:
    """Build a DataLoader that yields (idx_i, idx_j, y) same-genre positive pairs.

    For each sample i, a random different sample j from the same genre
    is selected as its positive pair.

    Parameters
    ----------
    X_sc: Scaled feature matrix (N, feat_dim).
    y_labels: String genre labels (N,).
    batch_size: Mini-batch size.

    Returns
    -------
    (loader, X_tensor, label_encoder)
    """
    X_t = torch.FloatTensor(X_sc)
    le = LabelEncoder()
    y_i = le.fit_transform(y_labels)
    y_t = torch.LongTensor(y_i)

    # Pre-build per-class index lists
    class_idx: dict[int, np.ndarray] = {
        int(cls): np.where(y_i == cls)[0]
        for cls in np.unique(y_i)
    }

    rng = np.random.default_rng(42)
    pairs_i: list[int] = []
    pairs_j: list[int] = []

    for idx in range(len(X_sc)):
        cls = int(y_i[idx])
        pos_idx = class_idx[cls]
        choices = pos_idx[pos_idx != idx]
        if len(choices) == 0:
            choices = pos_idx   # fallback: allow same sample
        j = int(rng.choice(choices))
        pairs_i.append(idx)
        pairs_j.append(j)

    ds = TensorDataset(
        torch.LongTensor(pairs_i),
        torch.LongTensor(pairs_j),
        y_t
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    return loader, X_t, le


# ------------------------------------------------------------------------------
# Training function
# ------------------------------------------------------------------------------

def train_contrastive_vae(X_sc: np.ndarray, y_labels: np.ndarray, latent_dim: int, 
                          h: tuple[int, ...] = (256, 128, 64), epochs: int = 100, 
                          lr: float = 1e-3, beta: float = 1.0, lam: float = 0.5, 
                          temperature: float = 0.07, batch_size: int = 256, 
                          device: torch.device | None = None) -> tuple[ContrastiveVAE, 
                                                                       dict[str, list[float]], 
                                                                       float]:
    """Train Contrastive VAE.

    Total loss = ELBO + λ · InfoNCE

    Parameters
    ----------
    X_sc: Scaled feature matrix (N, feat_dim).
    y_labels: String genre labels for building positive pairs.
    latent_dim: Latent code dimensionality.
    h: Hidden layer sizes.
    epochs: Training epochs.
    lr: AdamW learning rate.
    beta: β weight on KL term of ELBO.
    lam: Weight of InfoNCE loss relative to ELBO.
    temperature: InfoNCE temperature τ.
    batch_size: Mini-batch size.
    device: Target device.

    Returns
    -------
    (model, history_dict, best_loss)
      history_dict: Keys "total", "elbo", "infonce" — per-epoch averages.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ContrastiveVAE(X_sc.shape[1], latent_dim, h=h, temperature=temperature).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    loader, X_t, _ = make_contrastive_pairs(X_sc, y_labels, batch_size)

    best_loss: float = float("inf")
    best_state: dict | None = None
    history: dict[str, list[float]] = {"total": [], "elbo": [], "infonce": []}

    print(f"    Training ContrastiveVAE (λ={lam:.2f}, τ={temperature:.2f}, β={beta:.1f})")

    for epoch in range(1, epochs + 1):
        model.train()
        et = ee = ec = nb = 0.0

        for (idx_i, idx_j, _) in loader:
            x_i = X_t[idx_i].to(device)
            x_j = X_t[idx_j].to(device)
            opt.zero_grad()

            # Forward both views
            recon_i, mu_i, lv_i, _, p_i = model(x_i)
            recon_j, mu_j, lv_j, _, p_j = model(x_j)

            # Average ELBO over both views
            elbo_i, _, _ = vae_loss_fn(recon_i, x_i, mu_i, lv_i, beta)
            elbo_j, _, _ = vae_loss_fn(recon_j, x_j, mu_j, lv_j, beta)
            elbo = (elbo_i + elbo_j) * 0.5

            # InfoNCE contrastive loss
            nce = infonce_loss(p_i, p_j, temperature)

            loss = elbo + lam * nce
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            et += loss.item(); ee += elbo.item(); ec += nce.item()
            nb += 1

        sched.step()
        avg_t = et / nb
        history["total"].append(avg_t)
        history["elbo"].append(ee / nb)
        history["infonce"].append(ec / nb)

        if avg_t < best_loss:
            best_loss = avg_t
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 25 == 0 or epoch == 1:
            print(f"    Ep {epoch:3d}/{epochs}  total={avg_t:.4f}  "
                  f"elbo={ee/nb:.4f}  infonce={ec/nb:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, best_loss
