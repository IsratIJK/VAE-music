"""
Domain Adversarial Neural Network VAE (DANN-VAE).

Trains a shared VAE on multiple datasets simultaneously. A gradient reversal
layer forces the encoder to be domain-invariant — the encoder cannot predict
which dataset a sample came from, proving the latent space captures universal
audio structure.

Reference: Ganin et al. "Domain-Adversarial Training of Neural Networks" (2016).

Classes
-------
GradReverse: Gradient reversal autograd function.
DANNVAE: Domain adversarial VAE.

Functions
---------
dann_lambda_schedule: Progressive ramp-up of domain reversal strength.
build_dann_dataset: Align multiple datasets to a common feature dimension.
train_dann_vae: Train DANN-VAE on combined multi-domain data.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.models.base_vae import vae_loss_fn


# ------------------------------------------------------------------------------
# Gradient Reversal Layer
# ------------------------------------------------------------------------------

class GradReverse(torch.autograd.Function):
    """Forward: identity. Backward: multiply gradient by −λ.

    This forces the encoder to be confused about domain — minimising the
    domain classification loss with reversed gradients effectively maximises
    domain confusion in the encoder.
    """

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor, 
                lam: float) -> torch.Tensor:
        ctx.lam = lam  # type: ignore[attr-defined]
        return x.clone()

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad: torch.Tensor) -> tuple:
        return -ctx.lam * grad, None   # type: ignore[attr-defined]


def grad_reverse(x: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    """Apply gradient reversal to tensor x."""
    return GradReverse.apply(x, lam)


# ------------------------------------------------------------------------------
# DANN-VAE model
# ------------------------------------------------------------------------------

class DANNVAE(nn.Module):
    """Domain Adversarial VAE.

    Architecture
    ------------
    Shared encoder (VAE) -> z -> decoder (reconstruction)
                              z -> GradReverse -> domain classifier

    Parameters
    ----------
    in_dim: Input feature dimensionality (same across all domains after alignment).
    z_dim: Latent code dimensionality.
    n_domains: Number of source datasets (e.g. 3 for FMA/LMD/GTZAN).
    h: Hidden layer sizes.
    """

    def __init__(self, in_dim: int, z_dim: int, n_domains: int, 
                 h: tuple[int, ...] = (256, 128, 64)) -> None:
        super().__init__()
        self.n_domains = n_domains

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

        # -- Domain classifier (applied after gradient reversal) ----------------------
        self.domain_clf = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_domains)
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc_net(x)
        return self.mu_fc(h), self.lv_fc(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec_net(z)

    def reparameterize(self, mu: torch.Tensor, lv: torch.Tensor) -> torch.Tensor:
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return mu

    def forward(self, x: torch.Tensor, lam_d: float = 1.0) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, lv = self.encode(x)
        z = self.reparameterize(mu, lv)
        recon = self.decode(z)
        z_rev = grad_reverse(z, lam=lam_d)
        d_logit = self.domain_clf(z_rev)
        return recon, mu, lv, z, d_logit

    def enc(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Latent extraction interface (compatible with extract_latent)."""
        return self.encode(x)


# ------------------------------------------------------------------------------
# Lambda schedule
# ------------------------------------------------------------------------------

def dann_lambda_schedule(epoch: int, total_epochs: int, gamma: float = 10.0) -> float:
    """Gradually increase domain reversal strength from ~0 to 1.

    Starts near 0 (pure VAE training) and ramps to 1.0 (full adversarial).
    Uses the original schedule from Ganin et al. 2016.
    """
    p = epoch / total_epochs
    return float(2.0 / (1.0 + np.exp(-gamma * p)) - 1.0)


# ------------------------------------------------------------------------------
# Dataset alignment
# ------------------------------------------------------------------------------

def build_dann_dataset(ALL: dict[str, dict], datasets_raw: list[tuple], 
                       common_dim: int = 32) -> tuple[
                           np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple]]:
    """Align all datasets to a common feature dimension via PCA.

    Parameters
    ----------
    ALL: Results dict from main experiments; keys are dataset short names.
                   Must contain ALL[ds_key]['X_sc'] (scaled feature matrix).
    datasets_raw: List of (X_raw, y_labels, lang_labels, ds_name) tuples.
    common_dim: Target feature dimension after PCA alignment.

    Returns
    -------
    (X_all, d_all, y_all, lang_all, ds_info)
      X_all: (N_total, common_dim) combined feature matrix.
      d_all: (N_total,) domain integer labels.
      y_all: (N_total,) genre string labels.
      lang_all: (N_total,) language string labels.
      ds_info: [(ds_key, domain_id, le, y_true, n_samples), …]
    """
    X_parts: list[np.ndarray] = []
    d_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    lang_parts: list[np.ndarray] = []
    ds_info: list[tuple] = []

    for d_id, (_, y_labels, lang_labels, ds_name) in enumerate(datasets_raw):
        ds_key = ds_name.split()[0]
        X_sc = ALL[ds_key]["X_sc"]

        # PCA-align to common_dim
        pca_d = min(common_dim, X_sc.shape[1])
        X_pca = PCA(n_components=pca_d, random_state=42).fit_transform(X_sc)
        if pca_d < common_dim:
            pad = np.zeros((X_pca.shape[0], common_dim - pca_d), dtype=np.float32)
            X_pca = np.hstack([X_pca, pad])

        X_pca_sc = StandardScaler().fit_transform(X_pca).astype(np.float32)
        d_labels = np.full(len(X_pca_sc), d_id, dtype=np.int64)

        X_parts.append(X_pca_sc)
        d_parts.append(d_labels)
        y_parts.append(y_labels)
        lang_parts.append(lang_labels)
        ds_info.append((
            ds_key, d_id,
            ALL[ds_key]["le"],
            ALL[ds_key]["y_true"],
            len(X_pca_sc)
        ))

    X_all = np.vstack(X_parts).astype(np.float32)
    d_all = np.concatenate(d_parts)
    y_all = np.concatenate(y_parts)
    lang_all = np.concatenate(lang_parts)

    print(f"  Combined dataset: {X_all.shape} | Common dim: {common_dim} | Domains: {len(ds_info)}")
    return X_all, d_all, y_all, lang_all, ds_info


# ------------------------------------------------------------------------------
# Training function
# ------------------------------------------------------------------------------

def train_dann_vae(X_sc: np.ndarray, d_labels: np.ndarray, latent_dim: int, 
                   h: tuple[int, ...] = (256, 128, 64), epochs: int = 100, 
                   lr: float = 1e-3, beta: float = 1.0, lam_domain: float = 0.5, 
                   batch_size: int = 256, device: torch.device | None = None) \
                      -> tuple[DANNVAE, dict[str, list[float]], float]:
    """Train DANN-VAE on combined multi-domain dataset.

    Total loss = ELBO + λ_d · DomainCrossEntropy
    Gradient reversal in the domain path means minimising domain loss
    actually maximises domain confusion in the encoder.

    Parameters
    ----------
    X_sc: (N, common_dim) aligned feature matrix.
    d_labels: (N,) integer domain labels (0 = FMA, 1 = LMD, 2 = GTZAN).
    latent_dim: Latent code dimensionality.
    h: Hidden layer sizes.
    epochs: Training epochs.
    lr: AdamW learning rate.
    beta: β weight on KL term.
    lam_domain: Weight of domain adversarial loss.
    batch_size: Mini-batch size.
    device: Target device.

    Returns
    -------
    (model, history_dict, best_loss)
      history_dict: Keys "total", "elbo", "domain" — per-epoch averages.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_domains = int(len(np.unique(d_labels)))
    model = DANNVAE(X_sc.shape[1], latent_dim, n_domains, h=h).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_sc), torch.LongTensor(d_labels)),
        batch_size=batch_size, shuffle=True, drop_last=True
    )

    best_loss: float = float("inf")
    best_state: dict | None = None
    history: dict[str, list[float]] = {"total": [], "elbo": [], "domain": []}

    print(f"  Training DANN-VAE (λ_d={lam_domain:.2f}, β={beta:.1f})")

    for epoch in range(1, epochs + 1):
        model.train()
        lam_d = dann_lambda_schedule(epoch, epochs)
        et = ee = ed = nb = 0.0

        for (bx, bd) in loader:
            bx = bx.to(device)
            bd = bd.to(device)
            opt.zero_grad()

            recon, mu, lv, _, d_logit = model(bx, lam_d=lam_d)

            elbo, _, _  = vae_loss_fn(recon, bx, mu, lv, beta)
            # Minimising this with reversed gradients → encoder domain confusion
            d_loss = F.cross_entropy(d_logit, bd)

            loss = elbo + lam_domain * d_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            et += loss.item(); ee += elbo.item(); ed += d_loss.item()
            nb += 1

        sched.step()
        avg_t = et / nb
        history["total"].append(avg_t)
        history["elbo"].append(ee / nb)
        history["domain"].append(ed / nb)

        if avg_t < best_loss:
            best_loss = avg_t
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 25 == 0 or epoch == 1:
            print(f"    Ep {epoch:3d}/{epochs}  total={avg_t:.4f}  "
                  f"elbo={ee/nb:.4f}  domain={ed/nb:.4f}  λ_d={lam_d:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, best_loss
