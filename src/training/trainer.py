"""
Unified training engine for all model types.

Supports VAE, Beta-VAE, CVAE, Conv1D-VAE, and Autoencoder with:
  - AdamW optimiser
  - Cosine-annealing LR scheduler
  - Gradient clipping (max norm = 1.0)
  - Best-state checkpointing

Functions
---------
train_model: Train any supported model; returns (model, history, best_loss).
extract_latent: Extract latent codes from a trained model.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models.base_vae import vae_loss_fn


def train_model(X_sc: np.ndarray, model: nn.Module, y_onehot: np.ndarray | None = None, 
                model_type: str = "vae", epochs: int = 100, lr: float = 1e-3, 
                beta: float = 1.0, batch_size: int = 256, 
                device: torch.device | None = None, 
                verbose: bool = True) -> tuple[nn.Module, list[float], float]:
    """Train a VAE / AE model on scaled feature matrix X_sc.

    Parameters
    ----------
    X_sc: Scaled feature matrix (N, feat_dim).
    model: PyTorch model to train.
    y_onehot: One-hot genre labels (N, n_class) required for CVAE.
    model_type: One of: "vae", "beta_vae", "cvae", "ae".
    epochs: Number of training epochs.
    lr: AdamW learning rate.
    beta: β weight on KL term (for VAE / Beta-VAE).
    batch_size: Mini-batch size.
    device: Target device (defaults to cuda if available).
    verbose: Print progress every 25 epochs.

    Returns
    -------
    (model, history, best_loss)
      model: Best-state model (loaded after training).
      history: List of per-epoch average total losses.
      best_loss: Lowest training loss observed.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    X_t = torch.FloatTensor(X_sc)
    if y_onehot is not None:
        dataset = TensorDataset(X_t, torch.FloatTensor(y_onehot))
    else:
        dataset = TensorDataset(X_t)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    best_loss: float = float("inf")
    best_state: dict | None = None
    history: list[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_total = 0.0
        n_batches = 0

        for batch in loader:
            bx = batch[0].to(device)
            bc = batch[1].to(device) if y_onehot is not None else None
            opt.zero_grad()

            if model_type == "ae":
                recon, _ = model(bx)
                loss = F.mse_loss(recon, bx)
            elif model_type == "cvae":
                recon, mu, lv, _ = model(bx, bc)
                loss, _, _ = vae_loss_fn(recon, bx, mu, lv, beta)
            else:
                # vae or beta_vae
                recon, mu, lv, _ = model(bx)
                loss, _, _ = vae_loss_fn(recon, bx, mu, lv, beta)

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

        if verbose and (epoch % 25 == 0 or epoch == 1):
            print(f"    Epoch {epoch:3d}/{epochs}  loss={avg:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, best_loss


def extract_latent(model: nn.Module, X_sc: np.ndarray, batch_size: int = 256, 
                   device: torch.device | None = None) -> np.ndarray:
    """Extract latent codes from any trained model.

    Uses the ''model.enc(x)'' interface which all models expose:
      - VAE-like models: returns (mu, log_var) -> mu is used
      - Autoencoder: returns (z, None) -> z is used

    Parameters
    ----------
    model: Trained model with an ''enc'' method.
    X_sc: Scaled feature matrix (N, feat_dim).
    batch_size: Inference batch size.
    device: Target device.

    Returns
    -------
    Float32 array of shape (N, latent_dim).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model = model.to(device)
    Z_list: list[np.ndarray] = []
    X_t = torch.FloatTensor(X_sc)

    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            batch = X_t[i : i + batch_size].to(device)
            result = model.enc(batch)
            # result[0] is mu (VAE) or z (AE)
            Z_list.append(result[0].cpu().numpy())

    return np.vstack(Z_list)
