"""
vae.py
------
VAE architecture (Encoder, Decoder, VAE), loss function,
training loop, and latent-space extraction.

"""

import warnings
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# VAE Architecture


class Encoder(nn.Module):
    def __init__(self, in_dim, h_dims, z_dim):
        super().__init__()
        layers, prev = [], in_dim
        for h in h_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ]
            prev = h
        self.net   = nn.Sequential(*layers)
        self.mu_fc = nn.Linear(prev, z_dim)
        self.lv_fc = nn.Linear(prev, z_dim)

    def forward(self, x):
        h = self.net(x)
        return self.mu_fc(h), self.lv_fc(h)


class Decoder(nn.Module):
    def __init__(self, z_dim, h_dims, out_dim):
        super().__init__()
        layers, prev = [], z_dim
        for h in reversed(h_dims):
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class VAE(nn.Module):
    def __init__(self, in_dim, h_dims, z_dim):
        super().__init__()
        self.enc = Encoder(in_dim, h_dims, z_dim)
        self.dec = Decoder(z_dim, h_dims, in_dim)

    def reparameterize(self, mu, lv):
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return mu

    def forward(self, x):
        mu, lv = self.enc(x)
        z = self.reparameterize(mu, lv)
        return self.dec(z), mu, lv, z


def vae_loss(recon, x, mu, lv, beta=1.0):
    """beta-VAE ELBO: Reconstruction MSE + beta x KL divergence"""
    rl = F.mse_loss(recon, x, reduction='sum') / x.size(0)
    kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp()) / x.size(0)
    return rl + beta * kl, rl.item(), kl.item()


def train_vae(X_scaled, out_dir='/content',
              hidden_dims=None, latent_dim=16,
              batch_size=32, epochs=100, lr=1e-3, beta=1.0):
    """
    Build, train, and return the best VAE model plus training history.

    Parameters
    ----------
    X_scaled   : np.ndarray  - normalised feature matrix (N, D)
    out_dir    : str         - directory to save model checkpoint
    hidden_dims: list        - encoder hidden layer sizes  (default [256, 128])
    latent_dim : int         - latent space dimensionality
    batch_size : int         - mini-batch size (drop_last=True for BatchNorm safety)
    epochs     : int         - number of training epochs
    lr         : float       - initial AdamW learning rate
    beta       : float       - KL weight (beta-VAE; 1.0 = standard VAE)

    Returns
    -------
    vae        : trained VAE (best checkpoint loaded)
    history    : dict with keys 'total', 'recon', 'kl'
    best_loss  : float
    """
    if hidden_dims is None:
        hidden_dims = [256, 128]   # 2 layers suits dataset size (~240 tracks)

    INPUT_DIM = X_scaled.shape[1]  # 102 (audio-only)

    X_tensor = torch.FloatTensor(X_scaled)
    loader   = DataLoader(TensorDataset(X_tensor),
                          batch_size=batch_size, shuffle=True, drop_last=True)
                          # drop_last=True prevents single-sample batches
                          # that crash BatchNorm1d

    vae       = VAE(INPUT_DIM, hidden_dims, latent_dim).to(DEVICE)
    optimizer = optim.AdamW(vae.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    total_params = sum(p.numel() for p in vae.parameters())
    print(f'VAE | Params: {total_params:,}')
    print(f'   {INPUT_DIM} -> {hidden_dims} -> z{latent_dim}')
    print(f'   beta={beta} | LR={lr} | Batch={batch_size} | Epochs={epochs}')

    history    = {'total': [], 'recon': [], 'kl': []}
    best_loss  = float('inf')
    best_state = None

    for epoch in range(1, epochs + 1):
        vae.train()
        et = er = ek = nb = 0
        for (bx,) in loader:
            bx = bx.to(DEVICE)
            optimizer.zero_grad()
            recon, mu, lv, _ = vae(bx)
            loss, rl, kl     = vae_loss(recon, bx, mu, lv, beta)
            loss.backward()
            nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            optimizer.step()
            et += loss.item(); er += rl; ek += kl; nb += 1

        scheduler.step()
        at, ar, ak = et/nb, er/nb, ek/nb
        history['total'].append(at)
        history['recon'].append(ar)
        history['kl'].append(ak)

        if at < best_loss:
            best_loss  = at
            best_state = {k: v.cpu().clone() for k, v in vae.state_dict().items()}

        if epoch % 20 == 0 or epoch == 1:
            print(f'  Epoch {epoch:3d}/{epochs} | Total={at:.4f}  Recon={ar:.4f}  KL={ak:.4f}')

    print(f'Training done. Best loss: {best_loss:.4f}')
    vae.load_state_dict(best_state)

    import os
    os.makedirs(out_dir, exist_ok=True)
    import pickle
    torch.save({
        'model_state': best_state,
        'config': {
            'input_dim':   INPUT_DIM,
            'hidden_dims': hidden_dims,
            'latent_dim':  latent_dim,
            'beta':        beta,
        },
    }, f'{out_dir}/vae_music_model.pt')
    print(f'Checkpoint saved -> {out_dir}/vae_music_model.pt')

    return vae, history, best_loss


def plot_training_curves(history, out_dir='/content'):
    """Plot and save total / reconstruction / KL loss curves."""
    import os
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, color, title in zip(
        axes,
        ['total', 'recon', 'kl'],
        ['#4e8ef7', '#f7914e', '#6ab187'],
        ['Total Loss', 'Reconstruction Loss', 'KL Divergence']
    ):
        ax.plot(history[key], color=color, linewidth=2)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.grid(alpha=0.3)

    plt.suptitle('VAE Training Curves', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved: training_curves.png')


def extract_latent(vae, X_scaled, latent_dim, batch_size=256):
    """
    Run the trained encoder over the full dataset and return mu (mean) vectors.

    Parameters
    ----------
    vae        : trained VAE
    X_scaled   : np.ndarray - normalised feature matrix (N, D)
    latent_dim : int        - for the print statement
    batch_size : int        - inference batch size

    Returns
    -------
    Z : np.ndarray shape (N, latent_dim)
    """
    X_tensor = torch.FloatTensor(X_scaled)
    vae.eval()
    with torch.no_grad():
        all_Z = []
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size].to(DEVICE)
            _, mu, _, _ = vae(batch)
            all_Z.append(mu.cpu().numpy())

    Z = np.vstack(all_Z)
    print(f'Latent space extracted: {Z.shape}  (N samples x {latent_dim} dims)')
    return Z