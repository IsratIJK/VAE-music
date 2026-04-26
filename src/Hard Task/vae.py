"""
vae.py
------
All model architectures, shared helpers, loss function,
unified training engine, and latent extraction.
"""

import copy
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import normalize

warnings.filterwarnings('ignore')

# Global config (imported by other modules too)
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LATENT_DIM = 32
HIDDEN_DIMS = (256, 128, 64)
CONV_CHANNELS = (32, 64, 128)
EPOCHS = 100
EARLY_STOP_PATIENCE  = 10
LR = 1e-3
BETA = 1.0
BETA_VAE_B = 4.0
BETA_VALUES = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
BATCH_SIZE = 256
LYRIC_DIM = 128
FUSION_DIM = 256
KMEANS_NINIT = 20

# Spectrogram config
N_MFCC = 20
TIME_FRAMES = 128
N_MFCC_ROWS = 3 * N_MFCC   # 60 (MFCC + Δ + Δ²)
MFCC_2D_DIM = N_MFCC_ROWS * TIME_FRAMES   # 7680
AUDIO_FEAT_DIM = 65


def normalize_for_conv2d(X_flat, n_rows=N_MFCC_ROWS, time_frames=TIME_FRAMES):
    """
    Per-coefficient (row) standardization for Conv2D input.
    Input  : (N, n_rows * time_frames) flat
    Output : (N, n_rows * time_frames) per-row normalized
    Do NOT pass StandardScaler output — causes double normalization.
    """
    X_2d = X_flat.reshape(-1, n_rows, time_frames).copy()   # (N, 60, 128)
    mean = X_2d.mean(axis=(0, 2), keepdims=True)            # (1, 60, 1)
    std = X_2d.std(axis=(0, 2),  keepdims=True) + 1e-8
    X_2d = (X_2d - mean) / std
    return X_2d.reshape(-1, n_rows * time_frames).astype(np.float32)


def align_for_conv2d(X_sc, target=None):
    """Pad or crop flat array to exactly N_MFCC_ROWS * TIME_FRAMES."""
    expected = target or (N_MFCC_ROWS * TIME_FRAMES)
    current = X_sc.shape[1]
    if current == expected:
        return X_sc
    if current < expected:
        return np.pad(X_sc, ((0, 0), (0, expected - current)))
    return X_sc[:, :expected]


# Shared helpers
def make_mlp(dims, activation=nn.LeakyReLU, dropout=0.2, bn=True):
    """Build MLP with BN + activation on hidden layers only."""
    layers, prev = [], dims[0]
    for i, d in enumerate(dims[1:], 1):
        is_last = (i == len(dims) - 1)
        layers.append(nn.Linear(prev, d))
        if not is_last:
            if bn:
                layers.append(nn.BatchNorm1d(d))
            layers.append(
                activation(0.2) if activation == nn.LeakyReLU else activation()
            )
            if dropout:
                layers.append(nn.Dropout(dropout))
        prev = d
    return nn.Sequential(*layers)


def vae_loss_fn(recon, x, mu, lv, beta=1.0):
    """ELBO: reconstruction (MSE) + β * KL divergence, averaged per sample."""
    x_flat = x.view(x.size(0), -1)
    rl = F.mse_loss(recon, x_flat, reduction='sum') / x.size(0)
    kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp()) / x.size(0)
    return rl + beta * kl, rl.item(), kl.item()


# A) MLP-VAE
class MLPVAE(nn.Module):
    def __init__(self, in_dim, z_dim=LATENT_DIM, h=HIDDEN_DIMS):
        super().__init__()
        self.enc_net = make_mlp([in_dim] + list(h))
        self.mu_fc = nn.Linear(h[-1], z_dim)
        self.lv_fc = nn.Linear(h[-1], z_dim)
        self.dec_net = make_mlp([z_dim] + list(reversed(h)) + [in_dim])

    def encode(self, x):
        h = self.enc_net(x)
        return self.mu_fc(h), self.lv_fc(h)

    def reparameterize(self, mu, lv):
        lv = torch.clamp(lv, -10, 10)
        return mu + torch.randn_like(mu) * torch.exp(0.5 * lv) if self.training else mu

    def decode(self, z):
        return self.dec_net(z)

    def forward(self, x):
        mu, lv = self.encode(x)
        z = self.reparameterize(mu, lv)
        return self.decode(z), mu, lv, z

    def get_latent(self, x):
        mu, _ = self.encode(x)
        return mu


# B) Beta-VAE
class BetaVAE(nn.Module):
    """Deeper VAE with tighter lv clamp for stable high-β training."""
    def __init__(self, in_dim, z_dim=LATENT_DIM, beta=4.0, h=(512, 256, 128, 64)):
        super().__init__()
        self.beta = beta
        self.z_dim = z_dim
        self.in_dim = in_dim
        self.enc_net = make_mlp([in_dim] + list(h))
        self.mu_fc = nn.Linear(h[-1], z_dim)
        self.lv_fc = nn.Linear(h[-1], z_dim)
        self.dec_net = make_mlp([z_dim] + list(reversed(h)) + [in_dim])

    def encode(self, x):
        return self.mu_fc(self.enc_net(x)), self.lv_fc(self.enc_net(x))

    def reparameterize(self, mu, lv):
        lv = torch.clamp(lv, -4, 4)
        return mu + torch.randn_like(mu) * torch.exp(0.5 * lv) if self.training else mu

    def decode(self, z):
        return self.dec_net(z)

    def forward(self, x):
        mu, lv = self.encode(x)
        z = self.reparameterize(mu, lv)
        return self.decode(z), mu, lv, z

    def get_latent(self, x):
        mu, _ = self.encode(x)
        return mu

    @torch.no_grad()
    def disentanglement_score(self, X_sc, batch_size=256):
        self.eval()
        mus = []
        X_t = torch.FloatTensor(X_sc)
        for i in range(0, len(X_t), batch_size):
            bx = X_t[i:i+batch_size].to(next(self.parameters()).device)
            mu, _ = self.encode(bx)
            mus.append(mu.cpu())
        mus = torch.cat(mus, dim=0)
        var_per_dim = mus.var(dim=0)
        p = var_per_dim / (var_per_dim.sum() + 1e-8)
        entropy = -(p * torch.log(p + 1e-8)).sum().item()
        print(f' Dim variances (top-10): {var_per_dim.topk(min(10, self.z_dim)).values.numpy().round(3)}')
        print(f' Variance entropy: {entropy:.4f}  (lower = more axis-aligned)')
        return var_per_dim.numpy(), entropy


# C) CVAE
class CVAE(nn.Module):
    def __init__(self, in_dim, n_class, z_dim=LATENT_DIM, h=HIDDEN_DIMS):
        super().__init__()
        self.n_class = n_class
        self.cond_dim = n_class

        self.enc_net = make_mlp([in_dim + n_class] + list(h))
        self.mu_fc = nn.Linear(h[-1], z_dim)
        self.lv_fc = nn.Linear(h[-1], z_dim)
        self.dec_net = make_mlp([z_dim + n_class] + list(reversed(h)) + [in_dim])

    def encode(self, x, c):
        return (self.mu_fc(self.enc_net(torch.cat([x, c], dim=1))),
                self.lv_fc(self.enc_net(torch.cat([x, c], dim=1))))

    def reparameterize(self, mu, lv):
        lv = torch.clamp(lv, -10, 10)
        return mu + torch.randn_like(mu) * torch.exp(0.5 * lv) if self.training else mu

    def decode(self, z, c):
        return self.dec_net(torch.cat([z, c], dim=1))

    def forward(self, x, c):
        mu, lv = self.encode(x, c)
        z = self.reparameterize(mu, lv)
        return self.decode(z, c), mu, lv, z

    def encode_unconditional(self, x):
        c = torch.zeros(x.size(0), self.n_class, device=x.device)
        return self.encode(x, c)

    def get_latent(self, x):
        mu, _ = self.encode_unconditional(x)
        return mu


# D) Conv1D-VAE
class ConvVAE(nn.Module):
    def __init__(self, in_dim, z_dim=LATENT_DIM, channels=(32, 64, 128)):
        super().__init__()
        self.in_dim = in_dim
        enc_layers, prev = [], 1
        for ch in channels:
            enc_layers += [nn.Conv1d(prev, ch, 5, stride=2, padding=2),
                           nn.BatchNorm1d(ch), nn.LeakyReLU(0.2)]
            prev = ch
        self.conv_enc = nn.Sequential(*enc_layers)
        with torch.no_grad():
            dummy = self.conv_enc(torch.zeros(1, 1, in_dim))
        raw_flat = dummy.view(1, -1).shape[1]
        self.ch0 = channels[-1]
        self.seq0 = raw_flat // self.ch0
        self.flat = self.seq0 * self.ch0
        self.mu_fc = nn.Linear(self.flat, z_dim)
        self.lv_fc = nn.Linear(self.flat, z_dim)
        self.fc_dec = nn.Linear(z_dim, self.flat)
        dec_layers, prev = [], channels[-1]
        for ch in reversed(channels[:-1]):
            dec_layers += [nn.ConvTranspose1d(prev, ch, 4, stride=2, padding=1),
                           nn.BatchNorm1d(ch), nn.LeakyReLU(0.2)]
            prev = ch
        dec_layers.append(nn.ConvTranspose1d(prev, 1, 4, stride=2, padding=1))
        self.conv_dec = nn.Sequential(*dec_layers)

    def encode(self, x):
        h = self.conv_enc(x.unsqueeze(1))
        h = h.view(x.size(0), self.ch0, -1)[:, :, :self.seq0].reshape(x.size(0), -1)
        return self.mu_fc(h), self.lv_fc(h)

    def reparameterize(self, mu, lv):
        lv = torch.clamp(lv, -10, 10)
        return mu + torch.randn_like(mu) * torch.exp(0.5 * lv) if self.training else mu

    def decode(self, z):
        h = self.fc_dec(z).view(z.size(0), self.ch0, self.seq0)
        out = self.conv_dec(h)
        return F.adaptive_avg_pool1d(out, self.in_dim).squeeze(1)

    def forward(self, x):
        mu, lv = self.encode(x)
        z = self.reparameterize(mu, lv)
        return self.decode(z), mu, lv, z

    def get_latent(self, x):
        mu, _ = self.encode(x)
        return mu


# E) Autoencoder (deterministic baseline)
class Autoencoder(nn.Module):
    def __init__(self, in_dim, z_dim=LATENT_DIM, h=HIDDEN_DIMS):
        super().__init__()
        self.encoder = make_mlp([in_dim] + list(h) + [z_dim])
        self.decoder = make_mlp([z_dim] + list(reversed(h)) + [in_dim])

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def get_latent(self, x):
        return self.encoder(x)

    def enc(self, x):
        return self.encoder(x), None


# F) MultiModalVAE
class MultiModalVAE(nn.Module):
    """Joint audio+lyric encoder. Reconstructs audio only (primary modality)."""
    def __init__(self, audio_dim=AUDIO_FEAT_DIM, lyric_dim=LYRIC_DIM,
                 fusion_dim=FUSION_DIM, z_dim=LATENT_DIM, h=HIDDEN_DIMS):
        super().__init__()
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.ReLU()
        )
        self.lyric_proj = nn.Sequential(
            nn.Linear(lyric_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.ReLU()
        )
        self.enc_net = make_mlp([2 * fusion_dim] + list(h))
        self.mu_fc = nn.Linear(h[-1], z_dim)
        self.lv_fc = nn.Linear(h[-1], z_dim)
        self.dec_net = make_mlp([z_dim] + list(reversed(h)) + [audio_dim])

    def encode(self, audio, lyric):
        a = self.audio_proj(audio)
        l = self.lyric_proj(lyric)
        h = self.enc_net(torch.cat([a, l], dim=1))
        return self.mu_fc(h), self.lv_fc(h)

    def reparameterize(self, mu, lv):
        lv = torch.clamp(lv, -10, 10)
        return mu + torch.randn_like(mu) * torch.exp(0.5 * lv) if self.training else mu

    def decode(self, z):
        return self.dec_net(z)

    def forward(self, audio, lyric):
        mu, lv = self.encode(audio, lyric)
        z = self.reparameterize(mu, lv)
        return self.decode(z), mu, lv, z

    def get_latent(self, audio, lyric):
        mu, _ = self.encode(audio, lyric)
        return mu


# G) Conv2D-VAE
class Conv2DEncoder(nn.Module):
    """Input: (B, 1, N_MFCC_ROWS, TIME_FRAMES) = (B, 1, 60, 128)"""
    def __init__(self, n_mfcc=N_MFCC_ROWS, time_frames=TIME_FRAMES,
                 z_dim=LATENT_DIM, channels=CONV_CHANNELS):
        super().__init__()
        layers, in_ch = [], 1
        for out_ch in channels:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            dummy         = torch.zeros(1, 1, n_mfcc, time_frames)
            self.flat_dim = self.conv(dummy).view(1, -1).shape[1]
        self.mu_fc = nn.Linear(self.flat_dim, z_dim)
        self.lv_fc = nn.Linear(self.flat_dim, z_dim)

    def forward(self, x):
        h = self.conv(x).view(x.size(0), -1)
        return self.mu_fc(h), self.lv_fc(h)


class Conv2DDecoder(nn.Module):
    """Input: z (B, z_dim) → reconstructed flat spectrogram (B, N_MFCC_ROWS*TIME_FRAMES)"""
    def __init__(self, z_dim=LATENT_DIM, flat_dim=None,
                 n_mfcc=N_MFCC_ROWS, time_frames=TIME_FRAMES,
                 channels=CONV_CHANNELS):
        super().__init__()
        rev_ch = list(reversed(channels))
        self.ch0 = rev_ch[0]
        self.h0 = max(1, n_mfcc // (2 ** len(channels)))
        self.w0 = max(1, time_frames // (2 ** len(channels)))
        self.fc = nn.Linear(z_dim, self.ch0 * self.h0 * self.w0)
        layers, in_ch = [], rev_ch[0]
        for out_ch in rev_ch[1:]:
            layers += [
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2),
            ]
            in_ch = out_ch
        layers.append(nn.ConvTranspose2d(in_ch, 1, kernel_size=4, stride=2, padding=1))
        self.deconv = nn.Sequential(*layers)
        self.out_size = (n_mfcc, time_frames)

    def forward(self, z):
        h = self.fc(z).view(z.size(0), self.ch0, self.h0, self.w0)
        out = self.deconv(h)
        out = F.adaptive_avg_pool2d(out, self.out_size)
        return out.view(z.size(0), -1)


class Conv2DVAE(nn.Module):
    """
    Full Conv2D-VAE for delta-stacked MFCC spectrograms.
    Accepts flat (B, 7680) or 2D (B, 1, 60, 128) input.
    Input contract: pass normalize_for_conv2d(X_raw_2d).
    """
    def __init__(self, n_mfcc=N_MFCC_ROWS, time_frames=TIME_FRAMES,
                 z_dim=LATENT_DIM, channels=CONV_CHANNELS):
        super().__init__()
        self.n_mfcc = n_mfcc
        self.time_frames = time_frames
        self.flat_dim = n_mfcc * time_frames
        self.enc = Conv2DEncoder(n_mfcc, time_frames, z_dim, channels)
        self.dec = Conv2DDecoder(z_dim, self.enc.flat_dim, n_mfcc, time_frames, channels)

    def _to_2d(self, x):
        if x.dim() == 2:
            return x.view(x.size(0), 1, self.n_mfcc, self.time_frames)
        return x

    def reparameterize(self, mu, lv):
        lv = torch.clamp(lv, -10, 10)
        return mu + torch.randn_like(mu) * torch.exp(0.5 * lv) if self.training else mu

    def forward(self, x):
        x2d = self._to_2d(x)
        mu, lv = self.enc(x2d)
        z = self.reparameterize(mu, lv)
        return self.dec(z), mu, lv, z

    def get_latent(self, x):
        mu, _ = self.enc(self._to_2d(x))
        return mu


# H) HybridConvVAE
# Input contract:
#   Passed as single array X_hybrid_conv = np.hstack([X_conv2d, X_lyric_l2])
#   Shape: (N, MFCC_2D_DIM + LYRIC_DIM) = (N, 7808)
#   conv part [:7680] = normalize_for_conv2d output
#   lyric part [7680:] = L2-normalized lyrics from make_multimodal()
class HybridConvVAE(nn.Module):
    def __init__(self, lyric_dim=LYRIC_DIM, z_dim=LATENT_DIM,
                 n_mfcc=N_MFCC_ROWS, time_frames=TIME_FRAMES,
                 channels=CONV_CHANNELS, fusion_dim=FUSION_DIM):
        super().__init__()
        self.n_mfcc = n_mfcc
        self.time_frames = time_frames
        self.conv_dim = n_mfcc * time_frames   # 7680 = split point
        self.lyric_dim = lyric_dim

        # Conv2D encoder
        enc_layers, in_ch = [], 1
        for out_ch in channels:
            enc_layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2),
            ]
            in_ch = out_ch
        self.conv_enc = nn.Sequential(*enc_layers)
        with torch.no_grad():
            dummy          = torch.zeros(1, 1, n_mfcc, time_frames)
            self.conv_flat = self.conv_enc(dummy).view(1, -1).shape[1]

        # Lyric projection → same width as conv_flat for balanced fusion
        self.lyric_proj = nn.Sequential(
            nn.Linear(lyric_dim, self.conv_flat),
            nn.LayerNorm(self.conv_flat),
            nn.LeakyReLU(0.2),
        )
        # Fusion: conv_flat + lyric_flat → fusion_dim
        self.fusion = nn.Sequential(
            nn.Linear(self.conv_flat * 2, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.LeakyReLU(0.2),
        )
        self.mu_fc = nn.Linear(fusion_dim, z_dim)
        self.lv_fc = nn.Linear(fusion_dim, z_dim)

        # Conv2D decoder (mirrors encoder)
        rev_ch = list(reversed(channels))
        self.ch0 = rev_ch[0]
        self.h0 = max(1, n_mfcc // (2 ** len(channels)))
        self.w0 = max(1, time_frames // (2 ** len(channels)))
        self.fc_dec = nn.Linear(z_dim, self.ch0 * self.h0 * self.w0)
        dec_layers, in_ch = [], rev_ch[0]
        for out_ch in rev_ch[1:]:
            dec_layers += [
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2),
            ]
            in_ch = out_ch
        dec_layers.append(nn.ConvTranspose2d(in_ch, 1, kernel_size=4, stride=2, padding=1))
        self.conv_dec = nn.Sequential(*dec_layers)
        self.out_size = (n_mfcc, time_frames)

    def _to_2d(self, x):
        if x.dim() == 2:
            return x.view(x.size(0), 1, self.n_mfcc, self.time_frames)
        return x

    def encode(self, x_conv, x_lyric):
        h_conv = self.conv_enc(self._to_2d(x_conv)).view(x_conv.size(0), -1)
        h_lyric = self.lyric_proj(x_lyric)
        h_fused = self.fusion(torch.cat([h_conv, h_lyric], dim=1))
        return self.mu_fc(h_fused), self.lv_fc(h_fused)

    def reparameterize(self, mu, lv):
        lv = torch.clamp(lv, -10, 10)
        return mu + torch.randn_like(mu) * torch.exp(0.5 * lv) if self.training else mu

    def decode(self, z):
        h = self.fc_dec(z).view(z.size(0), self.ch0, self.h0, self.w0)
        out = self.conv_dec(h)
        out = F.adaptive_avg_pool2d(out, self.out_size)
        return out.view(z.size(0), -1)

    def forward(self, x_conv, x_lyric):
        mu, lv = self.encode(x_conv, x_lyric)
        z = self.reparameterize(mu, lv)
        return self.decode(z), mu, lv, z

    def get_latent(self, x_conv, x_lyric):
        mu, _ = self.encode(x_conv, x_lyric)
        return mu


def train_model(X_sc, model, y_onehot=None,
                epochs=EPOCHS, lr=LR, beta=1.0,
                batch_size=BATCH_SIZE, model_type='vae',
                audio_dim=None, verbose=True):
    """
    Unified training loop for all model types.

    model_type : 'vae' | 'ae' | 'cvae' | 'multimodal' | 'hybrid_conv'
    audio_dim  : required when model_type='multimodal'

    INPUT CONTRACT per model_type:
      'vae' on MLPVAE/BetaVAE/ConvVAE  → X_sc (StandardScaler audio, 65-dim)
      'vae' on Conv2DVAE → X_conv2d = normalize_for_conv2d(X_raw_2d) (7680-dim)
      'ae' → X_sc (65-dim)
      'cvae' → X_sc (65-dim) + y_onehot
      'multimodal' → X_multimodal = np.hstack([X_sc, X_lyric_l2]) (193-dim)
      'hybrid_conv' → X_hybrid_conv = np.hstack([X_conv2d, X_lyric_l2]) (7808-dim)
      'vae' on MLPVAE for Hybrid-MLP → X_hybrid (L2 audio ‖ L2 lyrics) (193-dim)
    """
    model = model.to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    X_t = torch.FloatTensor(X_sc)
    N = len(X_t)
    idx = torch.randperm(N, generator=torch.Generator().manual_seed(SEED))
    split = int(0.9 * N)
    tr_idx, va_idx = idx[:split], idx[split:]

    def make_ds(indices):
        Xs = X_t[indices]
        if y_onehot is not None:
            return TensorDataset(Xs, torch.FloatTensor(y_onehot)[indices])
        return TensorDataset(Xs)

    tr_loader = DataLoader(make_ds(tr_idx), batch_size=batch_size, shuffle=True,  drop_last=False)
    va_loader = DataLoader(make_ds(va_idx), batch_size=batch_size, shuffle=False, drop_last=False)

    best_val = float('inf')
    best_state = None
    patience = 0
    history = []

    def _forward(batch):
        bx = batch[0].to(DEVICE)
        bc = batch[1].to(DEVICE) if len(batch) > 1 else None
        if model_type == 'ae':
            recon, _ = model(bx)
            return F.mse_loss(recon, bx)
        elif model_type == 'cvae':
            recon, mu, lv, _ = model(bx, bc)
            loss, _, _ = vae_loss_fn(recon, bx, mu, lv, beta)
            return loss
        elif model_type == 'multimodal':
            bx_audio = bx[:, :audio_dim]
            bx_lyric = bx[:, audio_dim:]
            recon, mu, lv, _ = model(bx_audio, bx_lyric)
            loss, _, _ = vae_loss_fn(recon, bx_audio, mu, lv, beta)
            return loss
        elif model_type == 'hybrid_conv':
            split_pt = N_MFCC_ROWS * TIME_FRAMES
            bx_conv  = bx[:, :split_pt]
            bx_lyric = bx[:, split_pt:]
            recon, mu, lv, _ = model(bx_conv, bx_lyric)
            loss, _, _ = vae_loss_fn(recon, bx_conv, mu, lv, beta)
            return loss
        else:   # 'vae'
            recon, mu, lv, _ = model(bx)
            loss, _, _ = vae_loss_fn(recon, bx, mu, lv, beta)
            return loss

    for epoch in range(1, epochs + 1):
        model.train()
        tr_sum = 0.0
        for batch in tr_loader:
            opt.zero_grad()
            loss = _forward(batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_sum += loss.item()
        tr_avg = tr_sum / len(tr_loader)

        model.eval()
        va_sum = 0.0
        with torch.no_grad():
            for batch in va_loader:
                va_sum += _forward(batch).item()
        va_avg = va_sum / len(va_loader)

        current_lr = sched.get_last_lr()[0]
        sched.step()
        history.append((tr_avg, va_avg, current_lr))

        if va_avg < best_val:
            best_val = va_avg
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1

        if verbose and (epoch % 25 == 0 or epoch == 1):
            print(f' Ep {epoch:3d}/{epochs}  '
                  f'train={tr_avg:.4f}  val={va_avg:.4f}  '
                  f'lr={current_lr:.2e}  patience={patience}/{EARLY_STOP_PATIENCE}')

        if patience >= EARLY_STOP_PATIENCE:
            if verbose:
                print(f'    Early stop at epoch {epoch}  (best val={best_val:.4f})')
            break

    model.load_state_dict(best_state)
    return model, history, best_val


def extract_latent(model, X_sc, batch_size=BATCH_SIZE,
                   model_type='vae', audio_dim=None):
    """Extract latent mu vectors. model_type matches train_model contract."""
    model.eval()
    X_t = torch.FloatTensor(X_sc)
    Z_list = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            batch = X_t[i:i+batch_size].to(DEVICE)
            if model_type == 'multimodal':
                mu = model.get_latent(batch[:, :audio_dim], batch[:, audio_dim:])
            elif model_type == 'hybrid_conv':
                split_pt = N_MFCC_ROWS * TIME_FRAMES
                mu = model.get_latent(batch[:, :split_pt], batch[:, split_pt:])
            else:
                mu = model.get_latent(batch)
            Z_list.append(mu.cpu().numpy())
    return np.vstack(Z_list)
