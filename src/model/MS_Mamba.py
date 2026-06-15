#!/usr/bin/env python3
"""MS-Mamba — Mamba + MixStyle, standalone single-file model.

A lightweight 1D-CNN Mamba backbone with cross-instance feature-statistics
mixing (MixStyle) for domain generalization. Designed for short, multi-channel
time-series classification (e.g., 60-second 1Hz mobile-traffic windows).

Architecture
------------
    input (B, T, in_dim)
        → Linear + LN + GELU + Dropout          # embedding
        → MixStyle                              # train-time only, identity at eval
        → N × MambaBlock                        # unidirectional selective SSM
                                                #   + depthwise Conv1D (k=d_conv)
                                                #   + multi-scale Conv1D branch
                                                #     (k=3, 7, 15, 31)
        → LayerNorm
        → Global Average Pooling over time      # (B, D)
        → Classifier (Linear → GELU → Linear)   # (B, num_classes)

Per-view parameter count ≈ 1.9M with default config
(d_model=256, num_layers=4).

Dependencies
------------
    torch >= 2.0
    einops
    numpy

Example
-------
    from MS_Mamba import MSMamba

    model = MSMamba(input_dim=4, num_classes=101, d_model=256, num_layers=4)
    x = torch.randn(8, 60, 4)   # batch=8, T=60s, 4 channels
    logits = model(x)            # (8, 101)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat


# =================================================================
# MixStyle
# =================================================================

class MixStyle(nn.Module):
    """MixStyle (Zhou et al., ICLR 2021).

    Mixes channel-wise feature statistics (mean, std) between samples in
    the current batch via a random permutation. Content (the L2-normalised
    feature) is preserved while style (mean/std) is mixed, so labels remain
    unchanged. Active during training only; identity at evaluation.

    Args:
        alpha:    Beta distribution parameter (larger → more uniform mixing).
        mix_prob: Probability of applying MixStyle to a given batch.
        eps:      Numerical stability term for std computation.
    """

    def __init__(self, alpha: float = 0.3, mix_prob: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.mix_prob = mix_prob
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        if not self.training or self.mix_prob <= 0.0:
            return x
        if torch.rand(1).item() > self.mix_prob:
            return x

        B = x.size(0)
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True)
        sig = (var + self.eps).sqrt()

        x_normed = (x - mu) / sig

        perm = torch.randperm(B, device=x.device)
        mu_mix = mu[perm]
        sig_mix = sig[perm]

        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((B, 1, 1)).to(x.device)
        mu_new = lam * mu + (1 - lam) * mu_mix
        sig_new = lam * sig + (1 - lam) * sig_mix

        return x_normed * sig_new + mu_new


# =================================================================
# MambaBlock — vanilla Mamba (depthwise Conv1D + selective SSM)
#              augmented with a multi-scale convolution branch.
# =================================================================

class MambaBlock(nn.Module):
    """Single Mamba block with a parallel multi-scale Conv1D branch.

    Standard Mamba primitives:
        LayerNorm → in_proj (split into x, z)
                  → depthwise Conv1D (kernel = d_conv)
                  → SiLU
                  → selective scan (SSM)
                  → gate by SiLU(z)
                  → out_proj
                  → residual + dropout

    Our extension (preserved from MambaMixStyle):
        The selective scan body is replaced by a multi-scale depthwise
        Conv1D branch with kernel sizes {3, 7, 15, 31}, each multiplied
        by an input-dependent gate (B_t · C_t · σ(Δ_t)) from the SSM.
    """

    def __init__(self,
                 d_model: int,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner,
        )
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)

        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            'n -> d n', d=self.d_inner,
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Multi-scale Conv1D branch (custom extension over vanilla Mamba)
        self.kernel_sizes = [3, 7, 15, 31]
        self.ssm_convs = nn.ModuleList()
        for ks in self.kernel_sizes[:min(d_state, 4)]:
            conv = nn.Conv1d(
                self.d_inner, self.d_inner,
                kernel_size=ks, padding=ks - 1,
                groups=self.d_inner, bias=False,
            )
            with torch.no_grad():
                decay = 0.9 - 0.2 * (self.kernel_sizes.index(ks) / len(self.kernel_sizes))
                kernel = torch.tensor([decay ** i for i in range(ks)], dtype=torch.float32)
                kernel = kernel / kernel.sum()
                conv.weight.data = kernel.view(1, 1, ks).expand(self.d_inner, 1, ks).clone()
            self.ssm_convs.append(conv)

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, length, _ = x.shape
        x_norm = self.norm(x)
        xz = self.in_proj(x_norm)
        x_proj, z = xz.chunk(2, dim=-1)

        x_conv = x_proj.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :length]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        y = self.ssm(x_conv)
        y = y * F.silu(z)
        out = self.out_proj(y)
        return x + self.dropout(out)

    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, d_inner = x.shape

        x_dbl = self.x_proj(x)
        dt, B, C = torch.split(x_dbl, [1, self.d_state, self.d_state], dim=-1)
        gate = torch.sigmoid(dt)

        x_t = x.transpose(1, 2)
        y_accum = torch.zeros(batch, length, d_inner, device=x.device)
        for s, conv in enumerate(self.ssm_convs):
            y_scale = conv(x_t)[:, :, :length]
            y_scale = y_scale.transpose(1, 2)
            y_accum = y_accum + y_scale * (B[:, :, s:s + 1] * C[:, :, s:s + 1] * gate)

        return y_accum + self.D.view(1, 1, -1) * x


# =================================================================
# MS-Mamba — top-level model
# =================================================================

class MSMamba(nn.Module):
    """MS-Mamba: Mamba backbone with MixStyle for domain generalization.

    Unidirectional Mamba stack + MixStyle (train-time only) + global
    average pooling + linear classifier.

    Args:
        input_dim:      Input feature channels per time step (e.g., 4 raw
                        traffic channels: up/dn bytes/packets).
        num_classes:    Number of output classes.
        d_model:        Hidden dim (default 256).
        num_layers:     Number of MambaBlocks (default 4).
        d_state:        SSM state size (default 16).
        d_conv:         Depthwise conv kernel in MambaBlock (default 4).
        expand:         d_inner = expand * d_model (default 2).
        dropout:        Dropout rate (default 0.1).
        mixstyle_alpha: MixStyle Beta α (default 0.3).
        mixstyle_prob:  MixStyle apply probability (default 0.5).
    """

    def __init__(self,
                 input_dim: int = 4,
                 num_classes: int = 100,
                 d_model: int = 256,
                 num_layers: int = 4,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 dropout: float = 0.1,
                 mixstyle_alpha: float = 0.3,
                 mixstyle_prob: float = 0.5):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.mixstyle = MixStyle(alpha=mixstyle_alpha, mix_prob=mixstyle_prob)

        self.fwd_layers = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state, d_conv=d_conv,
                       expand=expand, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, input_dim) → logits (B, num_classes)."""
        x = self.embedding(x)
        x = self.mixstyle(x)

        for layer in self.fwd_layers:
            x = layer(x)

        x = self.norm(x)
        x = x.mean(dim=1)              # Global Average Pooling over time
        return self.classifier(x)


# =================================================================
# Convenience factory
# =================================================================

def MSMamba_Default(input_dim: int = 4, num_classes: int = 100) -> MSMamba:
    """Default MS-Mamba config: d_model=256, num_layers=4, MixStyle(α=0.3, p=0.5)."""
    return MSMamba(
        input_dim=input_dim, num_classes=num_classes,
        d_model=256, num_layers=4,
        d_state=16, d_conv=4, expand=2, dropout=0.1,
        mixstyle_alpha=0.3, mixstyle_prob=0.5,
    )


# =================================================================
# Smoke test
# =================================================================

if __name__ == '__main__':
    model = MSMamba_Default(input_dim=4, num_classes=100)
    x = torch.randn(8, 60, 4)              # batch=8, T=60, 4 channels
    y = model(x)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"MS-Mamba  in_dim=4 num_classes=100  params={n_params:.2f}M")
    print(f"input  shape: {tuple(x.shape)}")
    print(f"output shape: {tuple(y.shape)}")
