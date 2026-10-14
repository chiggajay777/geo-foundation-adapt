"""Tiny EO encoder + linear probe + LoRA-style adapters.

Implemented backbone: TinyEOEncoder (local CNN) — runs on CPU in minutes.
Documented (not downloaded here): Prithvi, SatMAE, TorchSpatial / location encoders.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyEOEncoder(nn.Module):
    """Small CNN encoder producing a 128-d embedding (stand-in for a frozen FM)."""

    def __init__(self, in_ch: int = 3, emb_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, emb_dim)
        self.emb_dim = emb_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).flatten(1)
        return self.proj(h)


class LoRALinear(nn.Module):
    """Low-rank adapter on a frozen linear layer: y = W x + (B A) x * scale."""

    def __init__(self, linear: nn.Linear, rank: int = 4, alpha: float = 8.0):
        super().__init__()
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad = False
        d_out, d_in = linear.weight.shape
        self.A = nn.Parameter(torch.zeros(rank, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        update = F.linear(x, self.B @ self.A) * self.scale
        return base + update


class LinearProbeClassifier(nn.Module):
    def __init__(self, encoder: TinyEOEncoder, n_classes: int = 4, freeze_encoder: bool = True):
        super().__init__()
        self.encoder = encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.head = nn.Linear(encoder.emb_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(self.encoder.training and any(p.requires_grad for p in self.encoder.parameters())):
            # Always eval-style for frozen encoder
            if not any(p.requires_grad for p in self.encoder.parameters()):
                self.encoder.eval()
                with torch.no_grad():
                    emb = self.encoder(x)
            else:
                emb = self.encoder(x)
        return self.head(emb)


class LoRAProbeClassifier(nn.Module):
    """Frozen encoder + LoRA on projection + trainable classifier head."""

    def __init__(self, encoder: TinyEOEncoder, n_classes: int = 4, rank: int = 4):
        super().__init__()
        # Deep-copy-like: reuse encoder weights, freeze, wrap proj with LoRA
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.proj = LoRALinear(self.encoder.proj, rank=rank)
        self.head = nn.Linear(encoder.emb_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(x)
        return self.head(emb)


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
