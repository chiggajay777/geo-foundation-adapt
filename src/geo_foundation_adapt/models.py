"""Backbones and adaptation wrappers.

TinyEOEncoder is the offline ``--demo`` stand-in.
The benchmark backbone is torchvision ResNet-18. Published weights are Seasonal
Contrast (SeCo) Sentinel-2 RGB from TorchGeo, not a transformer foundation model.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class TinyEOEncoder(nn.Module):
    """Small CNN used only by ``--demo``. Not a published encoder."""

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
        return self.proj(self.features(x).flatten(1))


class LoRALinear(nn.Module):
    """Low-rank update on a frozen linear map. Used by the synthetic demo."""

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
        return self.linear(x) + F.linear(x, self.B @ self.A) * self.scale


class LinearProbeClassifier(nn.Module):
    def __init__(self, encoder: TinyEOEncoder, n_classes: int = 4, freeze_encoder: bool = True):
        super().__init__()
        self.encoder = encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.head = nn.Linear(encoder.emb_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frozen = not any(p.requires_grad for p in self.encoder.parameters())
        if frozen:
            self.encoder.eval()
            with torch.no_grad():
                emb = self.encoder(x)
        else:
            emb = self.encoder(x)
        return self.head(emb)


class LoRAProbeClassifier(nn.Module):
    """Frozen TinyEOEncoder with LoRA on the projection, plus a new head."""

    def __init__(self, encoder: TinyEOEncoder, n_classes: int = 4, rank: int = 4):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.proj = LoRALinear(self.encoder.proj, rank=rank)
        self.head = nn.Linear(encoder.emb_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class LoRAConv2d(nn.Module):
    """Conv LoRA: frozen conv plus a low-rank kernel in channel space.

    ``A`` is a k×k convolution down to ``rank`` channels (same stride and padding
    as the frozen layer). ``B`` is a 1×1 convolution back to the output channels.
    ``B`` is initialized at 0, so the adapted layer matches the frozen layer at step 0.
    """

    def __init__(self, conv: nn.Conv2d, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        if conv.groups != 1:
            raise ValueError("LoRAConv2d supports groups=1 only")
        self.conv = conv
        for p in self.conv.parameters():
            p.requires_grad = False
        self.scale = alpha / rank
        self.A = nn.Conv2d(
            conv.in_channels,
            rank,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            bias=False,
        )
        self.B = nn.Conv2d(rank, conv.out_channels, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.B(self.A(x)) * self.scale


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def resnet_embedding(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """ResNet-18 forward through global pooling, excluding ``fc``."""
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    return torch.flatten(x, 1)


def inject_resnet_lora(model: nn.Module, rank: int, alpha: float) -> None:
    """Wrap both 3×3 convolutions in ``layer4``. The downsample projection stays frozen."""
    for block in model.layer4:
        block.conv1 = LoRAConv2d(block.conv1, rank=rank, alpha=alpha)
        block.conv2 = LoRAConv2d(block.conv2, rank=rank, alpha=alpha)


def freeze_batchnorm(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            for p in module.parameters():
                p.requires_grad = False


def build_resnet18(
    state: dict[str, Any],
    *,
    n_classes: int,
    method: str,
    seed: int,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
) -> nn.Module:
    """Load a backbone checkpoint and attach one adaptation method.

    The seed is reset before the classification head is created, and again before
    LoRA is created, so method order does not change initialization.
    """
    if method not in {"linear_probe", "lora", "full_finetune"}:
        raise ValueError(f"unknown method {method}")

    torch.manual_seed(seed)
    model = resnet18(weights=None)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing_ok = {"fc.weight", "fc.bias"}
    if unexpected or set(missing) - missing_ok:
        raise RuntimeError(f"unexpected checkpoint mismatch missing={missing} unexpected={unexpected}")

    torch.manual_seed(seed + 10_000)
    model.fc = nn.Linear(model.fc.in_features, n_classes)

    if method == "lora":
        torch.manual_seed(seed + 20_000)
        inject_resnet_lora(model, rank=lora_rank, alpha=lora_alpha)
        for name, param in model.named_parameters():
            # LoRAConv2d parameters are "*.A.weight" and "*.B.weight".
            param.requires_grad = name.startswith("fc.") or ".A.weight" in name or ".B.weight" in name
        freeze_batchnorm(model)
    elif method == "linear_probe":
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
        freeze_batchnorm(model)
    else:
        for param in model.parameters():
            param.requires_grad = True
    return model


def apply_train_mode(model: nn.Module, method: str) -> None:
    """Full finetune updates BatchNorm. Probe and LoRA keep pretrained running stats."""
    model.train()
    if method in {"linear_probe", "lora"}:
        freeze_batchnorm(model)
