"""Short CPU/GPU training loop: pretrain encoder briefly, then probe vs LoRA."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from geo_foundation_adapt.data import make_loaders
from geo_foundation_adapt.models import (
    LinearProbeClassifier,
    LoRAProbeClassifier,
    TinyEOEncoder,
    count_trainable,
)


def _accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def _train_epochs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> list[float]:
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    crit = nn.CrossEntropyLoss()
    losses: list[float] = []
    model.train()
    for _ in range(epochs):
        running = 0.0
        n = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * y.size(0)
            n += y.size(0)
        losses.append(running / max(n, 1))
    return losses


def run_demo(
    out_dir: Path,
    *,
    n_per_class: int = 200,
    pretrain_epochs: int = 3,
    adapt_epochs: int = 5,
    batch_size: int = 32,
    seed: int = 42,
    device: str | None = None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    train_loader, val_loader, class_names = make_loaders(
        n_per_class=n_per_class, batch_size=batch_size, seed=seed
    )

    # Stage A: short supervised pretrain (stand-in for a foundation model)
    encoder = TinyEOEncoder()
    warm = LinearProbeClassifier(encoder, n_classes=4, freeze_encoder=False).to(dev)
    _train_epochs(warm, train_loader, dev, pretrain_epochs, lr=1e-3)
    base_encoder = deepcopy(warm.encoder).cpu()
    for p in base_encoder.parameters():
        p.requires_grad = False

    # Stage B1: linear probe
    probe = LinearProbeClassifier(deepcopy(base_encoder), freeze_encoder=True).to(dev)
    _train_epochs(probe, train_loader, dev, adapt_epochs, lr=1e-2)
    probe_acc = _accuracy(probe, val_loader, dev)
    probe_params = count_trainable(probe)

    # Stage B2: LoRA on proj + head
    lora = LoRAProbeClassifier(deepcopy(base_encoder), rank=4).to(dev)
    _train_epochs(lora, train_loader, dev, adapt_epochs, lr=1e-2)
    lora_acc = _accuracy(lora, val_loader, dev)
    lora_params = count_trainable(lora)

    # Full finetune baseline
    full_enc = deepcopy(base_encoder)
    for p in full_enc.parameters():
        p.requires_grad = True
    full = LinearProbeClassifier(full_enc, freeze_encoder=False).to(dev)
    _train_epochs(full, train_loader, dev, adapt_epochs, lr=1e-3)
    full_acc = _accuracy(full, val_loader, dev)
    full_params = count_trainable(full)

    results = {
        "device": str(dev),
        "dataset": "SyntheticEOPatches (4-class EO-like textures)",
        "class_names": class_names,
        "n_per_class": n_per_class,
        "pretrain_epochs": pretrain_epochs,
        "adapt_epochs": adapt_epochs,
        "backbone_implemented": "TinyEOEncoder (local CNN stand-in for FM)",
        "backbone_recommended_next": [
            "IBM/NASA Prithvi-EO (Hugging Face) — multispectral FM",
            "SatMAE / SatMAE++ — masked autoencoder for RS",
            "TorchSpatial / location encoders — spatial inductive bias",
            "RemoteCLIP / SkyCLIP — vision-language EO",
        ],
        "methods": [
            {"name": "linear_probe", "val_accuracy": round(probe_acc, 4), "trainable_params": probe_params},
            {"name": "lora_r4", "val_accuracy": round(lora_acc, 4), "trainable_params": lora_params},
            {"name": "full_finetune", "val_accuracy": round(full_acc, 4), "trainable_params": full_params},
        ],
    }

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    csv_path = out_dir / "metrics_table.csv"
    lines = ["method,val_accuracy,trainable_params"]
    for m in results["methods"]:
        lines.append(f"{m['name']},{m['val_accuracy']},{m['trainable_params']}")
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    names = [m["name"] for m in results["methods"]]
    accs = [m["val_accuracy"] for m in results["methods"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, accs, color=["#4cc9f0", "#4361ee", "#3a0ca3"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Validation accuracy")
    ax.set_title("geo-foundation-adapt — probe vs LoRA vs full (synthetic EO)")
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.02, f"{a:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig_path = fig_dir / "adapt_accuracy.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "tiny_eo_encoder.pt"
    torch.save({"encoder": base_encoder.state_dict(), "meta": results}, ckpt_path)

    results["artifacts"] = {
        "metrics_json": str(metrics_path),
        "metrics_csv": str(csv_path),
        "figure": str(fig_path),
        "checkpoint": str(ckpt_path),
    }
    return results
