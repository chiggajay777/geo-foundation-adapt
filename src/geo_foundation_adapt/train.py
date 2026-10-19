"""Offline synthetic demo. The EuroSAT benchmark lives in ``benchmark.py``."""

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
            correct += (model(x).argmax(dim=1) == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def _train_epochs(model: nn.Module, loader: DataLoader, device: torch.device, epochs: int, lr: float) -> None:
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()


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
    """Synthetic textures and a local CNN. This is not the EuroSAT table."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    dev = torch.device(device or "cpu")
    train_loader, val_loader, class_names = make_loaders(
        n_per_class=n_per_class, batch_size=batch_size, seed=seed
    )

    encoder = TinyEOEncoder()
    warm = LinearProbeClassifier(encoder, n_classes=4, freeze_encoder=False).to(dev)
    _train_epochs(warm, train_loader, dev, pretrain_epochs, lr=1e-3)
    base_encoder = deepcopy(warm.encoder).cpu()
    for p in base_encoder.parameters():
        p.requires_grad = False

    probe = LinearProbeClassifier(deepcopy(base_encoder), freeze_encoder=True).to(dev)
    _train_epochs(probe, train_loader, dev, adapt_epochs, lr=1e-2)
    lora = LoRAProbeClassifier(deepcopy(base_encoder), rank=4).to(dev)
    _train_epochs(lora, train_loader, dev, adapt_epochs, lr=1e-2)
    full_enc = deepcopy(base_encoder)
    for p in full_enc.parameters():
        p.requires_grad = True
    full = LinearProbeClassifier(full_enc, freeze_encoder=False).to(dev)
    _train_epochs(full, train_loader, dev, adapt_epochs, lr=1e-3)

    results = {
        "demo": True,
        "device": str(dev),
        "dataset": "SyntheticEOPatches (4-class textures, not EuroSAT)",
        "class_names": class_names,
        "n_per_class": n_per_class,
        "pretrain_epochs": pretrain_epochs,
        "adapt_epochs": adapt_epochs,
        "backbone_implemented": "TinyEOEncoder (local CNN, not a pretrained EO model)",
        "methods": [
            {"name": "linear_probe", "val_accuracy": round(_accuracy(probe, val_loader, dev), 4), "trainable_params": count_trainable(probe)},
            {"name": "lora_r4", "val_accuracy": round(_accuracy(lora, val_loader, dev), 4), "trainable_params": count_trainable(lora)},
            {"name": "full_finetune", "val_accuracy": round(_accuracy(full, val_loader, dev), 4), "trainable_params": count_trainable(full)},
        ],
        "note": "Demo numbers are synthetic. The EuroSAT table is the default command without --demo.",
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    csv_path = out_dir / "metrics_table.csv"
    lines = ["method,val_accuracy,trainable_params"]
    for row in results["methods"]:
        lines.append(f"{row['name']},{row['val_accuracy']},{row['trainable_params']}")
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 4))
    names = [row["name"] for row in results["methods"]]
    accs = [row["val_accuracy"] for row in results["methods"]]
    bars = ax.bar(names, accs, color=["#4cc9f0", "#4361ee", "#3a0ca3"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Synthetic demo only — not the EuroSAT benchmark")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.02, f"{acc:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig_path = fig_dir / "adapt_accuracy.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)

    results["artifacts"] = {
        "metrics_json": str(metrics_path),
        "metrics_csv": str(csv_path),
        "figure": str(fig_path),
    }
    return results
