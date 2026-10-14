"""Tiny synthetic EO-like patch dataset (no download required).

Four land-cover-style classes with distinct spatial textures so a short
CPU demo can separate them. Replace with EuroSAT / SSL4EO / etc. for research.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


CLASS_NAMES = ["water", "vegetation", "urban", "bare_soil"]


def _make_patch(rng: np.random.Generator, class_id: int, size: int = 64) -> np.ndarray:
    """Return CHW float32 in [0, 1], C=3."""
    img = rng.normal(0.40, 0.12, size=(size, size, 3)).astype(np.float32)
    yy, xx = np.mgrid[0:size, 0:size]

    if class_id == 0:  # water — smooth blue-ish + horizontal ripples
        img[..., 2] += 0.18
        img[..., 0] -= 0.05
        img += 0.04 * np.sin(xx / 4.0)[..., None]
    elif class_id == 1:  # vegetation — green speckles
        img[..., 1] += 0.18
        speck = rng.random((size, size)) > 0.7
        img[speck, 1] += 0.08
    elif class_id == 2:  # urban — grid + bright roofs
        img += 0.06
        img[(yy % 8) < 2] += 0.10
        img[(xx % 8) < 2] += 0.06
        img[..., 0] += 0.04
    else:  # bare soil — warm tones + low freq noise
        img[..., 0] += 0.14
        img[..., 1] += 0.06
        img[..., 2] -= 0.02
        img += 0.05 * np.sin((xx + yy) / 12.0)[..., None]

    img += rng.normal(0, 0.10, img.shape).astype(np.float32)
    return np.clip(img, 0, 1).transpose(2, 0, 1)


class SyntheticEOPatches(Dataset):
    def __init__(self, n_per_class: int = 200, size: int = 64, seed: int = 42):
        self.size = size
        self.samples: list[tuple[np.ndarray, int]] = []
        rng = np.random.default_rng(seed)
        for c in range(4):
            for i in range(n_per_class):
                self.samples.append((_make_patch(rng, c, size), c))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y = self.samples[idx]
        return torch.from_numpy(x), y


def make_loaders(
    n_per_class: int = 200,
    batch_size: int = 32,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, list[str]]:
    ds = SyntheticEOPatches(n_per_class=n_per_class, seed=seed)
    n_val = max(1, int(0.2 * len(ds)))
    n_train = len(ds) - n_val
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=gen)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, CLASS_NAMES
