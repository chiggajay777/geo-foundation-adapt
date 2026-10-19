"""EuroSAT RGB loader and the offline synthetic fallback.

Images are the 8-bit EuroSAT RGB zip (Helber et al.). The train/val/test
partition is TorchGeo's longitude split (EuroSATSpatial): 16,200 / 5,400 / 5,400.
Those filename lists are vendored and checked against the TorchGeo SHA-256 hashes.
The multispectral zip is not used; it is about 2.07 GB.
"""

from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

# TorchGeo EuroSATSpatial split files, commit-independent hashes from torchgeo.datasets.eurosat.
SPLIT_SHA256 = {
    "train": "2db7d455afb8dcbca898ea19a00f1f90c091734efdbba89e22aaf24056da243f",
    "val": "6c758477604b7057a0fd990d7f6327b63b99a6725aac11a6a9d0174a7fdd8f0b",
    "test": "de22dec83d350cac3b3e4ca8e285cb6733c81ab94bf5bcf9213a567993402452",
}
SPLIT_FILES = {
    "train": "eurosat-spatial-train.txt",
    "val": "eurosat-spatial-val.txt",
    "test": "eurosat-spatial-test.txt",
}

EUROSAT_RGB_URL = (
    "https://huggingface.co/datasets/torchgeo/eurosat/resolve/"
    "1ce6f1bfb56db63fd91b6ecc466ea67f2509774c/EuroSAT.zip"
)
EUROSAT_RGB_MD5 = "c8fa014336c82ac7804f0398fcb19387"
EUROSAT_RGB_ZIP_BYTES = 94_280_567

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SYNTHETIC_CLASS_NAMES = ["water", "vegetation", "urban", "bare_soil"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    partial = dest.with_suffix(dest.suffix + ".partial")
    print(f"downloading {url}", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "geo-foundation-adapt/0.2"})
    with urllib.request.urlopen(request, timeout=180) as response, partial.open("wb") as handle:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
    partial.replace(dest)


def _split_dir() -> Path:
    return Path(__file__).resolve().parent / "splits"


def read_split(name: str) -> list[str]:
    path = _split_dir() / SPLIT_FILES[name]
    digest = _sha256(path)
    if digest != SPLIT_SHA256[name]:
        raise RuntimeError(f"{path.name} sha256 {digest} does not match the TorchGeo pin")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != len(set(lines)):
        raise RuntimeError(f"duplicate filenames in {path.name}")
    return lines


def ensure_eurosat_rgb(root: Path) -> Path:
    """Return the ``2750/`` image root, downloading the RGB zip when needed."""
    root = Path(root)
    image_root = root / "2750"
    if image_root.is_dir() and sum(1 for _ in image_root.glob("*/*.jpg")) == 27_000:
        return image_root
    zip_path = root / "EuroSAT.zip"
    _download(EUROSAT_RGB_URL, zip_path)
    digest = _md5(zip_path)
    if digest != EUROSAT_RGB_MD5:
        raise RuntimeError(f"EuroSAT.zip md5 {digest} != {EUROSAT_RGB_MD5}")
    print(f"extracting {zip_path}", flush=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(root)
    if not image_root.is_dir():
        raise RuntimeError(f"EuroSAT zip did not contain {image_root}")
    return image_root


class EuroSATSplit:
    """All RGB chips in memory, with spatial-split index arrays."""

    def __init__(self, root: Path):
        image_root = ensure_eurosat_rgb(root)
        by_name: dict[str, Path] = {}
        for class_name in CLASS_NAMES:
            folder = image_root / class_name
            if not folder.is_dir():
                raise RuntimeError(f"missing class folder {folder}")
            for path in folder.glob("*.jpg"):
                by_name[path.name] = path
        if len(by_name) != 27_000:
            raise RuntimeError(f"expected 27000 EuroSAT RGB images, found {len(by_name)}")

        splits = {name: read_split(name) for name in ("train", "val", "test")}
        seen: set[str] = set()
        for name, files in splits.items():
            overlap = seen.intersection(files)
            if overlap:
                raise RuntimeError(f"{name} overlaps another split, example {next(iter(overlap))}")
            seen.update(files)
        if seen != set(by_name):
            missing = len(seen - set(by_name))
            extra = len(set(by_name) - seen)
            raise RuntimeError(f"split lists do not match images (missing_files={missing}, unused_images={extra})")

        names = sorted(by_name)
        self.filenames = names
        self.labels = torch.empty(len(names), dtype=torch.long)
        self.images = torch.empty((len(names), 3, 64, 64), dtype=torch.uint8)
        index_of = {name: i for i, name in enumerate(names)}
        class_to_id = {name: i for i, name in enumerate(CLASS_NAMES)}
        print(f"loading {len(names)} EuroSAT RGB chips from {image_root}", flush=True)
        for seen, (name, path) in enumerate(by_name.items(), start=1):
            prefix = name.split("_", 1)[0]
            with Image.open(path) as image:
                array = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if array.shape != (64, 64, 3):
                raise RuntimeError(f"{path} has shape {array.shape}, expected 64x64x3")
            slot = index_of[name]
            copied = np.array(array, dtype=np.uint8, copy=True)
            self.images[slot] = torch.from_numpy(copied).permute(2, 0, 1)
            self.labels[slot] = class_to_id[prefix]
            if seen % 5000 == 0:
                print(f"  loaded {seen}/{len(names)}", flush=True)

        def indices(files: list[str]) -> np.ndarray:
            return np.asarray([index_of[name] for name in files], dtype=np.int64)

        self.split_indices = {name: indices(files) for name, files in splits.items()}
        self.zip_path = Path(root) / "EuroSAT.zip"
        self.archive_verified = self.zip_path.is_file() and _md5(self.zip_path) == EUROSAT_RGB_MD5

    def counts(self, indices: np.ndarray) -> dict[str, int]:
        labels = self.labels.numpy()[indices]
        return {CLASS_NAMES[c]: int((labels == c).sum()) for c in range(len(CLASS_NAMES))}


def stratified_pool(indices: np.ndarray, labels: np.ndarray, per_class: int, seed: int, n_classes: int) -> np.ndarray:
    """Fixed draw of up to ``per_class`` images from each class. ``per_class <= 0`` keeps every index."""
    rng = np.random.default_rng(seed)
    chosen: list[np.ndarray] = []
    for class_id in range(n_classes):
        pool = indices[labels[indices] == class_id]
        order = rng.permutation(pool)
        if per_class > 0:
            order = order[: min(per_class, len(order))]
        chosen.append(np.asarray(order, dtype=np.int64))
    merged = np.concatenate(chosen)
    rng.shuffle(merged)
    return merged


def nested_subsets(
    pool: np.ndarray,
    labels: np.ndarray,
    fractions: list[float],
    seed: int,
    n_classes: int,
) -> dict[float, np.ndarray]:
    """One permutation per class. Smaller fractions are prefixes of larger ones."""
    rng = np.random.default_rng(seed)
    by_class: list[np.ndarray] = []
    for class_id in range(n_classes):
        cls = pool[labels[pool] == class_id]
        by_class.append(rng.permutation(cls))
    out: dict[float, np.ndarray] = {}
    for fraction in sorted(fractions):
        if fraction <= 0.0 or fraction > 1.0:
            raise ValueError(f"label fraction must be in (0, 1], got {fraction}")
        parts: list[np.ndarray] = []
        for cls in by_class:
            if fraction >= 1.0:
                take = len(cls)
            else:
                take = min(len(cls), max(1, int(round(len(cls) * fraction))))
            parts.append(cls[:take])
        out[fraction] = np.concatenate(parts).astype(np.int64, copy=False)
    return out


def per_class_counts(indices: np.ndarray, labels: np.ndarray, n_classes: int) -> list[int]:
    return [int((labels[indices] == c).sum()) for c in range(n_classes)]


def preprocess_batch(images: torch.Tensor, indices: np.ndarray, image_size: int) -> torch.Tensor:
    """uint8 NCHW cache -> float NCHW, bilinear resize, ImageNet normalization."""
    index = torch.as_tensor(np.asarray(indices, dtype=np.int64), dtype=torch.long)
    batch = images.index_select(0, index).float().div_(255.0)
    if image_size != images.shape[-1] or image_size != images.shape[-2]:
        batch = torch.nn.functional.interpolate(
            batch, size=(image_size, image_size), mode="bilinear", align_corners=False
        )
    mean = batch.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = batch.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (batch - mean) / std


def _make_patch(rng: np.random.Generator, class_id: int, size: int = 64) -> np.ndarray:
    img = rng.normal(0.40, 0.12, size=(size, size, 3)).astype(np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    if class_id == 0:
        img[..., 2] += 0.18
        img[..., 0] -= 0.05
        img += 0.04 * np.sin(xx / 4.0)[..., None]
    elif class_id == 1:
        img[..., 1] += 0.18
        speck = rng.random((size, size)) > 0.7
        img[speck, 1] += 0.08
    elif class_id == 2:
        img += 0.06
        img[(yy % 8) < 2] += 0.10
        img[(xx % 8) < 2] += 0.06
        img[..., 0] += 0.04
    else:
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
        for class_id in range(4):
            for _ in range(n_per_class):
                self.samples.append((_make_patch(rng, class_id, size), class_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image, label = self.samples[idx]
        return torch.from_numpy(image), label


def make_loaders(
    n_per_class: int = 200,
    batch_size: int = 32,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, list[str]]:
    dataset = SyntheticEOPatches(n_per_class=n_per_class, seed=seed)
    n_val = max(1, int(0.2 * len(dataset)))
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, SYNTHETIC_CLASS_NAMES


def archive_bytes(dataset: EuroSATSplit) -> int | None:
    if dataset.zip_path.is_file():
        return int(dataset.zip_path.stat().st_size)
    return None
