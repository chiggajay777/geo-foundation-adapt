"""EuroSAT label-efficiency benchmark: linear probe, LoRA, full finetune."""

from __future__ import annotations

import json
import math
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from geo_foundation_adapt import __version__
from geo_foundation_adapt.data import (
    CLASS_NAMES,
    EUROSAT_RGB_MD5,
    EUROSAT_RGB_URL,
    EUROSAT_RGB_ZIP_BYTES,
    EuroSATSplit,
    archive_bytes,
    nested_subsets,
    per_class_counts,
    preprocess_batch,
    stratified_pool,
)
from geo_foundation_adapt.metrics import mean_std, scores
from geo_foundation_adapt.models import (
    apply_train_mode,
    build_resnet18,
    count_trainable,
    resnet_embedding,
)
from geo_foundation_adapt.reporting import write_reports

SECO_URL = (
    "https://huggingface.co/torchgeo/resnet18_sentinel2_rgb_seco/resolve/"
    "f8dcee692cf7142163b55a5c197d981fe0e717a0/resnet18_sentinel2_rgb_seco-cefca942.pth"
)
SECO_SHA256 = "cefca942e0765c9b6dea779e83bb328c8535dc9a3d49fb90e8b294525e8ef891"
SECO_BYTES = 44_781_193

METHODS = ("linear_probe", "lora", "full_finetune")
# 2 images/class is one optimizer step per epoch at batch size 64.
# Eight epochs is then eight updates, which is not a finished LoRA or finetune.
MIN_ADAPT_STEPS = 40


@dataclass
class BenchmarkConfig:
    out_dir: Path = Path("results")
    data_dir: Path = Path("data/eurosat")
    weight_dir: Path = Path("data/weights")
    seeds: tuple[int, ...] = (0, 1, 2)
    label_fractions: tuple[float, ...] = (0.01, 0.1, 1.0)
    epochs: int = 8
    patience: int = 3
    probe_epochs: int = 30
    probe_patience: int = 6
    batch_size: int = 64
    eval_batch_size: int = 128
    image_size: int = 224
    pool_per_class: int = 200
    val_per_class: int = 50
    max_test_per_class: int = 0
    backbone: str = "seco"
    device: str = "cpu"
    num_threads: int = 4
    lora_rank: int = 8
    lora_alpha: float = 16.0
    pool_seed: int = 0
    val_seed: int = 0
    test_seed: int = 0
    probe_lr: float = 1e-2
    head_lr: float = 1e-3
    backbone_lr: float = 1e-4
    lora_lr: float = 1e-3
    weight_decay: float = 1e-4
    publish_figures: Path | None = None
    argv: list[str] = field(default_factory=list)


def configure_threads(n: int) -> int:
    n = max(1, int(n))
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return int(torch.get_num_threads())


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_backbone(cfg: BenchmarkConfig) -> tuple[dict[str, torch.Tensor], dict]:
    if cfg.backbone == "seco":
        from geo_foundation_adapt.data import _download

        path = cfg.weight_dir / "resnet18_sentinel2_rgb_seco-cefca942.pth"
        _download(SECO_URL, path)
        digest = _sha256(path)
        if digest != SECO_SHA256:
            raise RuntimeError(f"SeCo checkpoint sha256 {digest} != {SECO_SHA256}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = {k: v for k, v in state.items() if not k.startswith("fc.")}
        meta = {
            "id": "seco",
            "name": "ResNet-18 SeCo (Sentinel-2 RGB)",
            "architecture": "torchvision resnet18",
            "pretraining": "Seasonal Contrast on Sentinel-2 RGB (Manas et al., ICCV 2021)",
            "citation": "https://arxiv.org/abs/2103.16607",
            "weights_url": SECO_URL,
            "weights_sha256": digest,
            "weights_bytes": int(path.stat().st_size),
            "source": "TorchGeo ResNet18_Weights.SENTINEL2_RGB_SECO",
            "is_foundation_model": False,
            "foundation_model_note": (
                "SeCo ResNet-18 is a self-supervised Sentinel-2 encoder with 11M parameters. "
                "It is not a transformer foundation model. Prithvi, SatMAE, DOFA, and Clay "
                "are not downloaded here."
            ),
        }
        return state, meta
    if cfg.backbone == "imagenet":
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
        state = {k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith("fc.")}
        meta = {
            "id": "imagenet",
            "name": "ResNet-18 ImageNet-1K V1",
            "architecture": "torchvision resnet18",
            "pretraining": "Supervised ImageNet-1K (torchvision ResNet18_Weights.IMAGENET1K_V1)",
            "citation": "https://arxiv.org/abs/1512.03385",
            "weights_url": getattr(weights, "url", ""),
            "weights_sha256": "",
            "weights_bytes": None,
            "source": "torchvision",
            "is_foundation_model": False,
            "foundation_model_note": (
                "ImageNet ResNet-18 is a supervised natural-image CNN, not an Earth-observation "
                "foundation model. It is the control backbone for 8-bit RGB."
            ),
        }
        return state, meta
    raise ValueError(f"unknown backbone {cfg.backbone}")


def _batches(indices: np.ndarray, batch_size: int, shuffle: bool, rng: np.random.Generator) -> list[np.ndarray]:
    order = np.array(indices, dtype=np.int64, copy=True)
    if shuffle:
        rng.shuffle(order)
    return [order[start : start + batch_size] for start in range(0, len(order), batch_size)]


@torch.inference_mode()
def _predict_images(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    indices: np.ndarray,
    image_size: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    for batch in _batches(indices, batch_size, shuffle=False, rng=np.random.default_rng(0)):
        x = preprocess_batch(images, batch, image_size).to(device)
        pred = model(x).argmax(dim=1).cpu().numpy()
        preds.append(pred)
    return np.concatenate(preds) if preds else np.zeros((0,), dtype=np.int64)


@torch.inference_mode()
def _embed(
    model: nn.Module,
    images: torch.Tensor,
    indices: np.ndarray,
    image_size: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    chunks: list[torch.Tensor] = []
    for batch in _batches(indices, batch_size, shuffle=False, rng=np.random.default_rng(0)):
        x = preprocess_batch(images, batch, image_size).to(device)
        chunks.append(resnet_embedding(model, x).cpu())
    if not chunks:
        return torch.empty((0, 512))
    return torch.cat(chunks, dim=0)


def planned_budget(method: str, n_train: int, cfg: BenchmarkConfig) -> tuple[int, int]:
    """Return (max_epochs, patience) for this subset.

    LoRA and full finetune keep ``cfg.epochs`` when the subset already yields
    enough updates, and otherwise extend until ``MIN_ADAPT_STEPS`` updates.
    The extension is capped at 8× the base epoch count.
    """
    if method == "linear_probe":
        return cfg.probe_epochs, cfg.probe_patience
    steps_per_epoch = max(1, math.ceil(n_train / cfg.batch_size))
    needed = math.ceil(MIN_ADAPT_STEPS / steps_per_epoch)
    epochs = min(max(cfg.epochs, needed), cfg.epochs * 8)
    return epochs, cfg.patience


def _optimizer(model: nn.Module, method: str, cfg: BenchmarkConfig) -> torch.optim.Optimizer:
    head = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("fc.")]
    other = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("fc.")]
    groups = []
    if method == "linear_probe":
        groups.append({"params": head, "lr": cfg.probe_lr})
    elif method == "lora":
        groups.append({"params": head + other, "lr": cfg.lora_lr})
    else:
        if other:
            groups.append({"params": other, "lr": cfg.backbone_lr})
        groups.append({"params": head, "lr": cfg.head_lr})
    return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)


def _train_model(
    model: nn.Module,
    method: str,
    images: torch.Tensor,
    labels: torch.Tensor,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    cfg: BenchmarkConfig,
    seed: int,
    device: torch.device,
    embeddings: torch.Tensor | None = None,
    embed_pos: dict[int, int] | None = None,
) -> dict:
    """Early-stop on validation macro-F1. Test labels are not read here."""
    model.to(device)
    opt = _optimizer(model, method, cfg)
    loss_fn = nn.CrossEntropyLoss()
    best_f1 = -1.0
    best_state = None
    best_epoch = 0
    best_val_acc = 0.0
    stale = 0
    epochs_ran = 0
    n_classes = len(CLASS_NAMES)
    y_val = labels.numpy()[val_idx]
    max_epochs, patience = planned_budget(method, len(train_idx), cfg)

    for epoch in range(1, max_epochs + 1):
        epochs_ran = epoch
        apply_train_mode(model, method)
        rng = np.random.default_rng(seed * 100_003 + epoch)
        total_loss = 0.0
        seen = 0
        if method == "linear_probe":
            assert embeddings is not None and embed_pos is not None
            positions = np.asarray([embed_pos[int(i)] for i in train_idx], dtype=np.int64)
            order = rng.permutation(len(positions))
            positions = positions[order]
            targets = labels.numpy()[train_idx][order]
            head = model.fc
            head.train()
            for start in range(0, len(positions), cfg.batch_size):
                sl = slice(start, start + cfg.batch_size)
                xb = embeddings[positions[sl]].to(device)
                yb = torch.as_tensor(targets[sl], dtype=torch.long, device=device)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(head(xb), yb)
                loss.backward()
                opt.step()
                total_loss += float(loss.item()) * yb.shape[0]
                seen += int(yb.shape[0])
            with torch.inference_mode():
                val_pos = np.asarray([embed_pos[int(i)] for i in val_idx], dtype=np.int64)
                val_logits = []
                for start in range(0, len(val_pos), cfg.eval_batch_size):
                    xb = embeddings[val_pos[start : start + cfg.eval_batch_size]].to(device)
                    val_logits.append(head(xb).argmax(dim=1).cpu().numpy())
                val_pred = np.concatenate(val_logits)
        else:
            for batch in _batches(train_idx, cfg.batch_size, shuffle=True, rng=rng):
                x = preprocess_batch(images, batch, cfg.image_size).to(device)
                y = labels[torch.as_tensor(batch, dtype=torch.long)].to(device)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(x), y)
                loss.backward()
                opt.step()
                total_loss += float(loss.item()) * y.shape[0]
                seen += int(y.shape[0])
            val_pred = _predict_images(
                model, images, labels, val_idx, cfg.image_size, cfg.eval_batch_size, device
            )
        val = scores(y_val, val_pred, n_classes)
        improved = float(val["macro_f1"]) > best_f1
        marker = "*" if improved else ""
        print(
            f"  epoch {epoch:02d} loss={total_loss / max(seen, 1):.4f} "
            f"val_acc={val['accuracy']:.3f} val_f1={val['macro_f1']:.3f}{marker}",
            flush=True,
        )
        if improved:
            best_f1 = float(val["macro_f1"])
            best_val_acc = float(val["accuracy"])
            best_epoch = epoch
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    return {
        "best_epoch": best_epoch,
        "epochs_ran": epochs_ran,
        "max_epochs": max_epochs,
        "val_accuracy": best_val_acc,
        "val_macro_f1": best_f1,
    }


def _evaluate_split(
    model: nn.Module,
    method: str,
    images: torch.Tensor,
    labels: torch.Tensor,
    indices: np.ndarray,
    cfg: BenchmarkConfig,
    device: torch.device,
    embeddings: torch.Tensor | None,
    embed_pos: dict[int, int] | None,
) -> dict:
    y_true = labels.numpy()[indices]
    if method == "linear_probe":
        assert embeddings is not None and embed_pos is not None
        positions = np.asarray([embed_pos[int(i)] for i in indices], dtype=np.int64)
        preds = []
        head = model.fc
        head.eval()
        with torch.inference_mode():
            for start in range(0, len(positions), cfg.eval_batch_size):
                xb = embeddings[positions[start : start + cfg.eval_batch_size]].to(device)
                preds.append(head(xb).argmax(dim=1).cpu().numpy())
        y_pred = np.concatenate(preds) if preds else np.zeros((0,), dtype=np.int64)
    else:
        y_pred = _predict_images(
            model, images, labels, indices, cfg.image_size, cfg.eval_batch_size, device
        )
    return scores(y_true, y_pred, len(CLASS_NAMES))


def _majority_baseline(train_labels: np.ndarray, test_labels: np.ndarray) -> dict:
    counts = np.bincount(train_labels, minlength=len(CLASS_NAMES))
    # Ties break toward the smaller class index.
    majority = int(np.argmax(counts))
    pred = np.full(test_labels.shape, majority, dtype=np.int64)
    scored = scores(test_labels, pred, len(CLASS_NAMES))
    return {
        "rule": "most frequent class in the full spatial train split; ties break to the smaller class index",
        "class_index": majority,
        "class_name": CLASS_NAMES[majority],
        "train_count": int(counts[majority]),
        "test_accuracy": scored["accuracy"],
        "test_macro_f1": scored["macro_f1"],
    }


def run_benchmark(cfg: BenchmarkConfig) -> dict:
    started = time.perf_counter()
    threads = configure_threads(cfg.num_threads)
    device = torch.device(cfg.device)
    print(
        f"geo-foundation-adapt {__version__}: EuroSAT RGB, backbone={cfg.backbone}, "
        f"device={device}, threads={threads}",
        flush=True,
    )
    load_started = time.perf_counter()
    dataset = EuroSATSplit(cfg.data_dir)
    labels_np = dataset.labels.numpy()
    pool = stratified_pool(
        dataset.split_indices["train"], labels_np, cfg.pool_per_class, cfg.pool_seed, len(CLASS_NAMES)
    )
    val_idx = stratified_pool(
        dataset.split_indices["val"], labels_np, cfg.val_per_class, cfg.val_seed, len(CLASS_NAMES)
    )
    test_idx = stratified_pool(
        dataset.split_indices["test"], labels_np, cfg.max_test_per_class, cfg.test_seed, len(CLASS_NAMES)
    )
    state, backbone_meta = load_backbone(cfg)
    load_seconds = time.perf_counter() - load_started
    print(
        f"loaded {len(dataset.filenames)} images in {load_seconds:.1f}s | "
        f"pool {len(pool)} val {len(val_idx)} test {len(test_idx)}",
        flush=True,
    )

    # Frozen embeddings are shared by every linear probe. LoRA and full finetune do not use them.
    probe_model = build_resnet18(state, n_classes=len(CLASS_NAMES), method="linear_probe", seed=0)
    probe_model.to(device)
    embed_indices = np.unique(np.concatenate([pool, val_idx, test_idx]))
    print(f"extracting frozen embeddings for {len(embed_indices)} chips", flush=True)
    embed_started = time.perf_counter()
    flat_emb = _embed(probe_model, dataset.images, embed_indices, cfg.image_size, cfg.eval_batch_size, device)
    embed_pos = {int(index): position for position, index in enumerate(embed_indices.tolist())}
    print(f"embeddings ready in {time.perf_counter() - embed_started:.1f}s", flush=True)
    del probe_model

    runs: list[dict] = []
    for seed in cfg.seeds:
        subsets = nested_subsets(pool, labels_np, list(cfg.label_fractions), seed, len(CLASS_NAMES))
        for fraction in cfg.label_fractions:
            train_idx = subsets[fraction]
            counts = per_class_counts(train_idx, labels_np, len(CLASS_NAMES))
            labels_per_class = counts[0] if len(set(counts)) == 1 else -1
            for method in METHODS:
                max_epochs, _patience = planned_budget(method, len(train_idx), cfg)
                print(
                    f"seed={seed} fraction={fraction:g} n_train={len(train_idx)} "
                    f"method={method} max_epochs={max_epochs}",
                    flush=True,
                )
                t0 = time.perf_counter()
                model = build_resnet18(
                    state,
                    n_classes=len(CLASS_NAMES),
                    method=method,
                    seed=seed,
                    lora_rank=cfg.lora_rank,
                    lora_alpha=cfg.lora_alpha,
                )
                trainable = count_trainable(model)
                fit = _train_model(
                    model,
                    method,
                    dataset.images,
                    dataset.labels,
                    train_idx,
                    val_idx,
                    cfg,
                    seed,
                    device,
                    embeddings=flat_emb if method == "linear_probe" else None,
                    embed_pos=embed_pos if method == "linear_probe" else None,
                )
                test = _evaluate_split(
                    model, method, dataset.images, dataset.labels, test_idx, cfg, device, flat_emb, embed_pos
                )
                train_scores = _evaluate_split(
                    model, method, dataset.images, dataset.labels, train_idx, cfg, device, flat_emb, embed_pos
                )
                seconds = time.perf_counter() - t0
                print(
                    f"  test_acc={test['accuracy']:.3f} test_f1={test['macro_f1']:.3f} "
                    f"best_epoch={fit['best_epoch']}/{fit['epochs_ran']} {seconds:.1f}s",
                    flush=True,
                )
                runs.append(
                    {
                        "seed": int(seed),
                        "label_fraction": float(fraction),
                        "labels_per_class": int(labels_per_class),
                        "labels_per_class_by_name": {
                            name: counts[i] for i, name in enumerate(CLASS_NAMES)
                        },
                        "n_train": int(len(train_idx)),
                        "method": method,
                        "test_accuracy": test["accuracy"],
                        "test_macro_f1": test["macro_f1"],
                        "test_confusion_matrix": test["confusion_matrix"],
                        "val_accuracy": fit["val_accuracy"],
                        "val_macro_f1": fit["val_macro_f1"],
                        "train_accuracy": train_scores["accuracy"],
                        "train_macro_f1": train_scores["macro_f1"],
                        "best_epoch": fit["best_epoch"],
                        "epochs_ran": fit["epochs_ran"],
                        "max_epochs": fit["max_epochs"],
                        "trainable_params": int(trainable),
                        "seconds": seconds,
                    }
                )
                del model

    summary = []
    for fraction in cfg.label_fractions:
        for method in METHODS:
            group = [row for row in runs if row["label_fraction"] == fraction and row["method"] == method]
            acc_mean, acc_std = mean_std([row["test_accuracy"] for row in group])
            f1_mean, f1_std = mean_std([row["test_macro_f1"] for row in group])
            summary.append(
                {
                    "label_fraction": float(fraction),
                    "labels_per_class": group[0]["labels_per_class"],
                    "n_train": group[0]["n_train"],
                    "method": method,
                    "test_accuracy_mean": acc_mean,
                    "test_accuracy_std": acc_std,
                    "test_macro_f1_mean": f1_mean,
                    "test_macro_f1_std": f1_std,
                    "trainable_params": group[0]["trainable_params"],
                    "n_seeds": len(group),
                }
            )

    cm_totals = []
    for fraction in cfg.label_fractions:
        for method in METHODS:
            mats = [
                np.asarray(row["test_confusion_matrix"], dtype=np.int64)
                for row in runs
                if row["label_fraction"] == fraction and row["method"] == method
            ]
            cm_totals.append(
                {
                    "label_fraction": float(fraction),
                    "method": method,
                    "confusion_matrix_sum_over_seeds": np.sum(mats, axis=0).tolist(),
                }
            )

    baseline = _majority_baseline(labels_np[dataset.split_indices["train"]], labels_np[test_idx])
    total_seconds = time.perf_counter() - started
    zip_bytes = archive_bytes(dataset)
    payload = {
        "version": __version__,
        "argv": cfg.argv,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "torch_num_threads": threads,
        "seconds_total": total_seconds,
        "seconds_data_load": load_seconds,
        "dataset": {
            "name": "EuroSAT RGB",
            "citation": "https://arxiv.org/abs/1709.00029",
            "journal": "https://ieeexplore.ieee.org/document/8736785",
            "url": EUROSAT_RGB_URL,
            "md5": EUROSAT_RGB_MD5,
            "zip_bytes_canonical": EUROSAT_RGB_ZIP_BYTES,
            "zip_bytes_on_disk": zip_bytes,
            "archive_md5_verified": bool(dataset.archive_verified),
            "n_images": int(len(dataset.filenames)),
            "image_size_native": 64,
            "image_size_model": int(cfg.image_size),
            "classes": CLASS_NAMES,
            "split": "TorchGeo EuroSATSpatial longitude bands, 60/20/20",
            "split_counts": {name: int(len(idx)) for name, idx in dataset.split_indices.items()},
            "spatial_train_counts": dataset.counts(dataset.split_indices["train"]),
            "spatial_val_counts": dataset.counts(dataset.split_indices["val"]),
            "spatial_test_counts": dataset.counts(dataset.split_indices["test"]),
            "normalization": "scale uint8 to [0, 1], bilinear resize, ImageNet mean/std",
        },
        "backbone": backbone_meta,
        "adaptation": {
            "methods": list(METHODS),
            "lora_rank": cfg.lora_rank,
            "lora_alpha": cfg.lora_alpha,
            "lora_targets": "ResNet-18 layer4 conv1 and conv2 (3x3). Downsample 1x1 stays frozen.",
            "linear_probe": "Frozen backbone, BatchNorm eval, trained fc only, on cached embeddings.",
            "full_finetune": "All weights trainable, BatchNorm in train mode.",
            "optimizer": "AdamW",
            "probe_lr": cfg.probe_lr,
            "lora_lr": cfg.lora_lr,
            "head_lr": cfg.head_lr,
            "backbone_lr": cfg.backbone_lr,
            "weight_decay": cfg.weight_decay,
            "selection": "best validation macro-F1; ties keep the earlier epoch",
            "epochs": cfg.epochs,
            "patience": cfg.patience,
            "probe_epochs": cfg.probe_epochs,
            "probe_patience": cfg.probe_patience,
            "min_adapt_steps": MIN_ADAPT_STEPS,
            "epoch_budget_note": (
                "Linear probe epochs are a cap on the classification head only. "
                "LoRA and full finetune use the base epoch cap, extended so each run "
                f"gets at least {MIN_ADAPT_STEPS} optimizer steps when the subset is small."
            ),
        },
        "protocol": {
            "seeds": list(cfg.seeds),
            "n_seeds": len(cfg.seeds),
            "label_fractions": list(cfg.label_fractions),
            "nested_subsets": True,
            "pool_per_class_requested": cfg.pool_per_class,
            "pool_seed": cfg.pool_seed,
            "pool_counts": dataset.counts(pool),
            "n_pool": int(len(pool)),
            "val_per_class_requested": cfg.val_per_class,
            "val_seed": cfg.val_seed,
            "val_counts": dataset.counts(val_idx),
            "n_val": int(len(val_idx)),
            "test_per_class_requested": cfg.max_test_per_class,
            "test_counts": dataset.counts(test_idx),
            "n_test": int(len(test_idx)),
            "epochs": cfg.epochs,
            "patience": cfg.patience,
            "probe_epochs": cfg.probe_epochs,
            "probe_patience": cfg.probe_patience,
            "min_adapt_steps": MIN_ADAPT_STEPS,
            "batch_size": cfg.batch_size,
            "eval_batch_size": cfg.eval_batch_size,
            "std": "sample standard deviation, divisor n-1; null when n_seeds < 2",
            "notes": [
                "Label fractions are fractions of the fixed CPU pool, not of all 16,200 spatial-train chips.",
                "Within a seed, the 1% subset is a prefix of the 10% subset, which is a prefix of the 100% subset.",
                "The pool, validation slice, and test slice are drawn once and do not depend on the training seed.",
                "The spatial test band is not used to choose epochs, learning rates, or the backbone.",
                "The validation slice is class-balanced. The spatial test band keeps its natural class counts.",
                "No geometric augmentation. The comparison is the adaptation rule, not a flip policy.",
                "LoRA and full finetune use at least 40 optimizer steps. The 2,000-image run stays at 8 epochs.",
            ],
        },
        "baselines": {
            "uniform_chance_accuracy": 1.0 / len(CLASS_NAMES),
            "spatial_train_majority": baseline,
        },
        "runs": runs,
        "summary": summary,
        "confusion_sums": cm_totals,
    }
    artifacts = write_reports(cfg.out_dir, payload, cfg.publish_figures)
    payload["artifacts"] = artifacts
    Path(artifacts["metrics_json"]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {artifacts['metrics_json']} in {total_seconds:.1f}s", flush=True)
    return payload
