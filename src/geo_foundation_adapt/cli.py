"""CLI for the EuroSAT benchmark and the offline synthetic demo."""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import argparse
import sys
from pathlib import Path

from geo_foundation_adapt.benchmark import BenchmarkConfig, run_benchmark
from geo_foundation_adapt.train import run_demo


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(part) for part in text.split(",") if part.strip())


def _ints(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part.strip())


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(
        prog="geo-foundation-adapt",
        description=(
            "Label-efficiency comparison of a linear probe, LoRA, and full finetune "
            "on EuroSAT RGB with a pretrained ResNet-18. --demo stays on synthetic data."
        ),
    )
    parser.add_argument("--demo", action="store_true", help="synthetic TinyEOEncoder demo; no download")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/eurosat"))
    parser.add_argument("--weight-dir", type=Path, default=Path("data/weights"))
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--label-fractions", type=str, default="0.01,0.1,1")
    parser.add_argument("--epochs", type=int, default=8, help="max epochs for LoRA and full finetune")
    parser.add_argument("--patience", type=int, default=3, help="early-stop patience for LoRA and full finetune")
    parser.add_argument("--probe-epochs", type=int, default=30, help="max epochs for the linear head")
    parser.add_argument("--probe-patience", type=int, default=6, help="early-stop patience for the linear head")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--pool-per-class",
        type=int,
        default=200,
        help="stratified cap drawn from the spatial train split; fractions are of this pool. 0 keeps every train chip",
    )
    parser.add_argument(
        "--val-per-class",
        type=int,
        default=50,
        help="stratified validation slice used only for checkpoint selection. 0 keeps the full spatial val band",
    )
    parser.add_argument(
        "--max-test-per-class",
        type=int,
        default=0,
        help="0 evaluates the full spatial test band",
    )
    parser.add_argument("--backbone", choices=("seco", "imagenet"), default="seco")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--publish-figures", type=Path, default=None)
    parser.add_argument("--n-per-class", type=int, default=200, help="demo only")
    parser.add_argument("--pretrain-epochs", type=int, default=3, help="demo only")
    parser.add_argument("--adapt-epochs", type=int, default=5, help="demo only")
    args = parser.parse_args(argv)

    if args.demo:
        print("geo-foundation-adapt: synthetic demo (not the EuroSAT table)")
        results = run_demo(
            args.out,
            n_per_class=args.n_per_class,
            pretrain_epochs=args.pretrain_epochs,
            adapt_epochs=args.adapt_epochs,
            batch_size=min(args.batch_size, 32),
            device=args.device,
        )
        print(f"demo artifacts: {results['artifacts']}")
        return 0

    if min(args.epochs, args.patience, args.probe_epochs, args.probe_patience) < 1:
        parser.error("epoch and patience values must be >= 1")
    cfg = BenchmarkConfig(
        out_dir=args.out,
        data_dir=args.data_dir,
        weight_dir=args.weight_dir,
        seeds=_ints(args.seeds),
        label_fractions=_floats(args.label_fractions),
        epochs=args.epochs,
        patience=args.patience,
        probe_epochs=args.probe_epochs,
        probe_patience=args.probe_patience,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        image_size=args.image_size,
        pool_per_class=args.pool_per_class,
        val_per_class=args.val_per_class,
        max_test_per_class=args.max_test_per_class,
        backbone=args.backbone,
        device=args.device,
        num_threads=args.num_threads,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        publish_figures=args.publish_figures,
        argv=list(argv),
    )
    run_benchmark(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
