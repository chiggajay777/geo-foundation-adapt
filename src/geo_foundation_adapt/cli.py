"""CLI: python -m geo_foundation_adapt --out output/demo"""

from __future__ import annotations

import argparse
import json
import sys

from geo_foundation_adapt.train import run_demo


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="geo-foundation-adapt")
    p.add_argument("--out", type=str, default="output/demo")
    p.add_argument("--n-per-class", type=int, default=200)
    p.add_argument("--pretrain-epochs", type=int, default=3)
    p.add_argument("--adapt-epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None, help="cpu | cuda (default: auto)")
    args = p.parse_args(argv)

    print("geo-foundation-adapt: starting short demo (CPU/GPU)...")
    results = run_demo(
        args.out,
        n_per_class=args.n_per_class,
        pretrain_epochs=args.pretrain_epochs,
        adapt_epochs=args.adapt_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps({k: results[k] for k in ("device", "backbone_implemented", "methods")}, indent=2))
    print(f"artifacts: {results['artifacts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
