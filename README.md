# geo-foundation-adapt

**Parameter-efficient adaptation for Earth observation backbones**  
Linear probe vs LoRA vs full finetune — a runnable PhD-portfolio demo.

## Problem

EO foundation models (Prithvi, SatMAE, …) are large. Downstream land-cover / hazard tasks need **cheap adaptation** (linear probe / LoRA) with clear metrics vs full finetune.

## What this repo implements

| Item | Choice |
|------|--------|
| **Implemented backbone** | `TinyEOEncoder` — small local CNN, pretrained briefly on synthetic EO-like patches |
| **Adaptation methods** | Linear probe · LoRA (r=4) on projection · full finetune |
| **Dataset** | `SyntheticEOPatches` (4 classes: water / vegetation / urban / bare_soil) — **no download** |
| **Why synthetic first** | Guarantees a short CPU demo; swap in EuroSAT / SSL4EO / real FM weights later |

## Recommended backbones (not auto-downloaded here)

1. **IBM/NASA Prithvi-EO** (Hugging Face) — multispectral foundation model  
2. **SatMAE / SatMAE++** — masked autoencoding for RS  
3. **TorchSpatial** / location encoders — spatial inductive bias for GeoAI  
4. **RemoteCLIP / SkyCLIP** — vision–language EO  

Use those when you have GPU + disk; keep the same probe/LoRA comparison protocol.

## How to run

```bash
cd /workspace/phd-github-portfolio/geo-foundation-adapt
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# ~1–3 minutes on CPU
python -m geo_foundation_adapt --out output/demo

# Faster smoke test
python -m geo_foundation_adapt --out output/smoke --n-per-class 80 --pretrain-epochs 1 --adapt-epochs 2
```

Outputs: `metrics.json`, `metrics_table.csv`, `figures/adapt_accuracy.png`, `checkpoints/tiny_eo_encoder.pt`.

## Layout

```
geo-foundation-adapt/
  src/geo_foundation_adapt/  # data, models, train, cli
  scripts/                   # optional future HF download helpers
  figures/                   # copy key plots for README
  checkpoints/
  data/
```

## Limitations

- TinyEOEncoder is a **stand-in**, not a published FM.
- Synthetic textures ≠ real Sentinel/Landsat statistics.
- LoRA is applied to the projection layer only (pedagogical).

## License

MIT

## Results note

On the synthetic toy set, methods may **saturate** near 1.0 accuracy. Treat **trainable parameter counts** as the PEFT signal; replace `TinyEOEncoder` with Prithvi/SatMAE + real labels for meaningful accuracy gaps.
