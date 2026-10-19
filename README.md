# geo-foundation-adapt

Label-efficiency comparison on EuroSAT RGB: a frozen linear probe, LoRA on a pretrained ResNet-18, and full finetune. Three training seeds. The table in this file is filled from the CPU run of the command in [Reproduce](#reproduce). `--demo` is a separate synthetic fallback and is not that table.

## Problem

Adapting a pretrained encoder to land cover gets expensive if every new label set updates every weight. The practical question is how much of that update is worth paying for when labels are scarce:

1. Does a linear probe on a frozen encoder already beat a majority-class baseline?
2. Does LoRA on the last residual stage pick up accuracy the probe cannot?
3. At what label count does full finetune overtake the cheaper methods, and what does it cost in trainable weights?

## Data

**EuroSAT RGB** (Helber, Bischke, Dengel, and Borth): 27,000 Sentinel-2 chips, 10 land-cover classes, 64×64 JPEG. The zip used here is the TorchGeo / Hugging Face copy of the RGB archive, 94,280,567 bytes, MD5 `c8fa014336c82ac7804f0398fcb19387`.

The 13-band archive (`EuroSATallBands.zip`) is about 2.07 GB, which is outside the download budget, so this benchmark does not use reflectance values. Inputs are the authors' 8-bit RGB rendering.

The split is **TorchGeo `EuroSATSpatial`**: longitude bands, 60/20/20, filename lists vendored under `src/geo_foundation_adapt/splits/` and checked against the TorchGeo SHA-256 pins.

| Split | Chips |
|---|---:|
| Spatial train | 16,200 |
| Spatial val | 5,400 |
| Spatial test | 5,400 |

Class counts are uneven inside those bands (the spatial test band has 115 Pasture chips and 1,077 Forest chips). Macro-F1 is reported for that reason. A random EuroSAT split is not used.

**CPU pool.** Finetuning all 16,200 train chips at 224×224 does not fit the 4-thread hour, so the published command draws a **fixed stratified pool of 200 chips per class** (2,000 chips) from the spatial train band with `pool_seed=0`. Label fractions **1% / 10% / 100%** are fractions of that pool: **2 / 20 / 200 images per class** (20 / 200 / 2,000 labeled chips). Within a seed the smaller subset is a prefix of the larger one. The pool does not depend on the training seed.

Checkpointing uses a **class-balanced 50-per-class slice** of the spatial validation band (`val_seed=0`). The number that is reported is accuracy and macro-F1 on the **full spatial test band** (5,400 chips). That band is not used to pick epochs, learning rates, or the backbone.

## Backbone

**ResNet-18 trained with Seasonal Contrast (SeCo)** on Sentinel-2 RGB (Manas et al., ICCV 2021). Weights are the TorchGeo checkpoint `ResNet18_Weights.SENTINEL2_RGB_SECO` (44,781,193 bytes, SHA-256 `cefca942e0765c9b6dea779e83bb328c8535dc9a3d49fb90e8b294525e8ef891`), loaded into `torchvision.models.resnet18`. TorchGeo itself is not installed; the checkpoint is one file from Hugging Face.

This is a self-supervised Earth-observation encoder. It is **not** a foundation model in the Prithvi / SatMAE / DOFA / Clay sense: the architecture is an 11M-parameter CNN, and the pretraining task is seasonal contrast on RGB rather than masked modeling of multispectral time series. Those weights are left as a later step. A ViT finetune does not fit this CPU budget, and the matching 13-band EuroSAT zip does not fit the download budget.

SeCo's raw-DN quantile stretch expects reflectance, which these JPEGs no longer are. The published preprocessing is the last stage of that pipeline only: scale the 8-bit chip to [0, 1], bilinear-resize 64→224, then ImageNet normalization. `--backbone imagenet` swaps in torchvision `ResNet18_Weights.IMAGENET1K_V1` without changing the protocol. The published table is SeCo.

## Adaptation

Same new 10-way head for every method. No flip augmentation, so the gap is the adaptation rule. Optimizer is AdamW, weight decay 1e-4. Early stopping keeps the earlier epoch on ties. Selection metric is validation macro-F1. Sample standard deviation uses divisor n−1.

| Method | What trains | Learning rate | BatchNorm |
|---|---|---|---|
| `linear_probe` | classification head | 1e-2 | frozen, eval (pretrained running stats) |
| `lora` | head + LoRA on `layer4` 3×3 convs, rank 8, alpha 16 | 1e-3 | frozen, eval |
| `full_finetune` | every weight | head 1e-3, backbone 1e-4 | train mode, running stats update |

LoRA uses the usual convolution factorization: a k×k map down to the rank, then a 1×1 map back, with the second matrix initialized at 0 so step 0 matches the frozen conv. The residual downsample (1×1) stays frozen. The linear probe trains on embeddings extracted once from that frozen eval-mode network; with no augmentation that matches an online frozen forward. LoRA and full finetune stop after at most 8 epochs with patience 3. The linear head is allowed 30 epochs and patience 6, because an epoch does not touch the ResNet and a probe that is cut off after a handful of steps is not the linear probe this comparison means. Batch size is 64. Seeds are 0, 1, and 2. The head seed is reset per method so method order does not change initialization.

The spatial-train majority baseline predicts the most common class in the full 16,200-chip train band (ties break to the smaller class index) and is scored on the same test chips. It does not depend on the seed. Uniform chance accuracy on 10 classes is 0.10; the spatial split moves class mass, so the train-majority accuracy is not 0.10.

## Reproduce

One block for macOS and for Windows Git Bash. Python 3.12. From the repository root. This command is what writes `results/` and `figures/label_efficiency.png`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
# Windows PowerShell instead of the line above: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
case "$(uname -s)" in
  Darwin) python -m pip install torch torchvision ;;
  *) python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu ;;
esac
python -m pip install -r requirements.txt
python -m pip install -e .
python -m geo_foundation_adapt --out results --data-dir data/eurosat --publish-figures figures
```

The EuroSAT RGB zip is about 90 MB and the SeCo checkpoint is about 43 MB (about 133 MB together). Both are gitignored. Offline synthetic fallback, which does not download anything and is not the table below:

```bash
python -m geo_foundation_adapt --demo --out output/demo
```

## Results

The measured table is written here after the reproduce command finishes. Until that run is committed, do not treat any accuracy number in an older README as this protocol.

## Limitations

- The published label fractions are 1%, 10%, and 100% of a 200-per-class pool, not of the 16,200-chip spatial train split. Two hundred images per class is a CPU cap chosen so a 224×224 ResNet-18 finetune can finish on 4 threads.
- Checkpoint selection sees 50 validation chips per class, not all 5,400 spatial-val chips. The chosen epoch is noisier than a full-val sweep.
- Three seeds cover the labeled subset and the head/LoRA initialization. They are not a bootstrap over longitude bands.
- EuroSAT RGB is an 8-bit JPEG, not 13-band reflectance. SeCo was pretrained on quantile-stretched reflectance. The domain gap is real; ImageNet initialization is the implemented control (`--backbone imagenet`) and is not the published table.
- ResNet-18 SeCo is not Prithvi, SatMAE, DOFA, or Clay.
- BatchNorm in full finetune uses batch statistics. At 2 images per class the batch is smaller than 64, so those updates are a harsh test of full finetune rather than a tuned small-batch recipe.
- Learning rates, rank, the two epoch caps, and the 224 input size were fixed for this CPU protocol. They were not swept on the test band. The linear head has a longer cap than LoRA and full finetune because its epochs are cheap.
- Chips near a longitude cut can still be correlated. The spatial split removes same-location random leakage; it does not make the bands independent landscapes.
- No geometric augmentation. Flips would likely move the low-label full-finetune number and are a separate control.

## License

Code: MIT. EuroSAT images are downloaded at runtime and are not stored in this repository. Cite Helber, Bischke, Dengel, and Borth, "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification," IEEE JSTARS 2019. Sentinel-2 data are Copernicus. SeCo weights follow Manas et al., "Seasonal Contrast: Unsupervised Pre-Training from Uncurated Remote Sensing Data," ICCV 2021, as packaged by TorchGeo.
