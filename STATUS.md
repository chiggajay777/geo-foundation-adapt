# STATUS — geo-foundation-adapt v0.2.0

**Runnable:** yes. The default command downloads EuroSAT RGB (94,280,567-byte zip, MD5 verified in this run) and the SeCo ResNet-18 Sentinel-2 RGB checkpoint (44,781,193 bytes), then compares a linear probe, layer-4 LoRA, and full finetune.

**Reported run:** Linux x86_64, 4 threads, PyTorch 2.14.1+cpu, device `cpu`, **3,906 seconds**. Metrics are in `results/`. The figure is `figures/label_efficiency.png`.

**Protocol:** TorchGeo longitude split. Label fractions are 1% / 10% / 100% of a fixed 200-per-class pool from the spatial train band (2 / 20 / 200 chips per class). Test metric is the full 5,400-chip spatial test band. Seeds 0, 1, 2.

**Result, briefly:** at 2 chips per class all three methods land near 0.53–0.62 accuracy and the seed noise covers the gaps; the 20 training chips are memorized. At 20 chips per class the linear probe leads (macro-F1 0.780 ± 0.017) and LoRA is behind (0.663 ± 0.017). At 200 chips per class full finetune leads on every seed (macro-F1 0.887 ± 0.006 versus 0.861 ± 0.005 for the probe). The spatial-train majority baseline is 0.094 accuracy on this test band.

**Offline fallback:** `python -m geo_foundation_adapt --demo` (synthetic textures, TinyEOEncoder, not the numbers above).

**Not claimed:** Prithvi, SatMAE, DOFA, or Clay; EuroSAT 13-band reflectance; a finetune of all 16,200 spatial-train chips; a Mac mini timing; a test-set hyperparameter sweep.
