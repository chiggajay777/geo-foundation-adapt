# STATUS — geo-foundation-adapt v0.2.0

**Runnable:** yes. The default command downloads EuroSAT RGB (94,280,567-byte zip) and the SeCo ResNet-18 Sentinel-2 RGB checkpoint (44,781,193 bytes), then compares a linear probe, layer-4 LoRA, and full finetune at 1% / 10% / 100% of a fixed 200-per-class spatial-train pool, seeds 0–2.

**Reported run:** not yet committed. The protocol and the reproduce command are in the README. Numbers in `results/` overwrite this file's result line after that run. Do not quote a synthetic demo accuracy as this benchmark.

**Offline fallback:** `python -m geo_foundation_adapt --demo` (synthetic textures, TinyEOEncoder, not the EuroSAT table).

**Not claimed:** Prithvi, SatMAE, DOFA, or Clay adaptation; EuroSAT 13-band reflectance; a full 16,200-image finetune; a test-set hyperparameter sweep.
