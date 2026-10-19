# EuroSAT label-efficiency results

Mean ± sample standard deviation across seeds (divisor n−1). Accuracy and macro-F1 are on the full spatial test band. The checkpoint is the best validation macro-F1 inside the epoch budget.

| Labels / class | Fraction of CPU pool | Method | Accuracy | Macro-F1 | Trainable params |
|---:|---:|---|---:|---:|---:|
| — | — | spatial-train majority | 0.094 | 0.017 | 0 |
| 2 | 1% | Linear probe | 0.594 ± 0.086 | 0.539 ± 0.060 | 5130 |
| 2 | 1% | LoRA (r=8, layer4) | 0.535 ± 0.086 | 0.484 ± 0.046 | 150538 |
| 2 | 1% | Full finetune | 0.624 ± 0.072 | 0.549 ± 0.039 | 11181642 |
| 20 | 10% | Linear probe | 0.839 ± 0.018 | 0.780 ± 0.017 | 5130 |
| 20 | 10% | LoRA (r=8, layer4) | 0.765 ± 0.010 | 0.663 ± 0.017 | 150538 |
| 20 | 10% | Full finetune | 0.815 ± 0.001 | 0.755 ± 0.008 | 11181642 |
| 200 | 100% | Linear probe | 0.895 ± 0.003 | 0.861 ± 0.005 | 5130 |
| 200 | 100% | LoRA (r=8, layer4) | 0.895 ± 0.011 | 0.855 ± 0.012 | 150538 |
| 200 | 100% | Full finetune | 0.909 ± 0.004 | 0.887 ± 0.006 | 11181642 |
