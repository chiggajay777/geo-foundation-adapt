"""Unit tests that do not download EuroSAT or pretrained weights."""

from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18

from geo_foundation_adapt.benchmark import BenchmarkConfig, planned_budget
from geo_foundation_adapt.data import SPLIT_SHA256, nested_subsets, per_class_counts, read_split
from geo_foundation_adapt.metrics import macro_f1_from_cm, mean_std, scores
from geo_foundation_adapt.models import LoRAConv2d, build_resnet18, count_trainable


class MetricsTest(unittest.TestCase):
    def test_perfect_and_empty_f1(self):
        perfect = np.eye(3, dtype=np.int64) * 4
        self.assertAlmostEqual(macro_f1_from_cm(perfect), 1.0)
        missed = np.zeros((3, 3), dtype=np.int64)
        missed[0, 1] = 5
        self.assertAlmostEqual(macro_f1_from_cm(missed), 0.0)

    def test_scores_match_labels(self):
        y = np.array([0, 1, 2, 2])
        result = scores(y, y, 3)
        self.assertAlmostEqual(result["accuracy"], 1.0)
        self.assertAlmostEqual(result["macro_f1"], 1.0)

    def test_sample_std_needs_two_points(self):
        mean, std = mean_std([0.5])
        self.assertAlmostEqual(mean, 0.5)
        self.assertIsNone(std)
        mean, std = mean_std([0.0, 1.0])
        self.assertAlmostEqual(mean, 0.5)
        self.assertAlmostEqual(std, 0.70710678118, places=5)


class SplitTest(unittest.TestCase):
    def test_spatial_lists_are_pinned_and_disjoint(self):
        splits = {name: read_split(name) for name in ("train", "val", "test")}
        self.assertEqual(len(splits["train"]), 16200)
        self.assertEqual(len(splits["val"]), 5400)
        self.assertEqual(len(splits["test"]), 5400)
        self.assertEqual(len(set(splits["train"]) | set(splits["val"]) | set(splits["test"])), 27000)
        self.assertEqual(set(SPLIT_SHA256), {"train", "val", "test"})

    def test_nested_fractions_are_prefixes(self):
        labels = np.repeat(np.arange(10), 100)
        pool = np.arange(1000)
        subsets = nested_subsets(pool, labels, [1.0, 0.01, 0.1], seed=3, n_classes=10)
        small, mid, full = subsets[0.01], subsets[0.1], subsets[1.0]
        self.assertTrue(set(small.tolist()).issubset(set(mid.tolist())))
        self.assertTrue(set(mid.tolist()).issubset(set(full.tolist())))
        self.assertEqual(per_class_counts(small, labels, 10), [1] * 10)
        self.assertEqual(per_class_counts(mid, labels, 10), [10] * 10)
        self.assertEqual(per_class_counts(full, labels, 10), [100] * 10)
        again = nested_subsets(pool, labels, [0.01, 0.1, 1.0], seed=3, n_classes=10)
        self.assertTrue(np.array_equal(again[0.1], mid))


class BudgetTest(unittest.TestCase):
    def test_small_subsets_get_a_step_floor(self):
        cfg = BenchmarkConfig()
        self.assertEqual(planned_budget("linear_probe", 20, cfg), (30, 6))
        self.assertEqual(planned_budget("lora", 20, cfg), (40, 3))
        self.assertEqual(planned_budget("full_finetune", 200, cfg), (10, 3))
        self.assertEqual(planned_budget("lora", 2000, cfg), (8, 3))


class AdaptationInitTest(unittest.TestCase):
    def test_lora_conv_is_identity_at_init(self):
        conv = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False)
        wrapped = LoRAConv2d(conv, rank=4, alpha=8.0)
        x = torch.randn(2, 8, 15, 15)
        self.assertTrue(torch.allclose(wrapped(x), conv(x), atol=1e-6))

    def test_methods_match_at_init_and_param_counts_differ(self):
        raw = resnet18(weights=None)
        state = {k: v.clone() for k, v in raw.state_dict().items() if not k.startswith("fc.")}
        full = build_resnet18(state, n_classes=10, method="full_finetune", seed=7)
        lora = build_resnet18(state, n_classes=10, method="lora", seed=7, lora_rank=8, lora_alpha=16.0)
        probe = build_resnet18(state, n_classes=10, method="linear_probe", seed=7)
        x = torch.randn(2, 3, 32, 32)
        full.eval()
        lora.eval()
        probe.eval()
        with torch.no_grad():
            full_y = full(x)
            self.assertTrue(torch.allclose(full_y, lora(x), atol=1e-5))
            self.assertTrue(torch.allclose(full_y, probe(x), atol=1e-5))
        n_probe = count_trainable(probe)
        n_lora = count_trainable(lora)
        n_full = count_trainable(full)
        self.assertEqual(n_probe, 512 * 10 + 10)
        self.assertGreater(n_lora, n_probe)
        self.assertLess(n_lora, 400_000)
        self.assertGreater(n_full, 10_000_000)
        self.assertTrue(all(not p.requires_grad for p in probe.parameters() if p is not probe.fc.weight and p is not probe.fc.bias))


if __name__ == "__main__":
    unittest.main()
