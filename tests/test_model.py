"""Unit tests for ResNet model construction and parameter counting."""

import unittest
import torch

from src.model import BASELINE_ARCHITECTURE, BasicBlock, build_model, count_parameters


class TestModel(unittest.TestCase):
    def test_basic_block(self):
        block = BasicBlock(in_channels=16, out_channels=32, kernel_size=3, stride=2)
        x = torch.zeros(2, 16, 32, 32)
        out = block(x)
        self.assertEqual(out.shape, (2, 32, 16, 16))

    def test_build_baseline_model(self):
        model = build_model(BASELINE_ARCHITECTURE)
        x = torch.zeros(2, 3, 64, 64)
        out = model(x)
        self.assertEqual(out.shape, (2, 7))
        params = count_parameters(model)
        self.assertEqual(params, 11172423)

    def test_build_custom_model(self):
        custom_arch = {
            "stage_blocks": [1, 1, 1, 1],
            "stage_channels": [16, 32, 64, 128],
            "kernel_sizes": [3, 3, 3, 3],
            "dropout": 0.1,
        }
        model = build_model(custom_arch)
        x = torch.zeros(2, 3, 64, 64)
        out = model(x)
        self.assertEqual(out.shape, (2, 7))
        self.assertTrue(count_parameters(model) > 0)


if __name__ == "__main__":
    unittest.main()
