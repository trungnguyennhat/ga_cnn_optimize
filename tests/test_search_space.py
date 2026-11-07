"""Unit tests for search space and genetic operators."""

import random
import unittest

from src.search_space import (
    DROPOUT_CHOICES,
    KERNEL_CHOICES,
    STAGE_BLOCK_CHOICES,
    STAGE_CHANNEL_CHOICES,
    canonical_architecture,
    mutate_architecture,
    random_architecture,
    repair_architecture,
    uniform_crossover,
    validate_architecture,
)


class TestSearchSpace(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(42)

    def test_random_architecture_validity(self):
        for _ in range(50):
            arch = random_architecture(self.rng)
            # Should not raise ValueError
            validate_architecture(arch)
            self.assertEqual(len(arch["stage_blocks"]), 4)
            self.assertEqual(len(arch["stage_channels"]), 4)
            self.assertEqual(len(arch["kernel_sizes"]), 4)
            self.assertIn(arch["dropout"], DROPOUT_CHOICES)

    def test_canonical_architecture_deterministic(self):
        arch1 = random_architecture(self.rng)
        arch2 = dict(arch1)
        self.assertEqual(canonical_architecture(arch1), canonical_architecture(arch2))

    def test_repair_invalid_architecture(self):
        bad_arch = {
            "stage_blocks": [99, 0, 4, -1],
            "stage_channels": [999, 999, 999, 999],
            "kernel_sizes": [7, 1, 9, 2],
            "dropout": 0.99,
        }
        repaired = repair_architecture(bad_arch)
        validate_architecture(repaired)

    def test_uniform_crossover(self):
        parent_a = random_architecture(self.rng)
        parent_b = random_architecture(self.rng)
        child = uniform_crossover(parent_a, parent_b, self.rng)
        validate_architecture(child)
        self.assertIn(child["dropout"], (parent_a["dropout"], parent_b["dropout"]))

    def test_mutate_architecture(self):
        parent = random_architecture(self.rng)
        mutated = mutate_architecture(parent, self.rng, probability=0.8)
        validate_architecture(mutated)


if __name__ == "__main__":
    unittest.main()
