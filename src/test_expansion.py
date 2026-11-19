"""Regression checks for costs, Pareto ranking and fidelity decisions."""

import random
import unittest

import torch
from torch import nn

from src.model import BASELINE_ARCHITECTURE, _conv_macs, build_model
from src.multi_fidelity import RungScheduler
from src.nsga2 import Individual, nondominated_sort
from src.objectives import MACS, PARAMS, architecture_costs, cost_bounds, hypervolume
from src.search_space import mutate_architecture, random_architecture, uniform_crossover, validate_architecture


class ExpansionTests(unittest.TestCase):
    def test_legacy_costs_and_grouped_convolution(self):
        self.assertEqual(
            architecture_costs(BASELINE_ARCHITECTURE, 7),
            {PARAMS: 11_172_423, MACS: 2_221_673_984},
        )
        with torch.device("meta"):
            conv = nn.Conv2d(4, 4, 3, stride=2, padding=1, groups=4)
        self.assertEqual(_conv_macs(conv, (8, 8)), (576, (4, 4)))

    def test_hybrid_operators_preserve_valid_architectures_and_cost_bounds(self):
        rng = random.Random(42)
        parents = [random_architecture(rng, "v2") for _ in range(2)]
        bounds = cost_bounds("v2", 8)
        for _ in range(20):
            child = mutate_architecture(uniform_crossover(*parents, rng), rng, probability=0.5)
            validate_architecture(child)
            for key, value in architecture_costs(child, 8).items():
                low, high = bounds[key]
                self.assertTrue(low <= value <= high)
            parents = [parents[1], child]
        with torch.device("meta"):
            model = build_model(parents[-1], 8).eval()
            self.assertEqual(tuple(model(torch.empty((2, 3, 64, 64))).shape), (2, 8))

    def test_third_objective_changes_dominance_and_hypervolume(self):
        a = Individual({}, (0.9, 100, 400))
        b = Individual({}, (0.8, 200, 100))
        c = Individual({}, (0.7, 300, 500))
        self.assertEqual(nondominated_sort([a, b, c]), [[a, b], [c]])
        self.assertAlmostEqual(hypervolume([(0.5, 0.5, 0.5), (1.0, 0.25, 0.5)]), 0.1875)

    def test_rung_decision_uses_same_fidelity_observations(self):
        def history(auc, epochs):
            return [{"val_macro_auc_ovr": auc}] * epochs

        scheduler = RungScheduler((5, 10), reduction_factor=2, grace_trials=2)
        scheduler.observe(history(0.9, 10), {}, (100,))
        scheduler.observe(history(0.8, 5), {}, (110,))
        self.assertFalse(scheduler.continue_trial(history(0.2, 5), {}, (120,)))
        self.assertTrue(scheduler.continue_trial(history(0.2, 10), {}, (120,)))


if __name__ == "__main__":
    unittest.main()
