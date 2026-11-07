"""Unit tests for evolutionary metrics and statistical analysis."""

import unittest

from src.evolution_stats import compute_schotts_spacing, statistical_comparison


class TestEvolutionStats(unittest.TestCase):
    def test_schotts_spacing(self):
        # Perfect equidistant front
        pareto = [
            {"val_macro_auc_ovr": 0.90, "parameter_count": 1000},
            {"val_macro_auc_ovr": 0.92, "parameter_count": 2000},
            {"val_macro_auc_ovr": 0.94, "parameter_count": 3000},
        ]
        spacing = compute_schotts_spacing(pareto)
        self.assertIsInstance(spacing, float)
        self.assertTrue(spacing >= 0.0)

    def test_statistical_comparison(self):
        ga_points = [{"val_macro_auc_ovr": 0.94 + i * 0.001} for i in range(10)]
        rand_points = [{"val_macro_auc_ovr": 0.90 + i * 0.001} for i in range(10)]

        res = statistical_comparison(ga_points, rand_points)
        self.assertIn("p_value", res)
        self.assertIn("mann_whitney_u", res)
        self.assertTrue(res["p_value"] < 0.05)


if __name__ == "__main__":
    unittest.main()
