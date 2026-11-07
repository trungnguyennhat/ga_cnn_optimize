"""Unit tests for NSGA-II core components."""

import random
import unittest

from src.nsga2 import (
    Individual,
    crowding_distance,
    dominates,
    nondominated_sort,
    select_survivors,
    tournament_selection,
)


class TestNSGA2(unittest.TestCase):
    def test_dominance_logic(self):
        # Format: (auc, params) -> maximize auc, minimize params
        a = (0.95, 1000)
        b = (0.90, 2000)
        c = (0.95, 2000)
        d = (0.90, 1000)

        self.assertTrue(dominates(a, b))
        self.assertTrue(dominates(a, c))
        self.assertTrue(dominates(a, d))
        self.assertFalse(dominates(b, a))
        self.assertFalse(dominates(c, d))  # c has higher auc, d has fewer params -> mutually non-dominated

    def test_nondominated_sort(self):
        ind1 = Individual(architecture={}, fitness=(0.95, 1000))
        ind2 = Individual(architecture={}, fitness=(0.90, 2000))
        ind3 = Individual(architecture={}, fitness=(0.92, 1500))

        fronts = nondominated_sort([ind1, ind2, ind3])
        # ind1 dominates both ind2 and ind3, so ind1 is in front 0
        self.assertIn(ind1, fronts[0])
        self.assertEqual(len(fronts[0]), 1)

    def test_crowding_distance(self):
        ind1 = Individual(architecture={}, fitness=(0.90, 1000))
        ind2 = Individual(architecture={}, fitness=(0.92, 2000))
        ind3 = Individual(architecture={}, fitness=(0.95, 3000))

        front = [ind1, ind2, ind3]
        crowding_distance(front)

        # Boundary points must receive infinity
        self.assertEqual(ind1.crowding, float("inf"))
        self.assertEqual(ind3.crowding, float("inf"))
        self.assertTrue(ind2.crowding > 0)

    def test_tournament_selection(self):
        rng = random.Random(42)
        ind_good = Individual(architecture={}, fitness=(0.95, 1000), rank=0, crowding=1.5)
        ind_bad = Individual(architecture={}, fitness=(0.80, 5000), rank=2, crowding=0.5)

        winner = tournament_selection([ind_good, ind_bad], rng, size=2)
        self.assertEqual(winner, ind_good)


if __name__ == "__main__":
    unittest.main()
