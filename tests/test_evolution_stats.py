"""Validate generation replay with the legacy and compute-aware objectives."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.evolution_stats import analyze_generations, plot_generational_progress
from src.hardware_analysis import OBJECTIVES
from src.model import BASELINE_ARCHITECTURE


class GenerationAnalysisTests(unittest.TestCase):
    def test_legacy_summary_replays_partial_generation_and_renders_png(self):
        evaluations = [
            {
                "evaluation": number,
                "architecture": {**BASELINE_ARCHITECTURE, "dropout": dropout},
                "val_macro_auc_ovr": auc,
                "parameter_count": parameters,
            }
            for number, dropout, auc, parameters in (
                (1, 0.0, 0.80, 100),
                (2, 0.1, 0.70, 120),
                (3, 0.2, 0.90, 110),
            )
        ]
        generations = analyze_generations({
            "config": {"population_size": 2},
            "evaluated": evaluations[::-1],
        })

        self.assertEqual(len(generations), 2)
        self.assertEqual(generations[0]["survivor_evaluations"], [1, 2])
        self.assertEqual(generations[1]["survivor_evaluations"], [1, 3])
        self.assertEqual(generations[1]["first_evaluation"], 3)
        self.assertEqual(generations[1]["last_evaluation"], 3)
        self.assertEqual(generations[1]["gene_frequencies"]["dropout"], {"0.0": 0.5, "0.2": 0.5})
        self.assertAlmostEqual(generations[1]["mean_normalized_hamming_distance"], 1 / 13)

        with TemporaryDirectory() as directory:
            path = Path(directory) / "generational_progression.png"
            plot_generational_progress(path, evaluations, generations)
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_macs_change_survival_in_hardware_aware_summary(self):
        evaluations = [
            {
                "evaluation": number,
                "architecture": {**BASELINE_ARCHITECTURE, "dropout": dropout},
                "val_macro_auc_ovr": auc,
                "parameter_count": parameters,
                "macs": macs,
            }
            for number, dropout, auc, parameters, macs in (
                (1, 0.0, 0.80, 100, 50),
                (2, 0.1, 0.90, 90, 500),
            )
        ]
        config = {"population_size": 1}
        legacy = analyze_generations({"config": config, "evaluated": evaluations})
        hardware = analyze_generations({
            "config": {**config, "objectives": OBJECTIVES},
            "evaluated": evaluations,
        })
        self.assertEqual(legacy[1]["survivor_evaluations"], [2])
        self.assertEqual(hardware[1]["survivor_evaluations"], [1])
