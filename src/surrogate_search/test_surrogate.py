import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.surrogate_search import SurrogateAssistedSearch, XGBoostSurrogate
from src.surrogate_search.evaluate_selected import select_best_surrogate


ARCHITECTURE_A = {
    "stage_blocks": [1, 2, 3, 1],
    "stage_channels": [16, 64, 256, 128],
    "kernel_sizes": [3, 5, 3, 5],
    "dropout": 0.2,
}
ARCHITECTURE_B = {
    "stage_blocks": [2, 1, 2, 3],
    "stage_channels": [32, 128, 64, 512],
    "kernel_sizes": [5, 3, 5, 3],
    "dropout": 0.5,
}


class RecordingRegressor:
    def __init__(self, predictions=(0.75,)):
        self.predictions = predictions
        self.fit_x = None
        self.fit_y = None

    def fit(self, x, y):
        self.fit_x = np.asarray(x)
        self.fit_y = np.asarray(y)
        return self

    def predict(self, x):
        return np.asarray(self.predictions[: len(x)], dtype=float)


class XGBoostSurrogateTests(unittest.TestCase):
    def test_encode_architecture_uses_all_thirteen_genes(self):
        encoded = XGBoostSurrogate.encode_architecture(ARCHITECTURE_A)

        self.assertEqual(
            encoded.tolist(),
            [1, 2, 3, 1, 16, 64, 256, 128, 3, 5, 3, 5, 0.2],
        )

    def test_fit_records_accepts_search_summary_records(self):
        regressor = RecordingRegressor()
        surrogate = XGBoostSurrogate(regressor=regressor)
        records = [
            {"architecture": ARCHITECTURE_A, "val_macro_auc_ovr": 0.71},
            {"architecture": ARCHITECTURE_B, "val_macro_auc_ovr": 0.83},
        ]

        surrogate.fit_records(records)

        self.assertEqual(regressor.fit_x.shape, (2, 13))
        np.testing.assert_allclose(regressor.fit_y, [0.71, 0.83])

    def test_predict_clips_auc_to_metric_range(self):
        surrogate = XGBoostSurrogate(regressor=RecordingRegressor((1.2, -0.1)))
        surrogate.fit([ARCHITECTURE_A, ARCHITECTURE_B], [0.7, 0.8])

        predictions = surrogate.predict([ARCHITECTURE_A, ARCHITECTURE_B])

        np.testing.assert_allclose(predictions, [1.0, 0.0])

    def test_predict_before_fit_is_rejected(self):
        surrogate = XGBoostSurrogate(regressor=RecordingRegressor())

        with self.assertRaisesRegex(RuntimeError, "fitted"):
            surrogate.predict_one(ARCHITECTURE_A)

    def test_as_fitness_callback_returns_predicted_auc_and_parameter_count(self):
        surrogate = XGBoostSurrogate(regressor=RecordingRegressor((0.79,)))
        surrogate.fit([ARCHITECTURE_A, ARCHITECTURE_B], [0.7, 0.8])
        callback = surrogate.as_fitness_callback(parameter_counter=lambda _: 12345)

        fitness = callback(ARCHITECTURE_A, 42)

        self.assertEqual(fitness, (0.79, 12345))


class RecordingSurrogate:
    def __init__(self):
        self.fit_sizes = []

    def fit_records(self, records):
        self.fit_sizes.append(len(list(records)))
        return self

    def predict(self, architectures):
        return np.asarray([
            sum(item["stage_blocks"]) / 12 for item in architectures
        ])


class SurrogateAssistedSearchTests(unittest.TestCase):
    def test_run_builds_training_data_only_from_current_real_evaluations(self):
        calls = []
        surrogate = RecordingSurrogate()

        def real_fitness(architecture, seed):
            calls.append((architecture, seed))
            return sum(architecture["stage_blocks"]) / 12, 1000

        search = SurrogateAssistedSearch(
            real_fitness,
            surrogate=surrogate,
            parameter_counter=lambda _: 1000,
            warmup_size=4,
            acquisition_batch_size=2,
            candidate_pool_size=8,
            population_size=4,
            seed=42,
        )

        result = search.run(evaluation_budget=8)

        self.assertEqual(len(calls), 8)
        self.assertEqual(len(result["evaluated"]), 8)
        self.assertEqual(surrogate.fit_sizes, [4, 6, 8])
        self.assertEqual(result["warmup_evaluations"], 4)
        self.assertEqual(search.acquisition_batch_size, 2)

    def test_run_rejects_budget_smaller_than_warmup(self):
        search = SurrogateAssistedSearch(
            lambda architecture, seed: (0.5, 1000),
            surrogate=RecordingSurrogate(),
            parameter_counter=lambda _: 1000,
            warmup_size=4,
            candidate_pool_size=8,
        )

        with self.assertRaisesRegex(ValueError, "warmup"):
            search.run(evaluation_budget=3)


class SurrogateFinalEvaluationTests(unittest.TestCase):
    def test_select_best_surrogate_uses_validation_auc_and_parameter_tiebreak(self):
        summary = {
            "experiment": "surrogate_search",
            "config": {"seed": 42},
            "evaluations": 40,
            "pareto_front": [
                {
                    "architecture": ARCHITECTURE_A,
                    "val_macro_auc_ovr": 0.91,
                    "parameter_count": 2000,
                },
                {
                    "architecture": ARCHITECTURE_B,
                    "val_macro_auc_ovr": 0.91,
                    "parameter_count": 1000,
                },
            ],
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(json.dumps(summary), encoding="utf-8")

            architecture, selection = select_best_surrogate(path, expected_seed=42)

        self.assertEqual(architecture, ARCHITECTURE_B)
        self.assertEqual(selection["rule"], "maximum_validation_macro_auc_ovr")
        self.assertEqual(selection["search_evaluations"], 40)
        self.assertEqual(selection["search_parameter_count"], 1000)

    def test_select_best_surrogate_rejects_wrong_seed(self):
        summary = {
            "experiment": "surrogate_search",
            "config": {"seed": 7},
            "evaluations": 40,
            "pareto_front": [{
                "architecture": ARCHITECTURE_A,
                "val_macro_auc_ovr": 0.91,
                "parameter_count": 2000,
            }],
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(json.dumps(summary), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "seed"):
                select_best_surrogate(path, expected_seed=42)


if __name__ == "__main__":
    unittest.main()
