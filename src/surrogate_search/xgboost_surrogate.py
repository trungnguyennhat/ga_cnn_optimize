"""XGBoost performance surrogate for the project's ResNet search space."""

import math
from collections.abc import Callable, Iterable, Sequence

import numpy as np

from src.search_space import validate_architecture


class XGBoostSurrogate:
    """Predict validation macro AUC from a ResNet architecture chromosome.

    Training records are supplied in memory by the current search run.
    ``as_fitness_callback`` adapts the fitted surrogate to the callback
    interface used by both search functions.
    """

    FEATURE_NAMES = (
        "blocks_1", "blocks_2", "blocks_3", "blocks_4",
        "channels_1", "channels_2", "channels_3", "channels_4",
        "kernel_1", "kernel_2", "kernel_3", "kernel_4", "dropout",
    )

    def __init__(
        self,
        *,
        regressor=None,
        random_state: int = 42,
        **xgboost_params,
    ) -> None:
        self.random_state = random_state
        self.xgboost_params = dict(xgboost_params)
        self.regressor = regressor if regressor is not None else self._build_regressor()
        self.is_fitted = False

    def _build_regressor(self):
        try:
            from xgboost import XGBRegressor
        except ImportError as error:
            raise ImportError(
                "XGBoostSurrogate requires xgboost; install it with "
                "'python -m pip install xgboost'."
            ) from error

        defaults = {
            "objective": "reg:squarederror",
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": self.random_state,
            "n_jobs": 1,
        }
        defaults.update(self.xgboost_params)
        return XGBRegressor(**defaults)

    @classmethod
    def encode_architecture(cls, architecture: dict) -> np.ndarray:
        """Encode a chromosome with a consistent feature layout within its space."""
        validate_architecture(architecture)
        values = (
            architecture["stage_blocks"]
            + architecture["stage_channels"]
            + architecture["kernel_sizes"]
            + [architecture["dropout"]]
        )
        if "block_types" in architecture:
            values += [kind == "mbconv" for kind in architecture["block_types"]]
            values += architecture["use_se"] + [architecture["activation"] == "silu"]
        return np.asarray(values, dtype=float)

    @classmethod
    def encode_many(cls, architectures: Iterable[dict]) -> np.ndarray:
        rows = [cls.encode_architecture(item) for item in architectures]
        if not rows:
            raise ValueError("architectures must not be empty")
        return np.vstack(rows)

    def fit(
        self,
        architectures: Sequence[dict],
        validation_aucs: Sequence[float],
    ) -> "XGBoostSurrogate":
        if len(architectures) != len(validation_aucs):
            raise ValueError("architectures and validation_aucs must have equal length")
        features = self.encode_many(architectures)
        targets = np.asarray(validation_aucs, dtype=float)
        if targets.ndim != 1 or not np.all(np.isfinite(targets)):
            raise ValueError("validation AUC values must be finite numbers")
        if np.any((targets < 0) | (targets > 1)):
            raise ValueError("validation AUC values must be in [0, 1]")
        self.regressor.fit(features, targets)
        self.is_fitted = True
        return self

    @staticmethod
    def _record_auc(record: dict) -> float:
        if "val_macro_auc_ovr" in record:
            return record["val_macro_auc_ovr"]
        metrics = record.get("metrics", {})
        if "val_macro_auc_ovr" in metrics:
            return metrics["val_macro_auc_ovr"]
        raise ValueError("record does not contain val_macro_auc_ovr")

    def fit_records(self, records: Iterable[dict]) -> "XGBoostSurrogate":
        records = list(records)
        if not records:
            raise ValueError("records must not be empty")
        try:
            architectures = [record["architecture"] for record in records]
        except (KeyError, TypeError) as error:
            raise ValueError("every record must contain an architecture") from error
        return self.fit(architectures, [self._record_auc(record) for record in records])

    def predict(self, architectures: Sequence[dict]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("surrogate must be fitted before prediction")
        predictions = np.asarray(
            self.regressor.predict(self.encode_many(architectures)), dtype=float
        ).reshape(-1)
        if len(predictions) != len(architectures) or not np.all(np.isfinite(predictions)):
            raise ValueError("regressor returned invalid predictions")
        return np.clip(predictions, 0.0, 1.0)

    def predict_one(self, architecture: dict) -> float:
        return float(self.predict([architecture])[0])

    def as_fitness_callback(
        self,
        parameter_counter: Callable[[dict], int] | None = None,
    ) -> Callable[[dict, int], tuple[float, int]]:
        """Return a callback compatible with GA and random-search functions.

        AUC is estimated by XGBoost. Parameter count remains exact because it
        can be computed by constructing the model without training it.
        """
        if not self.is_fitted:
            raise RuntimeError("surrogate must be fitted before creating a callback")
        if parameter_counter is None:
            from src.model import build_model, count_parameters

            def parameter_counter(architecture: dict) -> int:
                return count_parameters(build_model(architecture))

        def fitness(architecture: dict, seed: int) -> tuple[float, int]:
            del seed
            auc = self.predict_one(architecture)
            parameters = parameter_counter(architecture)
            if type(parameters) is not int or parameters <= 0 or not math.isfinite(auc):
                raise ValueError("surrogate fitness must contain valid AUC and parameters")
            return auc, parameters

        return fitness
