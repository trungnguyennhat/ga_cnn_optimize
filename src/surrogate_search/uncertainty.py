"""Uncertainty estimates for validation AUC and out-of-fold quality metrics."""

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from src.surrogate_search.xgboost_surrogate import XGBoostSurrogate


class UncertaintySurrogate:
    def __init__(self, backend: str, seed: int):
        self.backend = backend
        self.seed = seed
        self.scaler = StandardScaler()
        if backend == "rf":
            self.model = RandomForestRegressor(
                n_estimators=128, min_samples_leaf=2, random_state=seed, n_jobs=1,
            )
        elif backend == "gp":
            self.model = GaussianProcessRegressor(
                kernel=Matern(nu=2.5, length_scale_bounds="fixed") + WhiteKernel(),
                normalize_y=True, random_state=seed,
            )
        else:
            raise ValueError(f"Unsupported uncertainty backend: {backend}")

    def fit_records(self, records):
        x = XGBoostSurrogate.encode_many(row["architecture"] for row in records)
        y = [XGBoostSurrogate._record_auc(row) for row in records]
        self.model.fit(self.scaler.fit_transform(x), y)
        return self

    def predict_distribution(self, architectures):
        x = self.scaler.transform(XGBoostSurrogate.encode_many(architectures))
        if self.backend == "gp":
            mean, std = self.model.predict(x, return_std=True)
        else:
            predictions = np.asarray([tree.predict(x) for tree in self.model.estimators_])
            mean, std = predictions.mean(axis=0), predictions.std(axis=0, ddof=1)
        return np.clip(mean, 0, 1), std

    def predict(self, architectures):
        return self.predict_distribution(architectures)[0]

    def cross_validation(self, records):
        if len(records) < 4:
            return None
        y = np.asarray([XGBoostSurrogate._record_auc(row) for row in records])
        predictions = np.empty_like(y)
        for train, val in KFold(n_splits=3, shuffle=True, random_state=self.seed).split(records):
            model = UncertaintySurrogate(self.backend, self.seed)
            model.fit_records([records[i] for i in train])
            predictions[val] = model.predict([records[i]["architecture"] for i in val])
        return quality(y, predictions)


def quality(observed, predicted):
    observed, predicted = np.asarray(observed), np.asarray(predicted)
    if not len(observed):
        return None
    residuals = observed - predicted
    rho = spearmanr(observed, predicted).statistic if (
        len(observed) >= 2 and np.ptp(observed) > 0 and np.ptp(predicted) > 0
    ) else None
    return {
        "n": len(observed),
        "mae": float(np.abs(residuals).mean()),
        "rmse": float(np.sqrt(np.square(residuals).mean())),
        "spearman_rho": None if rho is None else float(rho),
    }
