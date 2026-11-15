"""Surrogate models that can be imported by architecture-search pipelines."""

from src.surrogate_search.online_search import SurrogateAssistedSearch
from src.surrogate_search.xgboost_surrogate import XGBoostSurrogate

__all__ = ["SurrogateAssistedSearch", "XGBoostSurrogate"]
