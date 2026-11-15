"""Online surrogate-assisted NSGA-II without historical evaluation data."""

import math
import random
from copy import deepcopy

from src.nsga2 import Individual, crowding_distance, nondominated_sort, select_survivors
from src.search_space import (
    canonical_architecture,
    mutate_architecture,
    random_architecture,
    uniform_crossover,
)
from src.surrogate_search.xgboost_surrogate import XGBoostSurrogate


class SurrogateAssistedSearch:
    """Alternate surrogate prediction and real candidate evaluation.

    The training set starts empty for every instance. A random warm-up is
    evaluated with the real fitness callback before the first surrogate fit.
    No summary, cache, or result from another search is read by this class.
    """

    def __init__(
        self,
        real_fitness_callback,
        *,
        surrogate=None,
        parameter_counter=None,
        warmup_size: int = 20,
        acquisition_batch_size: int = 5,
        candidate_pool_size: int = 100,
        population_size: int = 10,
        crossover_rate: float = 0.8,
        mutation_probability: float = 0.15,
        exploration_fraction: float = 0.2,
        seed: int = 42,
    ) -> None:
        for name, value in (
            ("warmup_size", warmup_size),
            ("acquisition_batch_size", acquisition_batch_size),
            ("candidate_pool_size", candidate_pool_size),
            ("population_size", population_size),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if candidate_pool_size < acquisition_batch_size:
            raise ValueError(
                "candidate_pool_size must be at least acquisition_batch_size"
            )
        for name, value in (
            ("crossover_rate", crossover_rate),
            ("mutation_probability", mutation_probability),
            ("exploration_fraction", exploration_fraction),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")

        self.real_fitness_callback = real_fitness_callback
        self.surrogate = surrogate or XGBoostSurrogate(random_state=seed)
        self.parameter_counter = parameter_counter or self._default_parameter_counter
        self.warmup_size = warmup_size
        self.acquisition_batch_size = acquisition_batch_size
        self.candidate_pool_size = candidate_pool_size
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_probability = mutation_probability
        self.exploration_fraction = exploration_fraction
        self.seed = seed
        self.rng = random.Random(seed)
        self._evaluated = {}

    @staticmethod
    def _default_parameter_counter(architecture: dict) -> int:
        from src.model import build_model, count_parameters

        return count_parameters(build_model(architecture))

    def _evaluate_real(self, architectures: list[dict]) -> None:
        for architecture in architectures:
            key = canonical_architecture(architecture)
            if key in self._evaluated:
                continue
            auc, parameters = self.real_fitness_callback(deepcopy(architecture), self.seed)
            if not math.isfinite(auc) or not 0 <= auc <= 1:
                raise ValueError("real fitness AUC must be finite and in [0, 1]")
            if type(parameters) is not int or parameters <= 0:
                raise ValueError("real parameter count must be a positive integer")
            self._evaluated[key] = Individual(
                deepcopy(architecture), (float(auc), parameters)
            )

    def _training_records(self) -> list[dict]:
        return [
            {
                "architecture": deepcopy(individual.architecture),
                "val_macro_auc_ovr": individual.fitness[0],
                "parameter_count": individual.fitness[1],
            }
            for individual in self._evaluated.values()
        ]

    def _unseen_random(self, count: int, excluded=()) -> list[dict]:
        result = []
        keys = set(self._evaluated) | set(excluded)
        while len(result) < count:
            architecture = random_architecture(self.rng)
            key = canonical_architecture(architecture)
            if key not in keys:
                keys.add(key)
                result.append(architecture)
        return result

    def _candidate_pool(self, population: list[Individual]) -> list[dict]:
        candidates = {}
        attempts = 0
        while len(candidates) < self.candidate_pool_size and attempts < 10000:
            attempts += 1
            if len(population) >= 2:
                first, second = self.rng.sample(population, 2)
                child = (
                    uniform_crossover(first.architecture, second.architecture, self.rng)
                    if self.rng.random() < self.crossover_rate
                    else deepcopy(first.architecture)
                )
                child = mutate_architecture(
                    child, self.rng, self.mutation_probability
                )
            else:
                child = random_architecture(self.rng)
            key = canonical_architecture(child)
            if key not in self._evaluated:
                candidates[key] = child
        if len(candidates) < self.candidate_pool_size:
            for child in self._unseen_random(
                self.candidate_pool_size - len(candidates), candidates
            ):
                candidates[canonical_architecture(child)] = child
        return list(candidates.values())

    def _select_candidates(self, candidates: list[dict], count: int) -> list[dict]:
        predictions = self.surrogate.predict(candidates)
        predicted = [
            Individual(
                deepcopy(architecture),
                (float(auc), self.parameter_counter(architecture)),
            )
            for architecture, auc in zip(candidates, predictions)
        ]
        ranked = []
        for front in nondominated_sort(predicted):
            crowding_distance(front)
            ranked.extend(sorted(front, key=lambda item: item.crowding, reverse=True))

        exploration_count = min(count, math.ceil(count * self.exploration_fraction))
        exploitation_count = count - exploration_count
        selected = ranked[:exploitation_count]
        selected_keys = {
            canonical_architecture(individual.architecture) for individual in selected
        }
        remainder = [
            individual for individual in ranked[exploitation_count:]
            if canonical_architecture(individual.architecture) not in selected_keys
        ]
        selected.extend(self.rng.sample(remainder, exploration_count))
        return [deepcopy(individual.architecture) for individual in selected]

    def run(self, *, evaluation_budget: int) -> dict:
        if type(evaluation_budget) is not int or evaluation_budget < self.warmup_size:
            raise ValueError("evaluation_budget must be an integer at least warmup_size")
        if self._evaluated:
            raise RuntimeError("a SurrogateAssistedSearch instance can only run once")

        self._evaluate_real(self._unseen_random(self.warmup_size))
        self.surrogate.fit_records(self._training_records())
        population = select_survivors(
            list(self._evaluated.values()),
            min(self.population_size, len(self._evaluated)),
        )
        iterations = 0

        while len(self._evaluated) < evaluation_budget:
            remaining = min(
                self.acquisition_batch_size,
                evaluation_budget - len(self._evaluated),
            )
            candidates = self._candidate_pool(population)
            selected = self._select_candidates(candidates, remaining)
            self._evaluate_real(selected)
            self.surrogate.fit_records(self._training_records())
            population = select_survivors(
                list(self._evaluated.values()),
                min(self.population_size, len(self._evaluated)),
            )
            iterations += 1

        archive = list(self._evaluated.values())
        fronts = nondominated_sort(archive)
        for front in fronts:
            crowding_distance(front)
        return {
            "seed": self.seed,
            "evaluations": len(archive),
            "warmup_evaluations": self.warmup_size,
            "surrogate_iterations": iterations,
            "evaluated": archive,
            "pareto_front": fronts[0],
            "population": population,
        }
