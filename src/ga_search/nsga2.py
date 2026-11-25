"""NSGA-II with a supplied fitness callback; no training or data access."""

import math
import random
from copy import deepcopy
from dataclasses import dataclass

from src.search_space import (
    canonical_architecture,
    mutate_architecture,
    random_architecture,
    uniform_crossover,
)


DEFAULT_DIRECTIONS = ("max", "min")
HARDWARE_DIRECTIONS = (*DEFAULT_DIRECTIONS, "min")


@dataclass
class Individual:
    architecture: dict
    fitness: tuple[float | int, ...]
    rank: int = 0
    crowding: float = 0.0


def dominates(first, second, directions=DEFAULT_DIRECTIONS) -> bool:
    if len(first) != len(second) or len(first) != len(directions):
        raise ValueError("fitness lengths must match the objective directions")
    improved = False
    for left, right, direction in zip(first, second, directions):
        if direction == "min":
            left, right = right, left
        elif direction != "max":
            raise ValueError(f"Invalid objective direction: {direction}")
        if left < right:
            return False
        improved |= left > right
    return improved


def nondominated_sort(population: list[Individual], directions=DEFAULT_DIRECTIONS) -> list[list[Individual]]:
    # ponytail: quadratic sorting is enough for populations of 10; optimize if scaled up.
    dominated = [[] for _ in population]
    counts = [0] * len(population)
    for i, first in enumerate(population):
        for j in range(i + 1, len(population)):
            second = population[j]
            if dominates(first.fitness, second.fitness, directions):
                dominated[i].append(j)
                counts[j] += 1
            elif dominates(second.fitness, first.fitness, directions):
                dominated[j].append(i)
                counts[i] += 1
    current = [i for i, count in enumerate(counts) if count == 0]
    fronts = []
    while current:
        front = [population[i] for i in current]
        for individual in front:
            individual.rank = len(fronts)
        fronts.append(front)
        following = []
        for i in current:
            for j in dominated[i]:
                counts[j] -= 1
                if counts[j] == 0:
                    following.append(j)
        current = following
    return fronts


def crowding_distance(front: list[Individual]) -> None:
    for individual in front:
        individual.crowding = 0.0
    if len(front) <= 2:
        for individual in front:
            individual.crowding = math.inf
        return
    for objective in range(len(front[0].fitness)):
        ordered = sorted(front, key=lambda item: item.fitness[objective])
        span = ordered[-1].fitness[objective] - ordered[0].fitness[objective]
        if span == 0:
            continue
        ordered[0].crowding = ordered[-1].crowding = math.inf
        for i in range(1, len(ordered) - 1):
            ordered[i].crowding += (
                ordered[i + 1].fitness[objective] - ordered[i - 1].fitness[objective]
            ) / span


def select_survivors(population: list[Individual], size: int, directions=DEFAULT_DIRECTIONS) -> list[Individual]:
    if type(size) is not int or not 0 <= size <= len(population):
        raise ValueError("survivor size must be an integer between 0 and population length")
    if size == 0:
        return []
    survivors = []
    for front in nondominated_sort(population, directions):
        crowding_distance(front)
        remaining = size - len(survivors)
        survivors.extend(sorted(front, key=lambda item: item.crowding, reverse=True)[:remaining])
        if len(survivors) == size:
            break
    return survivors


def tournament_selection(population: list[Individual], rng, size: int = 3) -> Individual:
    if not population or type(size) is not int or size <= 0:
        raise ValueError("tournament needs a population and positive integer size")
    contenders = rng.sample(population, min(size, len(population)))
    return min(contenders, key=lambda item: (item.rank, -item.crowding))


def run_nsga2(
    fitness_callback,
    *,
    population_size: int = 10,
    evaluation_budget: int = 80,
    tournament_size: int = 3,
    crossover_rate: float = 0.8,
    mutation_probability: float = 0.15,
    seed: int = 1,
    directions=DEFAULT_DIRECTIONS,
) -> dict:
    """Callback(architecture, seed) returns AUC and one or more integer costs.

    Only unseen architectures invoke the callback. The budget counts successful
    callback calls, including the initial population and a partial final generation.
    """
    for name, value in (("population_size", population_size),
                        ("evaluation_budget", evaluation_budget),
                        ("tournament_size", tournament_size)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not 0 <= crossover_rate <= 1 or not 0 <= mutation_probability <= 1:
        raise ValueError("crossover and mutation probabilities must be in [0, 1]")
    # 3^4 block counts, 3^4 channel layouts, 2^4 kernels and 5 dropout values.
    if evaluation_budget > (3**4) * (3**4) * (2**4) * 5:
        raise ValueError("evaluation budget exceeds the number of unique architectures")
    rng = random.Random(seed)
    evaluated = {}

    def evaluate(architecture):
        key = canonical_architecture(architecture)
        if key in evaluated:
            return None
        auc, *costs = fitness_callback(deepcopy(architecture), seed)
        if not math.isfinite(auc) or not 0 <= auc <= 1:
            raise ValueError("fitness AUC must be finite and in [0, 1]")
        if len(costs) + 1 != len(directions) or any(type(cost) is not int or cost <= 0 for cost in costs):
            raise ValueError("fitness must have one positive integer per cost objective")
        individual = Individual(deepcopy(architecture), (float(auc), *costs))
        evaluated[key] = individual
        return individual

    def unseen_candidate(population):
        for attempt in range(1000):
            if population and attempt < 100:
                first = tournament_selection(population, rng, tournament_size)
                second = tournament_selection(population, rng, tournament_size)
                child = (uniform_crossover(first.architecture, second.architecture, rng)
                         if rng.random() < crossover_rate else deepcopy(first.architecture))
                child = mutate_architecture(child, rng, mutation_probability)
            else:
                # ponytail: random fallback prevents duplicate-only stalls; enumerate if near exhaustion.
                child = random_architecture(rng)
            individual = evaluate(child)
            if individual is not None:
                return individual
        raise RuntimeError("Could not find an unseen architecture after 1000 attempts")

    size = min(population_size, evaluation_budget)
    population = [unseen_candidate([]) for _ in range(size)]
    population = select_survivors(population, size, directions)
    generations = 0
    while len(evaluated) < evaluation_budget:
        remaining = min(size, evaluation_budget - len(evaluated))
        offspring = [unseen_candidate(population) for _ in range(remaining)]
        population = select_survivors(population + offspring, size, directions)
        generations += 1
    # Archive Pareto front covers all evaluations, including discarded individuals.
    archive = deepcopy(list(evaluated.values()))
    fronts = nondominated_sort(archive, directions)
    for front in fronts:
        crowding_distance(front)
    pareto = fronts[0]
    return {
        "seed": seed,
        "evaluations": len(evaluated),
        "generations": generations,
        "population": population,
        "evaluated": archive,
        "pareto_front": pareto,
    }
