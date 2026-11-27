"""MOEA/D for validation AUC, parameter count and MACs."""

import math
import random
from copy import deepcopy
from itertools import product

from src.ga_search.nsga2 import (
    HARDWARE_DIRECTIONS,
    Individual,
    crowding_distance,
    nondominated_sort,
)
from src.search_space import (
    DROPOUT_CHOICES,
    KERNEL_CHOICES,
    STAGE_BLOCK_CHOICES,
    STAGE_CHANNEL_CHOICES,
    canonical_architecture,
    mutate_architecture,
    random_architecture,
    uniform_crossover,
)


MAX_EVALUATIONS = (
    len(STAGE_BLOCK_CHOICES) ** 4
    * math.prod(map(len, STAGE_CHANNEL_CHOICES))
    * len(KERNEL_CHOICES) ** 4
    * len(DROPOUT_CHOICES)
)


def _all_architectures():
    choices = (STAGE_BLOCK_CHOICES,) * 4 + STAGE_CHANNEL_CHOICES + (KERNEL_CHOICES,) * 4
    for genes in product(*choices, DROPOUT_CHOICES):
        yield {
            "stage_blocks": list(genes[:4]),
            "stage_channels": list(genes[4:8]),
            "kernel_sizes": list(genes[8:12]),
            "dropout": genes[12],
        }


def _objectives(individual):
    auc, parameters, macs = individual.fitness
    return 1 - auc, parameters, macs


def _reference_weights(size):
    if size == 1:
        return [(1 / 3, 1 / 3, 1 / 3)]
    if size == 2:
        return [(1.0, 0.0, 0.0), (0.0, 0.5, 0.5)]

    divisions = 1
    while math.comb(divisions + 2, 2) < size:
        divisions += 1
    corners = [(divisions, 0, 0), (0, divisions, 0), (0, 0, divisions)]
    candidates = [
        (auc, parameters, divisions - auc - parameters)
        for auc in range(divisions + 1)
        for parameters in range(divisions - auc + 1)
        if (auc, parameters, divisions - auc - parameters) not in corners
    ]
    # Keep each objective's extreme and spread any remaining directions across the grid.
    selected = corners.copy()
    distances = {point: min(math.dist(point, corner) for corner in corners) for point in candidates}
    while len(selected) < size:
        point = max(distances, key=distances.__getitem__)
        selected.append(point)
        del distances[point]
        for candidate in distances:
            distances[candidate] = min(distances[candidate], math.dist(candidate, point))
    return [tuple(value / divisions for value in point) for point in selected]


def run_moead(
    fitness_callback,
    *,
    population_size: int = 10,
    evaluation_budget: int = 60,
    neighborhood_size: int = 4,
    crossover_rate: float = 0.8,
    mutation_probability: float = 0.15,
    seed: int = 42,
) -> dict:
    """Evaluate each unique architecture once with callback(architecture, seed).

    The callback returns (validation macro AUC, parameter count, MACs).
    Subproblems minimize normalized weighted Tchebycheff distances to the
    best observed AUC, parameter count and MACs.
    """
    for name, value in (
        ("population_size", population_size),
        ("evaluation_budget", evaluation_budget),
        ("neighborhood_size", neighborhood_size),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if evaluation_budget > MAX_EVALUATIONS:
        raise ValueError("evaluation budget exceeds the number of unique architectures")
    if not 0 <= crossover_rate <= 1 or not 0 <= mutation_probability <= 1:
        raise ValueError("crossover and mutation probabilities must be in [0, 1]")

    rng = random.Random(seed)
    size = min(population_size, evaluation_budget)
    weights = _reference_weights(size)
    neighborhoods = [
        sorted(range(size), key=lambda other: math.dist(weight, weights[other]))[:neighborhood_size]
        for weight in weights
    ]
    evaluated = {}
    ideal = [math.inf] * len(HARDWARE_DIRECTIONS)
    upper = [-math.inf] * len(HARDWARE_DIRECTIONS)
    fallback = _all_architectures()

    def evaluate_unseen(make_candidate):
        for attempt in range(100):
            architecture = make_candidate() if attempt < 50 else random_architecture(rng)
            key = canonical_architecture(architecture)
            if key not in evaluated:
                break
        else:
            for architecture in fallback:
                key = canonical_architecture(architecture)
                if key not in evaluated:
                    break
            else:
                raise RuntimeError("No unevaluated architectures remain")

        auc, parameters, macs = fitness_callback(deepcopy(architecture), seed)
        if not math.isfinite(auc) or not 0 <= auc <= 1:
            raise ValueError("fitness AUC must be finite and in [0, 1]")
        if any(type(cost) is not int or cost <= 0 for cost in (parameters, macs)):
            raise ValueError("parameter count and MACs must be positive integers")
        individual = Individual(architecture, (float(auc), parameters, macs))
        evaluated[key] = individual
        for index, value in enumerate(_objectives(individual)):
            ideal[index] = min(ideal[index], value)
            upper[index] = max(upper[index], value)
        return individual

    def score(individual, weight):
        return max(
            max(coefficient, 1e-6) * (value - best) / (worst - best or 1)
            for coefficient, value, best, worst in zip(weight, _objectives(individual), ideal, upper)
        )

    def offspring(neighbors):
        first, second = rng.choices(neighbors, k=2)
        architecture = population[first].architecture
        if rng.random() < crossover_rate:
            architecture = uniform_crossover(architecture, population[second].architecture, rng)
        return mutate_architecture(architecture, rng, mutation_probability)

    population = [evaluate_unseen(lambda: random_architecture(rng)) for _ in weights]
    generations = 0
    while len(evaluated) < evaluation_budget:
        order = rng.sample(range(size), size)
        for index in order[:evaluation_budget - len(evaluated)]:
            neighbors = neighborhoods[index]
            child = evaluate_unseen(lambda: offspring(neighbors))
            replacements = 0
            for other in rng.sample(neighbors, len(neighbors)):
                weight = weights[other]
                if score(child, weight) < score(population[other], weight):
                    population[other] = child
                    replacements += 1
                    if replacements == 2:
                        break
        generations += 1

    archive = list(evaluated.values())
    fronts = nondominated_sort(archive, HARDWARE_DIRECTIONS)
    for front in fronts:
        crowding_distance(front)
    return {
        "seed": seed,
        "evaluations": len(archive),
        "generations": generations,
        "weights": weights,
        "population": population,
        "evaluated": archive,
        "pareto_front": fronts[0],
    }
