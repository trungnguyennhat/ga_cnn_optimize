"""Real-valued GA with bounded SBX and polynomial mutation."""

import math
import random


BOUNDS = {
    "log10_lr": (-4.0, math.log10(3e-3)),
    "log10_wd": (-5.0, -2.0),
    "class_weight_alpha": (0.0, 0.5),
}
INITIAL_CHROMOSOMES = ((-3.0, -2.0, 0.0), (-3.0, -3.0, 0.5))
CROSSOVER_RATE = 0.9
CROSSOVER_ETA = 15
MUTATION_ETA = 20
MUTATION_RATE = 1 / len(BOUNDS)
TOURNAMENT_SIZE = 3

Chromosome = tuple[float, float, float]


def decode(chromosome: Chromosome) -> dict[str, float]:
    log10_lr, log10_wd, alpha = chromosome
    return {
        "learning_rate": 10 ** log10_lr,
        "weight_decay": 10 ** log10_wd,
        "class_weight_alpha": alpha,
    }


def sbx(first: Chromosome, second: Chromosome, rng: random.Random):
    if rng.random() >= CROSSOVER_RATE:
        return first, second

    child_a, child_b = [], []
    exponent = 1 / (CROSSOVER_ETA + 1)
    for a, b, (lower, upper) in zip(first, second, BOUNDS.values()):
        if rng.random() < 0.5 and abs(a - b) > 1e-14:
            left, right = sorted((a, b))
            distance = right - left
            draw = rng.random()

            def spread(gap):
                beta = 1 + 2 * gap / distance
                alpha = 2 - beta ** -(CROSSOVER_ETA + 1)
                if draw <= 1 / alpha:
                    return (draw * alpha) ** exponent
                return (1 / (2 - draw * alpha)) ** exponent

            midpoint = (left + right) / 2
            a = max(lower, midpoint - spread(left - lower) * distance / 2)
            b = min(upper, midpoint + spread(upper - right) * distance / 2)
            if rng.random() < 0.5:
                a, b = b, a
        child_a.append(a)
        child_b.append(b)
    return tuple(child_a), tuple(child_b)


def polynomial_mutation(chromosome: Chromosome, rng: random.Random) -> Chromosome:
    mutated = []
    power = MUTATION_ETA + 1
    for value, (lower, upper) in zip(chromosome, BOUNDS.values()):
        if rng.random() < MUTATION_RATE:
            width = upper - lower
            draw = rng.random()
            if draw < 0.5:
                distance = (upper - value) / width
                delta = (2 * draw + (1 - 2 * draw) * distance ** power) ** (1 / power) - 1
            else:
                distance = (value - lower) / width
                delta = 1 - (2 * (1 - draw) + (2 * draw - 1) * distance ** power) ** (1 / power)
            value = min(upper, max(lower, value + delta * width))
        mutated.append(value)
    return tuple(mutated)


def run_ga(fitness, *, budget: int, population_size: int, seed: int) -> int:
    """Evaluate unique chromosomes and return the number of offspring generations."""
    if budget < 1 or population_size < 2:
        raise ValueError("budget must be positive and population_size must be at least 2")
    rng = random.Random(seed)
    seen = set()

    def evaluate(chromosome, generation):
        # Unchanged offspring can occur when both genetic operators leave them intact.
        while chromosome is None or chromosome in seen:
            chromosome = tuple(rng.uniform(lower, upper) for lower, upper in BOUNDS.values())
        score = fitness(chromosome, generation)
        seen.add(chromosome)
        return score, chromosome

    population = [
        evaluate(INITIAL_CHROMOSOMES[index] if index < len(INITIAL_CHROMOSOMES) else None, 0)
        for index in range(min(budget, population_size))
    ]

    def select():
        tournament = rng.choices(population, k=TOURNAMENT_SIZE)
        return max(tournament, key=lambda individual: individual[0])[1]

    generation = 0
    while len(seen) < budget:
        generation += 1
        offspring = []
        remaining = min(population_size, budget - len(seen))
        while len(offspring) < remaining:
            children = sbx(select(), select(), rng)
            for child in children[:remaining - len(offspring)]:
                offspring.append(evaluate(polynomial_mutation(child, rng), generation))
        population = sorted(population + offspring, key=lambda individual: individual[0],
                            reverse=True)[:population_size]
    return generation
