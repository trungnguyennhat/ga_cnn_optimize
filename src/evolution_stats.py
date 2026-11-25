"""Replay NSGA-II selection and visualize chromosome diversity."""

from collections import Counter
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.ga_search.nsga2 import DEFAULT_DIRECTIONS, Individual, select_survivors


def analyze_generations(summary: dict) -> list[dict]:
    """Replay NSGA-II survivor selection from evaluations in their original order."""
    config = summary["config"]
    population_size = config["population_size"]
    if type(population_size) is not int or population_size <= 0:
        raise ValueError("population_size must be a positive integer")
    objectives = config.get("objectives", dict(zip(
        ("val_macro_auc_ovr", "parameter_count"), DEFAULT_DIRECTIONS,
    )))
    directions = tuple(objectives.values())
    ordered = sorted(summary["evaluated"], key=lambda point: point["evaluation"])
    if not ordered or [p["evaluation"] for p in ordered] != list(range(1, len(ordered) + 1)):
        raise ValueError("Expected consecutive GA evaluations starting at 1")

    evaluated = [
        Individual(p["architecture"], tuple(p[objective] for objective in objectives))
        for p in ordered
    ]
    evaluation_ids = {id(item): index for index, item in enumerate(evaluated, 1)}
    population = []
    generations = []
    for start in range(0, len(evaluated), population_size):
        candidates = evaluated[start:start + population_size]
        combined = population + candidates
        population = select_survivors(combined, min(population_size, len(combined)), directions)

        counts = {
            f"{gene}[{stage + 1}]": Counter(item.architecture[gene][stage] for item in population)
            for gene in ("stage_blocks", "stage_channels", "kernel_sizes")
            for stage in range(4)
        }
        counts["dropout"] = Counter(item.architecture["dropout"] for item in population)
        size = len(population)
        total_pairs = comb(size, 2)
        gene_diversity = {
            gene: (total_pairs - sum(comb(count, 2) for count in values.values())) / total_pairs
            if total_pairs else 0.0
            for gene, values in counts.items()
        }
        generations.append({
            "generation": len(generations),
            "first_evaluation": start + 1,
            "last_evaluation": start + len(candidates),
            "survivor_evaluations": [evaluation_ids[id(item)] for item in population],
            "mean_normalized_hamming_distance": sum(gene_diversity.values()) / len(gene_diversity),
            "gene_diversity": gene_diversity,
            "gene_frequencies": {
                gene: {str(value): count / size for value, count in sorted(values.items())}
                for gene, values in counts.items()
            },
        })
    return generations


def plot_generational_progress(path: Path, evaluations: list[dict], generations: list[dict]):
    """Plot measured fitness by generation and diversity after survivor selection."""
    fig, (fitness_ax, diversity_ax, gene_ax) = plt.subplots(
        3, 1, figsize=(10, 10), height_ratios=(3, 1.4, 2), layout="constrained",
    )
    points = {point["evaluation"]: point for point in evaluations}
    labels = []
    for row in generations:
        label = f"G{row['generation']}"
        labels.append(label)
        batch = [points[number] for number in range(row["first_evaluation"], row["last_evaluation"] + 1)]
        fitness_ax.scatter(
            [point["parameter_count"] / 1e6 for point in batch],
            [point["val_macro_auc_ovr"] for point in batch],
            alpha=0.8, label=label,
        )
    fitness_ax.set(
        title="GA architectures evaluated in each generation",
        xlabel="Parameters (millions)", ylabel="Validation macro ROC AUC",
    )
    fitness_ax.grid(alpha=0.25)
    fitness_ax.legend(title="Generation", ncols=min(6, len(labels)))

    diversity_ax.plot(labels, [row["mean_normalized_hamming_distance"] for row in generations], "o-")
    diversity_ax.set(title="Diversity of the surviving population", ylabel="Mean Hamming distance", ylim=(0, 1))
    diversity_ax.grid(alpha=0.25)

    loci = list(generations[0]["gene_diversity"])
    heatmap = gene_ax.imshow(
        [[row["gene_diversity"][locus] for locus in loci] for row in generations],
        aspect="auto", cmap="viridis", vmin=0, vmax=1,
    )
    gene_ax.set_xticks(range(len(loci)), loci, rotation=40, ha="right")
    gene_ax.set_yticks(range(len(labels)), labels)
    gene_ax.set(title="Diversity by chromosome position", ylabel="Generation")
    fig.colorbar(heatmap, ax=gene_ax, label="Fraction of pairs with different values")
    fig.savefig(path, dpi=180)
    plt.close(fig)
