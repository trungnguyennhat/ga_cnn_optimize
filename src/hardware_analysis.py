"""Compare compute-aware ResNet searches using validation logs only."""

import csv
from statistics import mean, stdev

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analyze import load_json
from src.ga_search.nsga2 import HARDWARE_DIRECTIONS, dominates
from src.model import BASELINE_ARCHITECTURE, architecture_cost
from src.search_runtime import HARDWARE_CACHE_VERSION, HARDWARE_OBJECTIVES, MAC_CONVENTION, save_json
from src.search_space import DROPOUT_CHOICES, KERNEL_CHOICES, STAGE_BLOCK_CHOICES, STAGE_CHANNEL_CHOICES


METHODS = ("ga_search", "random_search")
PROTOCOL_KEYS = (
    "budget", "epochs", "batch_size", "learning_rate", "optimizer", "scheduler",
    "loss", "normalization", "architecture_space", "train_val_sha256", "device",
)
OBJECTIVES = dict(zip(HARDWARE_OBJECTIVES, HARDWARE_DIRECTIONS))


def cost_bounds():
    return tuple(architecture_cost({
        "stage_blocks": [STAGE_BLOCK_CHOICES[end]] * 4,
        "stage_channels": [choices[end] for choices in STAGE_CHANNEL_CHOICES],
        "kernel_sizes": [KERNEL_CHOICES[end]] * 4,
        "dropout": DROPOUT_CHOICES[end],
    }) for end in (0, -1))


def fitness(point):
    return tuple(point[name] for name in HARDWARE_OBJECTIVES)


def pareto_front(points):
    values = [fitness(point) for point in points]
    return [point for point, target in zip(points, values)
            if not any(dominates(candidate, target, HARDWARE_DIRECTIONS) for candidate in values)]


def coverage(first, second):
    return sum(any(dominates(fitness(left), fitness(right), HARDWARE_DIRECTIONS)
                   for left in first) for right in second) / len(second)


def normalized_fitness(point, bounds):
    auc, parameters, macs = fitness(point)
    low, high = bounds
    if not 0 <= auc <= 1 or not (low[0] <= parameters <= high[0] and low[1] <= macs <= high[1]):
        raise ValueError(f"Fitness outside search-space bounds: {fitness(point)}")
    return auc, (high[0] - parameters) / (high[0] - low[0]), (high[1] - macs) / (high[1] - low[1])


def hypervolume_3d(points):
    """Volume of the union of boxes from the origin to normalized maximizing points."""
    volume, previous_auc = 0.0, 0.0
    for auc in sorted({point[0] for point in points}):
        # In this AUC slab, the remaining two coordinates determine the union area.
        rectangles = sorted(
            ((size, compute) for x, size, compute in points if x >= auc), reverse=True,
        )
        area, highest_compute = 0.0, 0.0
        for size, compute in rectangles:
            if compute > highest_compute:
                area += size * (compute - highest_compute)
                highest_compute = compute
        volume += (auc - previous_auc) * area
        previous_auc = auc
    return volume


def summarize_search(points, bounds):
    front = pareto_front(points)
    scores = [normalized_fitness(point, bounds) for point in points]
    best = max(front, key=lambda point: (point["val_macro_auc_ovr"], -point["parameter_count"]))
    return {
        "pareto_front": front,
        "best_architecture": best["architecture"],
        "metrics": {
            "hypervolume_3d": hypervolume_3d(scores),
            "best_val_auc": best["val_macro_auc_ovr"],
            "params_at_best_val_auc": best["parameter_count"],
            "macs_at_best_val_auc": best["macs"],
            "pareto_size": len(front),
        },
        "pareto_kernel5_fraction_by_stage": [
            sum(point["architecture"]["kernel_sizes"][index] == 5 for point in front) / len(front)
            for index in range(4)
        ],
        "convergence": [hypervolume_3d(scores[:end]) for end in range(1, len(scores) + 1)],
    }


def load_searches(root, seeds, *, budget, epochs, population_size):
    summaries = {}
    protocol = None
    for seed in seeds:
        summaries[seed] = {}
        for method in METHODS:
            path = root / method / f"seed_{seed}" / "summary.json"
            summary = load_json(path)
            config = summary["config"]
            expected = {
                "seed": seed, "budget": budget, "epochs": epochs,
                "objectives": OBJECTIVES, "training_version": HARDWARE_CACHE_VERSION,
                "mac_convention": MAC_CONVENTION,
            }
            if method == "ga_search":
                expected["population_size"] = population_size
            if (summary["experiment"] != method or summary["evaluations"] != budget
                    or len(summary["evaluated"]) != budget
                    or any(config.get(key) != value for key, value in expected.items())):
                raise ValueError(f"Search summary does not match the requested experiment: {path}")
            current = {key: config[key] for key in PROTOCOL_KEYS}
            if protocol is not None and current != protocol:
                raise ValueError(f"Search protocol differs across runs: {path}")
            protocol = current
            summaries[seed][method] = summary
    return summaries


def describe(values):
    return {"mean": mean(values), "sample_std": stdev(values) if len(values) > 1 else None}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_search(path, by_seed, baseline):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    projections = (
        ("parameter_count", "val_macro_auc_ovr", 1e6, 1, "Parameters (M)", "Validation macro AUC"),
        ("macs", "val_macro_auc_ovr", 1e9, 1, "MACs (G)", "Validation macro AUC"),
        ("parameter_count", "macs", 1e6, 1e9, "Parameters (M)", "MACs (G)"),
    )
    for axis, (x, y, x_scale, y_scale, xlabel, ylabel) in zip(axes, projections):
        for method, label in zip(METHODS, ("GA", "Random")):
            front = [point for seed in by_seed.values() for point in seed[method]["pareto_front"]]
            axis.scatter([p[x] / x_scale for p in front], [p[y] / y_scale for p in front],
                         alpha=0.65, label=label)
        if x in baseline and y in baseline:
            axis.scatter(baseline[x] / x_scale, baseline[y] / y_scale, marker="*", color="black",
                         s=100, label="ResNet-18")
        axis.set(xlabel=xlabel, ylabel=ylabel)
        axis.grid(alpha=0.2)
        axis.legend()
    fig.suptitle("Projections of each seed's 3D validation Pareto front")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_convergence(path, by_seed):
    fig, axis = plt.subplots(figsize=(8, 5))
    for method, label in zip(METHODS, ("GA", "Random")):
        curves = np.array([seed[method]["convergence"] for seed in by_seed.values()])
        average = curves.mean(axis=0)
        steps = range(1, len(average) + 1)
        line, = axis.plot(steps, average, label=label)
        if len(curves) > 1:
            std = curves.std(axis=0, ddof=1)
            axis.fill_between(steps, average - std, average + std, color=line.get_color(), alpha=0.18)
    axis.set(xlabel="Evaluated architectures", ylabel="3D hypervolume", title="Validation search convergence")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def compare_legacy(path, searches, by_seed, bounds):
    legacy = load_json(path)
    seed = legacy["config"]["seed"]
    if seed not in searches:
        raise ValueError(f"Legacy comparator seed {seed} is absent from the experiment")
    current = searches[seed]["ga_search"]["config"]
    ga_keys = ("population_size", "tournament_size", "crossover_rate", "mutation_probability")
    if (legacy["experiment"] != "ga_search" or legacy["config"].get("objectives")
            or legacy["evaluations"] != current["budget"] or len(legacy["evaluated"]) != current["budget"]
            or any(legacy["config"].get(key) != current[key] for key in (*PROTOCOL_KEYS, *ga_keys))):
        raise ValueError(f"Legacy GA summary does not match the seed's search protocol: {path}")
    points = []
    for point in legacy["evaluated"]:
        parameters, macs = architecture_cost(point["architecture"])
        if parameters != point["parameter_count"]:
            raise ValueError(f"Parameter count differs from current model builder: {path}")
        points.append({**point, "macs": macs})
    reference = summarize_search(points, bounds)
    current_run = by_seed[seed]["ga_search"]
    return {
        "source": str(path), "seed": seed, "legacy_reprojected": reference,
        "ga_3d_minus_legacy": {key: current_run["metrics"][key] - value
                               for key, value in reference["metrics"].items()},
        "coverage_ga_3d_of_legacy": coverage(current_run["pareto_front"], reference["pareto_front"]),
        "coverage_legacy_of_ga_3d": coverage(reference["pareto_front"], current_run["pareto_front"]),
    }


def write_search_report(output, searches, legacy_path=None):
    bounds = cost_bounds()
    parameters, macs = architecture_cost(BASELINE_ARCHITECTURE)
    baseline = {"parameter_count": parameters, "macs": macs}
    by_seed = {}
    rows = []
    for seed, summaries in searches.items():
        by_seed[seed] = {method: summarize_search(summaries[method]["evaluated"], bounds) for method in METHODS}
        for method, other in (METHODS, METHODS[::-1]):
            run = by_seed[seed][method]
            metrics = run["metrics"]
            metrics.update({
                "coverage_of_other": coverage(run["pareto_front"], by_seed[seed][other]["pareto_front"]),
                "param_reduction_vs_resnet18": 1 - metrics["params_at_best_val_auc"] / parameters,
                "mac_reduction_vs_resnet18": 1 - metrics["macs_at_best_val_auc"] / macs,
            })
            rows.append({"seed": seed, "method": method, **metrics})
    keys = [key for key in rows[0] if key not in ("seed", "method")]
    report = {
        "objectives": OBJECTIVES,
        "mac_convention": MAC_CONVENTION,
        "normalization": {"cost_min": bounds[0], "cost_max": bounds[1], "reference": [0, 0, 0]},
        "protocol": {key: next(iter(searches.values()))["ga_search"]["config"][key] for key in PROTOCOL_KEYS},
        "baseline_cost": baseline,
        "by_seed": by_seed,
        "across_seeds": {method: {key: describe([row[key] for row in rows if row["method"] == method])
                                  for key in keys} for method in METHODS},
        "paired_ga_minus_random": {
            key: describe([seed["ga_search"]["metrics"][key] - seed["random_search"]["metrics"][key]
                           for seed in by_seed.values()]) for key in keys
        },
    }
    if legacy_path is not None:
        report["legacy_ga_comparison"] = compare_legacy(legacy_path, searches, by_seed, bounds)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "analysis.json", report)
    write_csv(output / "search_metrics.csv", rows)
    plot_search(output / "pareto_projections.png", by_seed, baseline)
    plot_convergence(output / "hypervolume_convergence.png", by_seed)


def write_test_report(root, output, seeds):
    rows = []
    for seed in seeds:
        summary = load_json(root / "test_eval" / f"seed_{seed}" / "summary.json")
        for result in summary["results"]:
            rows.append({
                "seed": seed, "model": result["model"], "parameter_count": result["parameter_count"],
                "macs": result["macs"], "gflops_conv_linear": 2 * result["macs"] / 1e9,
                "best_epoch": result["best_epoch"], **result["test_metrics"],
            })
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "test_metrics.csv", rows)
    save_json(output / "test_analysis.json", {
        "seeds": seeds,
        "mac_convention": MAC_CONVENTION,
        "across_seeds": {
            model: {key: describe([row[key] for row in rows if row["model"] == model])
                    for key in rows[0] if key not in ("seed", "model")}
            for model in ("ga", "random", "baseline")
        },
    })
