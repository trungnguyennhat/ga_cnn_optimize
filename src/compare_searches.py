"""Compare completed search methods with fixed normalization and compute accounting."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.datasets import DATASETS
from src.objectives import AUC, MACS, OBJECTIVES, PARAMS, architecture_costs, cost_bounds, coverage, front_of, hypervolume, normalize
from src.search_runtime import save_json
from src.search_space import canonical_architecture


def attach_costs(rows, num_classes):
    cache = {}
    for row in rows:
        if MACS not in row or PARAMS not in row:
            key = canonical_architecture(row["architecture"])
            if key not in cache:
                cache[key] = architecture_costs(row["architecture"], num_classes)
            row.update(cache[key])


def _gene_frequencies(front):
    counters = {}
    for row in front:
        arch = row["architecture"]
        genes = {
            "stage_blocks": arch["stage_blocks"], "stage_channels": arch["stage_channels"],
            "kernel_sizes": arch["kernel_sizes"], "dropout": [arch["dropout"]],
            "block_types": arch.get("block_types", ["basic"] * 4),
            "use_se": arch.get("use_se", [False] * 4),
            "activation": [arch.get("activation", "relu")],
        }
        for gene, values in genes.items():
            by_stage = counters.setdefault(gene, [Counter() for _ in values])
            for counts, value in zip(by_stage, values):
                counts[str(value)] += 1
    return {gene: [dict(counts) for counts in stages] for gene, stages in counters.items()}


def compare_searches(searches, objectives, dataset):
    if not searches or any(not search.get("evaluated") for search in searches.values()):
        raise ValueError("Comparison requires nonempty search summaries")
    num_classes = DATASETS[dataset].num_classes
    configs = [search["config"] for search in searches.values()]
    shared_keys = (
        "seed", "epochs", "optimizer", "scheduler", "loss", "augmentation",
        "weight_decay", "amp", "preload_gpu", "learning_rate", "batch_size", "train_val_sha256",
    )
    defaults = {"augmentation": "none", "weight_decay": 0.0, "amp": False, "preload_gpu": False}
    if (any(config.get("dataset", "dermamnist") != dataset for config in configs)
        or any(len({json.dumps(config.get(key, defaults.get(key)), sort_keys=True)
                    for config in configs}) != 1 for key in shared_keys)):
        raise ValueError("Search summaries must use the same dataset, seed and proxy training configuration")
    spaces = {config.get("architecture_space", "v1").rsplit("_", 1)[-1] for config in configs}
    limits = [cost_bounds(space, num_classes) for space in sorted(spaces)]
    bounds = {key: (min(row[key][0] for row in limits), max(row[key][1] for row in limits)) for key in (PARAMS, MACS)}
    fronts, methods = {}, {}
    for name, search in searches.items():
        rows = search["evaluated"]
        if len(rows) != search["evaluations"]:
            raise ValueError(f"{name}: incomplete evaluation records")
        attach_costs(rows, num_classes)
        completed = []
        curve = []
        for index, row in enumerate(rows, start=1):
            if row.get("full_fidelity", True):
                completed.append(row)
            front = front_of(completed, objectives)
            fitnesses = [tuple(point[key] for key in objectives) for point in front]
            curve.append({
                "evaluation": index,
                "full_evaluations": len(completed),
                "elapsed_seconds": row.get("elapsed_seconds"),
                "training_seconds": row.get("training_seconds_so_far"),
                "best_auc": max((point[AUC] for point in front), default=None),
                "log_hypervolume": hypervolume([normalize(f, objectives, bounds) for f in fitnesses]),
                "linear_hypervolume": hypervolume([
                    normalize(f, objectives, bounds, log_scale=False) for f in fitnesses
                ]),
            })
        fronts[name] = front
        device = search["config"].get("device") or ""
        seconds = curve[-1]["training_seconds"]
        methods[name] = {
            **curve[-1], "evaluations": len(rows), "runtime_seconds": search["runtime_seconds"],
            "cache_hits": search.get("cache_hits"), "device": device,
            "training_gpu_hours": seconds / 3600 if device.startswith("cuda") and seconds is not None else None,
            "search_objectives": search["config"].get("objectives", list(OBJECTIVES["params"])),
            "total_epochs": sum(row.get("epochs_trained", search["config"]["epochs"]) for row in rows),
            "pareto_front": front, "pareto_gene_counts": _gene_frequencies(front),
            "convergence": curve,
        }
    return {
        "dataset": dataset, "seed": configs[0]["seed"], "objectives": list(objectives),
        "normalization": {"auc": [0, 1], "cost_bounds": bounds, "reference": [0] * len(objectives)},
        "methods": methods,
        "coverage": {
            left: {right: coverage(fronts[left], fronts[right], objectives) for right in methods if right != left}
            for left in methods
        },
    }


def write_comparison(report, directory):
    directory.mkdir(parents=True, exist_ok=True)
    save_json(directory / "search_comparison.json", report)
    fields = ("evaluation", "full_evaluations", "elapsed_seconds", "training_seconds",
              "best_auc", "log_hypervolume", "linear_hypervolume")
    with (directory / "search_convergence.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("method", *fields))
        writer.writeheader()
        for name, method in report["methods"].items():
            writer.writerows({"method": name, **row} for row in method["convergence"])

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for name, method in report["methods"].items():
        curve = method["convergence"]
        for row_index, key in enumerate(("best_auc", "log_hypervolume")):
            for col, x_key in enumerate(("evaluation", "elapsed_seconds")):
                valid = [point for point in curve if point[x_key] is not None and point[key] is not None]
                if valid:
                    axes[row_index, col].plot([p[x_key] for p in valid], [p[key] for p in valid], label=name)
                axes[row_index, col].set(xlabel=x_key, ylabel=key)
    for axis in axes.flat:
        axis.grid(alpha=0.3)
        if axis.lines:
            axis.legend()
    fig.tight_layout()
    fig.savefig(directory / "search_convergence.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, (x, y) in zip(axes, ((PARAMS, AUC), (MACS, AUC), (PARAMS, MACS))):
        for name, method in report["methods"].items():
            front = method["pareto_front"]
            axis.scatter([row[x] for row in front], [row[y] for row in front], label=name)
        axis.set(xlabel=x, ylabel=y, xscale="log", yscale="log" if y == MACS else "linear")
        axis.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(directory / "pareto_projections.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="NAME=SUMMARY")
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="dermamnist")
    parser.add_argument("--objectives", choices=tuple(OBJECTIVES), default="params-macs")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    searches = {}
    for item in args.run:
        name, separator, path = item.partition("=")
        if not separator or not name or not path or name in searches:
            parser.error("--run requires a unique NAME=SUMMARY")
        searches[name] = json.loads(Path(path).read_text(encoding="utf-8"))
    write_comparison(compare_searches(searches, OBJECTIVES[args.objectives], args.dataset), args.output_dir)
    print(args.output_dir / "search_comparison.json")


if __name__ == "__main__":
    main()
