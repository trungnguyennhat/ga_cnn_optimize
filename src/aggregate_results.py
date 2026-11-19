"""Summarize independent search seeds and compare dataset-specific outcomes."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.search_runtime import save_json


def stats(values):
    return {"n": len(values), "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) >= 2 else None}


def paired_test(deltas):
    if len(deltas) < 3:
        return None
    rounded = np.round(deltas, 12)
    return 1.0 if not np.any(rounded) else float(wilcoxon(rounded, alternative="two-sided").pvalue)


def summarize(analyses):
    analyses = sorted(analyses, key=lambda record: record["seed"])
    example = analyses[0]
    seeds = [record["seed"] for record in analyses]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Duplicate seed for {example['dataset']}")
    for field in ("objectives", "protocol", "space", "search_settings"):
        if any(record[field] != example[field] for record in analyses):
            raise ValueError(f"Inputs must share {field} within {example['dataset']}")

    search_metrics = ("best_auc", "objective_log_hypervolume", "objective_linear_hypervolume", "runtime_seconds")
    test_metrics = ("test_macro_auc_ovr", "test_macro_f1", "test_macro_recall", "test_accuracy",
                    "parameter_count", "macs_per_image", "gflops_per_image")
    search = {
        method: {key: stats([record["search"][method][key] for record in analyses]) for key in search_metrics}
        for method in ("ga", "random")
    }
    coverage = {
        f"{a}_over_{b}": stats([record["coverage_objectives"][a][b] for record in analyses])
        for a, b in (("ga", "random"), ("random", "ga"))
    }
    test = {}
    for record in analyses:
        for model in record["test_results"]:
            test.setdefault(model["model"], {})[record["seed"]] = model
    test_summary = {
        name: {key: stats([row[key] for row in models.values()]) for key in test_metrics}
        for name, models in test.items()
    }
    paired = {}
    for key in search_metrics:
        delta = [row["search"]["ga"][key] - row["search"]["random"][key] for row in analyses]
        paired[f"search_{key}"] = {"seeds": seeds, **stats(delta), "wilcoxon_p": paired_test(delta)}
    for suffix in ("", "_knee", "_smallest_baseline"):
        ga, random = test.get(f"ga{suffix}", {}), test.get(f"random{suffix}", {})
        shared = sorted(ga.keys() & random.keys())
        if shared:
            for key in ("test_macro_auc_ovr", "test_macro_f1"):
                delta = [ga[seed][key] - random[seed][key] for seed in shared]
                paired[f"{key}{suffix}"] = {"seeds": shared, **stats(delta), "wilcoxon_p": paired_test(delta)}
    return {
        "seeds": seeds, "protocol": example["protocol"], "space": example["space"],
        "objectives": example["objectives"], "search": search, "coverage": coverage,
        "test": test_summary, "paired_ga_minus_random": paired,
    }


def plot_bands(records, path):
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for method in ("ga", "random"):
        curves = [record["search"][method]["convergence"] for record in records]
        x = [row["evaluation"] for row in curves[0]]
        if any([row["evaluation"] for row in curve] != x for curve in curves):
            raise ValueError("Convergence bands require the same evaluation grid")
        for axis, key in zip(axes, ("best_auc", "objective_log_hypervolume")):
            values = np.asarray([[row[key] for row in curve] for curve in curves])
            means = values.mean(axis=0)
            line, = axis.plot(x, means, label=method)
            if len(curves) > 1:
                spread = values.std(axis=0, ddof=1)
                axis.fill_between(x, means - spread, means + spread, alpha=0.18, color=line.get_color())
            axis.set_ylabel(key)
            axis.grid(alpha=0.3)
            axis.legend()
    axes[-1].set_xlabel("Evaluated architectures")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    groups = {}
    for path in args.analysis:
        result = json.loads(path.read_text(encoding="utf-8"))
        groups.setdefault(result["dataset"], []).append(result)
    summaries = {dataset: summarize(rows) for dataset, rows in groups.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "multi_seed.json", {"datasets": summaries})
    with (args.output_dir / "dataset_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("dataset", "scope", "method", "metric", "n", "mean", "std"))
        writer.writeheader()
        for dataset, summary in summaries.items():
            for scope in ("search", "test"):
                for method, metrics in summary[scope].items():
                    for key, values in metrics.items():
                        writer.writerow({"dataset": dataset, "scope": scope, "method": method, "metric": key, **values})
            for key, values in summary["coverage"].items():
                writer.writerow({"dataset": dataset, "scope": "search", "method": key, "metric": "coverage", **values})
    for dataset, rows in groups.items():
        plot_bands(rows, args.output_dir / f"convergence_{dataset}.png")
    print(args.output_dir / "multi_seed.json")


if __name__ == "__main__":
    main()
