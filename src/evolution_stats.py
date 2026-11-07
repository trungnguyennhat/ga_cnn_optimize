"""Advanced evolutionary metrics and statistical analysis for GA-CNN/ResNet NAS."""

import json
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.nsga2 import dominates
from src.search_space import DROPOUT_CHOICES, KERNEL_CHOICES, STAGE_BLOCK_CHOICES, STAGE_CHANNEL_CHOICES


def compute_schotts_spacing(pareto_points: list[dict]) -> float:
    """Compute Schott's spacing metric S measuring uniformity of Pareto front solutions.
    
    A smaller value indicates more uniform spacing between consecutive solutions.
    """
    if len(pareto_points) < 2:
        return 0.0

    # Sort solutions by parameter_count
    sorted_points = sorted(pareto_points, key=lambda p: p["parameter_count"])
    aucs = np.array([p["val_macro_auc_ovr"] for p in sorted_points])
    params = np.array([p["parameter_count"] for p in sorted_points])

    # Normalize objectives to [0, 1] for fair distance calculation
    auc_norm = (aucs - aucs.min()) / (aucs.max() - aucs.min() + 1e-12)
    params_norm = (params - params.min()) / (params.max() - params.min() + 1e-12)
    norm_points = np.column_stack([auc_norm, params_norm])

    # Find minimum Euclidean distance to any other point
    distances = []
    n = len(norm_points)
    for i in range(n):
        diffs = norm_points - norm_points[i]
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        dists[i] = np.inf  # Ignore self
        distances.append(np.min(dists))

    distances = np.array(distances)
    mean_dist = np.mean(distances)
    spacing = np.sqrt(np.sum((distances - mean_dist) ** 2) / (n - 1))
    return float(spacing)


def statistical_comparison(ga_points: list[dict], random_points: list[dict]) -> dict:
    """Perform Mann-Whitney U test (Wilcoxon rank-sum test) between GA and Random Search."""
    ga_aucs = [p["val_macro_auc_ovr"] for p in ga_points]
    rand_aucs = [p["val_macro_auc_ovr"] for p in random_points]

    u_stat, p_val = stats.mannwhitneyu(ga_aucs, rand_aucs, alternative="greater")

    # Effect size (rank-biserial correlation)
    n1, n2 = len(ga_aucs), len(rand_aucs)
    rank_biserial = 1.0 - (2.0 * u_stat) / (n1 * n2)

    return {
        "mann_whitney_u": float(u_stat),
        "p_value": float(p_val),
        "statistically_significant_005": bool(p_val < 0.05),
        "rank_biserial_effect_size": float(rank_biserial),
        "ga_auc_mean": float(np.mean(ga_aucs)),
        "ga_auc_std": float(np.std(ga_aucs)),
        "random_auc_mean": float(np.mean(rand_aucs)),
        "random_auc_std": float(np.std(rand_aucs)),
        "ga_auc_median": float(np.median(ga_aucs)),
        "random_auc_median": float(np.median(rand_aucs)),
    }


def analyze_gene_frequencies(evaluations: list[dict], pareto_front: list[dict]) -> dict:
    """Analyze the survival rate and frequency of architectural choices (genes)."""
    def extract_stats(arch_list):
        stage_blocks = [arch["architecture"]["stage_blocks"] for arch in arch_list]
        stage_channels = [arch["architecture"]["stage_channels"] for arch in arch_list]
        kernel_sizes = [arch["architecture"]["kernel_sizes"] for arch in arch_list]
        dropouts = [arch["architecture"]["dropout"] for arch in arch_list]

        return {
            "kernel_3_count": sum(k == 3 for ks in kernel_sizes for k in ks),
            "kernel_5_count": sum(k == 5 for ks in kernel_sizes for k in ks),
            "avg_blocks_per_stage": [
                float(np.mean([sb[stage] for sb in stage_blocks])) for stage in range(4)
            ],
            "avg_channels_per_stage": [
                float(np.mean([sc[stage] for sc in stage_channels])) for stage in range(4)
            ],
            "dropout_distribution": dict(Counter(dropouts)),
        }

    return {
        "all_evaluations": extract_stats(evaluations),
        "pareto_front": extract_stats(pareto_front),
    }


def plot_gene_distribution(path: Path, ga_analysis: dict, random_analysis: dict):
    """Plot gene distribution comparison between GA Pareto front and Random Pareto front."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Kernel size preference
    ax = axes[0, 0]
    methods = ["GA Pareto", "Random Pareto"]
    k3_counts = [
        ga_analysis["pareto_front"]["kernel_3_count"],
        random_analysis["pareto_front"]["kernel_3_count"],
    ]
    k5_counts = [
        ga_analysis["pareto_front"]["kernel_5_count"],
        random_analysis["pareto_front"]["kernel_5_count"],
    ]
    x = np.arange(len(methods))
    width = 0.35
    ax.bar(x - width/2, k3_counts, width, label="Kernel 3x3", color="royalblue")
    ax.bar(x + width/2, k5_counts, width, label="Kernel 5x5", color="darkorange")
    ax.set_ylabel("Occurrences")
    ax.set_title("Kernel Size Preference on Pareto Front")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 2. Average Blocks per Stage
    ax = axes[0, 1]
    stages = ["Stage 1", "Stage 2", "Stage 3", "Stage 4"]
    x = np.arange(len(stages))
    ax.plot(x, ga_analysis["pareto_front"]["avg_blocks_per_stage"], "o-", color="royalblue", linewidth=2.5, label="GA Pareto")
    ax.plot(x, random_analysis["pareto_front"]["avg_blocks_per_stage"], "s--", color="darkorange", linewidth=2.5, label="Random Pareto")
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel("Average Block Count")
    ax.set_title("Blocks per Stage Distribution")
    ax.grid(alpha=0.3)
    ax.legend()

    # 3. Average Channels per Stage
    ax = axes[1, 0]
    ax.plot(x, ga_analysis["pareto_front"]["avg_channels_per_stage"], "o-", color="royalblue", linewidth=2.5, label="GA Pareto")
    ax.plot(x, random_analysis["pareto_front"]["avg_channels_per_stage"], "s--", color="darkorange", linewidth=2.5, label="Random Pareto")
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel("Average Channels")
    ax.set_title("Channel Scaling across Stages")
    ax.grid(alpha=0.3)
    ax.legend()

    # 4. Dropout rate frequencies
    ax = axes[1, 1]
    ga_drop = ga_analysis["pareto_front"]["dropout_distribution"]
    rand_drop = random_analysis["pareto_front"]["dropout_distribution"]
    drop_keys = sorted(list(set(list(ga_drop.keys()) + list(rand_drop.keys()))))
    ga_vals = [ga_drop.get(k, 0) for k in drop_keys]
    rand_vals = [rand_drop.get(k, 0) for k in drop_keys]
    x_drop = np.arange(len(drop_keys))
    ax.bar(x_drop - width/2, ga_vals, width, label="GA Pareto", color="royalblue")
    ax.bar(x_drop + width/2, rand_vals, width, label="Random Pareto", color="darkorange")
    ax.set_xticks(x_drop)
    ax.set_xticklabels([str(k) for k in drop_keys])
    ax.set_xlabel("Dropout Rate")
    ax.set_ylabel("Count")
    ax.set_title("Dropout Rate Distribution on Pareto Front")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_generational_progress(path: Path, ga_progress_path: Path):
    """Plot generational Pareto front evolution and genotypic diversity."""
    if not ga_progress_path.is_file():
        return

    progress_data = json.loads(ga_progress_path.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(8, 6))

    evaluations = progress_data.get("records", [])
    if not evaluations:
        return

    # Color code by evaluation phase (early: 1-20, mid: 21-40, late: 41-60)
    evals = np.array([rec["evaluation"] for rec in evaluations])
    aucs = np.array([rec["metrics"]["val_macro_auc_ovr"] for rec in evaluations])
    params = np.array([rec["parameter_count"] / 1e6 for rec in evaluations])

    scatter = ax.scatter(params, aucs, c=evals, cmap="plasma", alpha=0.8, s=40)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Evaluation Index (Evolution Time)")

    ax.set(
        title="Generational Search Progression: Parameter Count vs Val AUC",
        xlabel="Parameters (Millions)",
        ylabel="Validation Macro ROC AUC",
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
