"""Run paired GA and Random Search experiments with AUC, parameters and MACs."""

import argparse
import subprocess
import sys
from pathlib import Path

from src.hardware_analysis import METHODS, load_searches, write_search_report, write_test_report
from src.search_runtime import HARDWARE_RESULTS, SEARCH_EPOCHS
from src.train import DEFAULT_EPOCHS, PROJECT_ROOT


def run_module(module, *arguments):
    subprocess.run([sys.executable, "-m", module, *map(str, arguments)], check=True)


def run_searches(args):
    for seed in args.seeds:
        common = ("--hardware-aware", "--seed", seed, "--budget", args.budget,
                  "--epochs", args.epochs, "--device", args.device)
        for method in METHODS:
            extra = ("--population-size", args.population_size) if method == "ga_search" else ()
            run_module(f"src.{method}.run_search", *common, *extra, "--output-dir", args.output_dir / method)


def run_test_evaluations(args):
    for seed in args.seeds:
        seed_dir = f"seed_{seed}"
        baseline_root = args.output_dir / "baseline"
        baseline_dir = baseline_root / seed_dir
        if not all((baseline_dir / name).is_file() for name in ("checkpoint.pt", "result.json")):
            run_module("src.train", "--seed", seed, "--epochs", DEFAULT_EPOCHS,
                       "--device", args.device, "--output-dir", baseline_root)
        run_module(
            "src.test_eval", "--hardware-aware", "--seed", seed, "--device", args.device,
            "--ga-summary", args.output_dir / "ga_search" / seed_dir / "summary.json",
            "--random-summary", args.output_dir / "random_search" / seed_dir / "summary.json",
            "--baseline-dir", baseline_dir, "--output-dir", args.output_dir / "test_eval" / seed_dir,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=SEARCH_EPOCHS)
    parser.add_argument("--population-size", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / HARDWARE_RESULTS)
    parser.add_argument("--phase", choices=("search", "analysis", "test"), default="search",
                        help="Run searches, analyze existing validation logs, or perform final test evaluation")
    parser.add_argument("--legacy-ga-summary", type=Path,
                        help="Compare against a completed two-objective GA run in search or analysis phase")
    args = parser.parse_args()

    if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= seed < 2**32 for seed in args.seeds):
        parser.error("seeds must be unique integers in [0, 2**32)")
    if min(args.budget, args.epochs, args.population_size) <= 0:
        parser.error("budget, epochs and population-size must be positive")
    if args.phase == "test":
        if args.budget != 60 or args.epochs != SEARCH_EPOCHS:
            parser.error("final evaluation requires 60 search evaluations at 20 epochs per candidate")
        if args.legacy_ga_summary is not None:
            parser.error("--legacy-ga-summary is for the search or analysis phase")

    args.output_dir = args.output_dir.resolve()
    if args.phase == "search":
        run_searches(args)
    searches = load_searches(args.output_dir, args.seeds, budget=args.budget,
                             epochs=args.epochs, population_size=args.population_size)
    output = args.output_dir / "analysis"
    if args.phase == "test":
        run_test_evaluations(args)
        write_test_report(args.output_dir, output, args.seeds)
    else:
        write_search_report(output, searches, args.legacy_ga_summary)
    print(f"reports={output}", flush=True)


if __name__ == "__main__":
    main()
