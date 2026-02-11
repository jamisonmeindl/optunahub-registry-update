"""Example runner for the synthetic GP benchmark suite."""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import optuna
import optunahub
from optunahub import hub as _hub
import sys


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if not __package__:
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    from benchmarks.synthetic.plotting_example import plot_landscape
else:
    from .plotting_example import plot_landscape

SYNTHETIC_BENCHMARK_VARIANTS = [
    "synthetic",
    "synthetic_aug",
    "synthetic_noise",
    "synthetic_fail",
    "synthetic_aug_noise",
    "synthetic_aug_fail",
    "synthetic_noise_fail",
    "synthetic_aug_noise_fail",
]


def _load_synthetic_module():
    registry_root = os.getenv("OPTUNAHUB_REGISTRY_PATH")
    if registry_root:
        return _hub.load_local_module("benchmarks/synthetic", registry_root=registry_root)

    possible_root = Path(__file__).resolve().parents[2]
    if (possible_root / "benchmarks" / "synthetic").exists():
        return _hub.load_local_module("benchmarks/synthetic", registry_root=str(possible_root))

    return optunahub.load_module("benchmarks/synthetic")


def _trial_value(trial: optuna.trial.FrozenTrial) -> float | None:
    if trial.values:
        return float(trial.values[0])
    if trial.value is not None:
        return float(trial.value)
    return None


def _variant_to_options(benchmark: str) -> dict[str, Any]:
    if benchmark == "synthetic":
        return {
            "does_augmentations": False,
            "adds_noise": False,
            "has_failure_regions": False,
            "seed_tag": benchmark,
        }
    parts = benchmark.split("_")[1:]
    selected = set(parts)
    options: dict[str, Any] = {
        "does_augmentations": "aug" in selected,
        "adds_noise": "noise" in selected,
        "has_failure_regions": "fail" in selected,
        "seed_tag": benchmark,
    }
    return options


def _run_demo(
    *,
    benchmark: str,
    seed: int = 0,
    dim: int = 2,
    data_type: str = "Mixed",
    num_objectives: int = 1,
    n_trials: int = 25,
) -> None:
    """Run one synthetic benchmark configuration."""
    synth_module = _load_synthetic_module()
    options = {
        "num_objectives": num_objectives,
        **_variant_to_options(benchmark),
    }
    problem = synth_module.Problem(
        dim=dim,
        seed=seed,
        data_type=data_type,
        options=options,
    )
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=seed),
        directions=problem.directions,
    )
    study.optimize(problem, n_trials=n_trials)

    best = min(
        (value for trial in study.trials if (value := _trial_value(trial)) is not None),
        default=None,
    )
    if best is None:
        print(
            f"[benchmark={benchmark} seed={seed} dim={dim} obj={num_objectives}] "
            "No trial produced a scalar objective."
        )
        return

    print(
        f"[benchmark={benchmark} seed={seed} dim={dim} obj={num_objectives}] "
        f"Best synthetic value ({problem.directions[0].name.title()}): {best:.4f}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic GP benchmark demos.")
    parser.add_argument("--seeds", type=int, nargs="+", default=range(3), help="Seeds to evaluate.")
    parser.add_argument("--dims", type=int, nargs="+", default=[2,5], help="Problem dimensions.")
    parser.add_argument(
        "--data-type",
        choices=["Mixed", "Continuous"],
        default="Mixed",
        help="Parameter type distribution.",
    )
    parser.add_argument(
        "--num-objectives",
        type=int,
        nargs="+",
        default=[1,2],
        help="Objective counts to test.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=SYNTHETIC_BENCHMARK_VARIANTS,
        default=SYNTHETIC_BENCHMARK_VARIANTS,
        help="Synthetic variants to run.",
    )
    parser.add_argument("--n-trials", type=int, default=25, help="Trials per run.")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate landscapes for each selected benchmark variant.",
    )
    return parser.parse_args()


def _run_all() -> None:
    args = _parse_args()
    for seed in args.seeds:
        for dim in args.dims:
            for benchmark in args.benchmarks:
                for obj in args.num_objectives:
                    _run_demo(
                        benchmark=benchmark,
                        seed=seed,
                        dim=dim,
                        data_type=args.data_type,
                        num_objectives=obj,
                        n_trials=args.n_trials,
                    )
                if args.plot:
                    plot_landscape(
                        seed=seed,
                        dim=dim,
                        data_type=args.data_type,
                        benchmark=benchmark,
                    )


if __name__ == "__main__":
    _run_all()
