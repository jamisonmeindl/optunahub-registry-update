"""Simple landscape plotting utilities for the synthetic benchmark."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import optuna
import seaborn as sns
import optunahub
from optunahub import hub as _hub

def _load_synthetic_module():
    registry_root = os.getenv("OPTUNAHUB_REGISTRY_PATH")
    if registry_root:
        return _hub.load_local_module("benchmarks/synthetic", registry_root=registry_root)

    possible_root = Path(__file__).resolve().parents[2]
    if (possible_root / "benchmarks" / "synthetic").exists():
        return _hub.load_local_module("benchmarks/synthetic", registry_root=str(possible_root))

    if str(possible_root) not in sys.path:
        sys.path.insert(0, str(possible_root))
    return optunahub.load_module("benchmarks/synthetic")


sns.set_theme(style="whitegrid", context="talk")

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
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


def _save_plot(fig: plt.Figure, name: str) -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / f"{name}.pdf"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved landscape PDF:", path)
    return path


def _value_grid(dist: optuna.distributions.BaseDistribution, points=200):
    if isinstance(dist, optuna.distributions.FloatDistribution):
        is_log = bool(getattr(dist, "log", False)) and dist.low > 0 and dist.high > 0
        if is_log:
            values = np.geomspace(dist.low, dist.high, points)
            return values, True, True, "Float (log)"
        return np.linspace(dist.low, dist.high, points), True, False, "Float"
    if isinstance(dist, optuna.distributions.IntDistribution):
        low, high = int(dist.low), int(dist.high)
        is_log = bool(getattr(dist, "log", False)) and low > 0
        if is_log:
            values = np.geomspace(float(low), float(high), points)
            return np.unique(values), True, True, "Int (log)"
        step_count = min(points, high - low + 1)
        rng = np.linspace(low, high, step_count)
        rounded = np.unique(np.round(rng).astype(int))
        return rounded.astype(float), True, False, "Int"
    if isinstance(dist, optuna.distributions.CategoricalDistribution):
        choices = list(dist.choices)
        is_boolean = all(isinstance(value, bool) for value in choices) and len({False, True}.intersection(set(choices))) > 0
        label = "Boolean" if is_boolean else "Categorical"
        return np.array(choices, dtype=object), False, False, label
    raise ValueError(f"Unsupported distribution {dist}")


def _cast_value(dist, value):
    if isinstance(dist, optuna.distributions.FloatDistribution):
        return float(value)
    if isinstance(dist, optuna.distributions.IntDistribution):
        return int(round(float(value)))
    return value


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
    options = {
        "does_augmentations": "aug" in selected,
        "adds_noise": "noise" in selected,
        "has_failure_regions": "fail" in selected,
        "seed_tag": benchmark,
    }
    return options


def _plot_2d_legends(ax, x_vals, x_is_continuous, x_name, y_vals, y_is_continuous, y_name):
    def _select_ticks(values, is_cont):
        n = len(values)
        if not is_cont and n < 25:
            idxs = np.arange(n)
        else:
            count = min(6, n)
            idxs = np.linspace(0, n - 1, count, dtype=int)
        pos = idxs + 0.5
        if is_cont:
            labels = [f"{values[int(i)]:.2g}" for i in idxs]
        else:
            labels = [str(values[int(i)]) for i in idxs]
        return pos, labels

    x_pos, x_labels = _select_ticks(x_vals, x_is_continuous)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        x_labels,
        rotation=45 if not x_is_continuous or len(str(x_labels[0])) > 5 else 0,
        ha="right",
        rotation_mode="anchor",
    )
    y_pos, y_labels = _select_ticks(y_vals[::-1], y_is_continuous)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, rotation=0)
    ax.set_xlabel(f"{x_name}")
    ax.set_ylabel(f"{y_name}")


def plot_landscape(
    *,
    seed: int = 0,
    dim: int = 2,
    data_type: str = "Mixed",
    benchmark: str = "synthetic",
) -> None:
    synth_module = _load_synthetic_module()
    options = _variant_to_options(benchmark)
    problem = synth_module.Problem(dim=dim, seed=seed, data_type=data_type, options=options)
    space = problem.search_space
    params = list(space.items())
    if len(params) < 2:
        print("Need at least two parameters to plot.")
        return

    (x_name, x_dist), (y_name, y_dist) = params[0], params[1]
    x_vals, x_is_cont, _, x_label = _value_grid(x_dist)
    y_vals, y_is_cont, _, y_label = _value_grid(y_dist)

    base_params = {}
    for name, dist in space.items():
        if isinstance(dist, optuna.distributions.FloatDistribution):
            base_params[name] = 0.5 * (dist.low + dist.high)
        elif isinstance(dist, optuna.distributions.IntDistribution):
            base_params[name] = dist.low
        else:
            base_params[name] = dist.choices[0]

    total = len(x_vals) * len(y_vals)
    samples = np.empty(total, dtype=object)
    idx = 0
    for y in y_vals:
        for x in x_vals:
            samples[idx] = {
                **base_params,
                x_name: _cast_value(x_dist, x),
                y_name: _cast_value(y_dist, y),
            }
            idx += 1

    if hasattr(problem, "_env") and hasattr(problem._env, "evaluate_with_failures"):
        values, failure_mask = problem._env.evaluate_with_failures(samples)
    else:
        values = problem._env.evaluate(samples)
        failure_mask = np.zeros((samples.shape[0],), dtype=bool)

    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    grid = arr[:, 0].reshape(len(y_vals), len(x_vals))
    fail_grid = np.asarray(failure_mask, dtype=bool).reshape(len(y_vals), len(x_vals))

    fig, ax = plt.subplots(figsize=(10, 8))
    heatmap = sns.heatmap(
        grid[::-1, :],
        ax=ax,
        cmap="viridis",
        cbar_kws={"label": "Objective value"},
        xticklabels=False,
        yticklabels=False,
        rasterized=True,
    )
    for mesh in heatmap.collections:
        mesh.set_edgecolor("face")
    ax.set_facecolor("white")
    if fail_grid.any():
        overlay = fail_grid[::-1, :].astype(float)
        ax.contourf(
            overlay,
            levels=[0.5, 1.5],
            colors=["red"],
            alpha=0.2,
        )
    _plot_2d_legends(ax, x_vals, x_is_cont, x_name, y_vals, y_is_cont, y_name)
    failure_tag = " fail" if getattr(problem, "failure_active", False) else ""
    direction = "Minimize"
    if hasattr(problem, "directions") and problem.directions:
        direction = problem.directions[0].name.title()
    ax.set_title(f"{direction}: {x_name} vs {y_name} (seed={seed}){failure_tag}")
    base_name = problem.name()
    _save_plot(fig, f"{base_name}_landscape_{dim}D_{seed}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot synthetic GP landscapes.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7], help="Seeds to draw.")
    parser.add_argument("--dims", type=int, nargs="+", default=[2], help="Dimension tuples.")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=SYNTHETIC_BENCHMARK_VARIANTS,
        default=SYNTHETIC_BENCHMARK_VARIANTS,
        help="Synthetic variants to plot.",
    )
    parser.add_argument(
        "--data-type",
        choices=["Mixed", "Continuous"],
        default="Mixed",
        help="Parameter type distribution.",
    )
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    for seed in args.seeds:
        for dim in args.dims:
            for benchmark in args.benchmarks:
                plot_landscape(
                    seed=seed,
                    dim=dim,
                    data_type=args.data_type,
                    benchmark=benchmark,
                )
    print("Synthetic landscapes plotted.")


if __name__ == "__main__":
    _main()
