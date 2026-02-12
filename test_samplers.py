"""Compare Optuna samplers across built-in benchmark presets."""
from __future__ import annotations

import os
import sys

# --- ROBUSTNESS FIX: PREVENT THREAD CONTENTION ---
# Must be set before importing numpy/torch/optuna to prevent workers
# from spawning thousands of threads (Process Parallelism vs Thread Parallelism).
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# -------------------------------------------------

from collections import Counter
import argparse
import concurrent.futures
import contextlib
import io
import json
import logging
import signal
from pathlib import Path
from statistics import mean
import time
from typing import Any, Callable, Sequence

import numpy as np

import optuna
import optunahub
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers import NSGAIISampler, NSGAIIISampler, RandomSampler, TPESampler
from optuna._hypervolume import compute_hypervolume
from optunahub import hub as _hub
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)

_HEBO_UPDATED_VARIANTS = {
    "hebo_updated",
    "hebo_updated_gp",
    "hebo_updated_rf",
    "hebo_updated_deep_ensemble",
    "hebo_updated_catboost",
    "hebo_updated_psgld",
    "hebo_updated_svidkl",
}
_HEBO_UPDATED_MODEL_NAME_BY_SAMPLER = {
    "hebo_updated": None,
    "hebo_updated_gp": "gp",
    "hebo_updated_rf": "rf",
    "hebo_updated_deep_ensemble": "deep_ensemble",
    "hebo_updated_catboost": "catboost",
    "hebo_updated_psgld": "psgld",
    "hebo_updated_svidkl": "svidkl",
}

_BBOB_DIMS = (2, 3, 5, 10, 20, 40)
_BBOB_MIXINT_DIMS = (5, 10, 20, 40, 80, 160)
_BBOB_LARGESCALE_DIMS = (20, 40, 80, 160, 320, 640)
_BBOB_INSTANCE_LIMIT = 110
_BBOB_COCO_INSTANCE_LIMIT = 15
_BBOB_FUNCTION_COUNT = 24
_BBOB_BIOBJ_FUNCTION_COUNT = 92
_BBOB_NOISY_FUNCTION_START = 101
_BBOB_NOISY_FUNCTION_COUNT = 30
_BBOB_CONSTRAINED_FUNCTION_COUNT = 54
_BINARY_CONSTRAINT_KEY = "_binary_constraint"
_DTLZ_CONSTRAINED_COMBINATIONS = (
    {"constraint_type": 1, "function_id": 1},  # C1-DTLZ1
    {"constraint_type": 1, "function_id": 3},  # C1-DTLZ3
    {"constraint_type": 2, "function_id": 2},  # C2-DTLZ2
    {"constraint_type": 3, "function_id": 1},  # C3-DTLZ1
    {"constraint_type": 3, "function_id": 4},  # C3-DTLZ4
)
_BENCHMARKS_USING_NUM_OBJECTIVES = {
    "synthetic",
    "synthetic_aug",
    "synthetic_noise",
    "synthetic_fail",
    "synthetic_aug_noise",
    "synthetic_aug_fail",
    "synthetic_noise_fail",
    "synthetic_aug_noise_fail",
    "dtlz",
    "dtlz_constrained",
    "dtlz_unknown_constraints",
    "wfg",
}
_BENCHMARKS_USING_DIM = {
    "synthetic",
    "synthetic_aug",
    "synthetic_noise",
    "synthetic_fail",
    "synthetic_aug_noise",
    "synthetic_aug_fail",
    "synthetic_noise_fail",
    "synthetic_aug_noise_fail",
    "dtlz",
    "dtlz_constrained",
    "dtlz_unknown_constraints",
    "wfg",
    "bbob",
    "bbob_biobj",
    "bbob_noisy",
    "bbob_mixint",
    "bbob_largescale",
    "bbob_constrained",
    "bbob_unknown_constraints",
    "bbob_biobj_mixint",
}
_SYNTHETIC_ALIAS_COMPONENTS = {"aug", "noise", "fail"}


class _ConstraintViolationError(Exception):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple Optuna samplers on built-in benchmark presets and score them."
    )
    parser.add_argument(
        "--benchmark",
        choices=[
            "synthetic",
            "synthetic_aug",
            "synthetic_noise",
            "synthetic_fail",
            "synthetic_aug_noise",
            "synthetic_aug_fail",
            "synthetic_noise_fail",
            "synthetic_aug_noise_fail",
            "zdt",
            "dtlz",
            "dtlz_constrained",
            "dtlz_unknown_constraints",
            "wfg",
            "bbob",
            "bbob_biobj",
            "bbob_noisy",
            "bbob_mixint",
            "bbob_largescale",
            "bbob_constrained",
            "bbob_unknown_constraints",
            "bbob_biobj_mixint",
            "hpolib",
        ],
        default="synthetic",
        help=(
            "Benchmark preset. Includes built-in problem generation so you do not need "
            "an external config file."
        ),
    )
    parser.add_argument(
        "--n-problems",
        type=int,
        default=25,
        help="Number of distinct problem instances to evaluate."
    )
    parser.add_argument(
        "--problem-offset",
        type=int,
        default=0,
        help="Offset into the preset sequence (for function IDs / seeds).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of Optuna trials per optimizer."
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=5,
        help="Problem dimensionality for presets that use it."
    )
    parser.add_argument(
        "--data-type",
        choices=["Mixed", "Continuous"],
        default="Mixed",
        help="Parameter space type for the synthetic benchmark."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base RNG seed shared across problems."
    )
    parser.add_argument(
        "--samplers",
        nargs="+",
        default=[
            "random",
            "tpe",
            # "nsgaii",
            # "nsgaiii",
            "autosampler",
            "hebo_updated_gp",
            "hebo_updated_rf",
            # "hebo_updated_deep_ensemble",
            "hebo_updated_catboost",
            "hebo_updated_psgld",
            "hebo_updated_svidkl",
            # "smac_mo",
            # "moead",
            # "speaii",
        ],
        help=(
            "Samplers to compare (choices: random, tpe, nsgaii, nsgaiii, "
            "autosampler, hebo, hebo_updated, hebo_updated_gp, hebo_updated_rf, "
            "hebo_updated_deep_ensemble, hebo_updated_catboost, hebo_updated_psgld, "
            "hebo_updated_svidkl, smac, smac_mo, moead, speaii)."
        ),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help=(
            "Path to store the comparison summary. If omitted, uses "
            "experiment_results/sampler_comparison_<benchmark>[_<dim>d][_<num_objectives>obj]_<n_problems>_<trials>problems.json."
        ),
    )
    parser.add_argument(
        "--num-objectives",
        type=int,
        default=1,
        help="Number of objectives for presets that support it (e.g. synthetic/dtlz/wfg).",
    )
    parser.add_argument(
        "--instance-id",
        type=int,
        default=1,
        help="Base instance ID for BBOB-style presets.",
    )
    parser.add_argument(
        "--hpolib-dataset-id",
        type=int,
        default=None,
        help="Fixed dataset_id for hpolib (omit to cycle across available datasets).",
    )
    parser.add_argument(
        "--optuna-log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="ERROR",
        help="Optuna logging level. Use WARNING to hide per-trial completion logs.",
    )
    parser.add_argument(
        "--print-step-history",
        action="store_true",
        help=(
            "Print per-trial step history (selected action, predicted mean/std, true evaluation)."
        ),
    )
    parser.add_argument(
        "--sampler-output",
        choices=["hide", "show"],
        default="hide",
        help="Whether to hide or show sampler backend stdout/stderr during study.optimize.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of worker processes (1 disables multiprocessing).",
    )
    return parser.parse_args()


def _resolve_registry_root() -> Path:
    env_root = os.getenv("OPTUNAHUB_REGISTRY_PATH")
    if env_root:
        return Path(env_root)
    base = Path(__file__).resolve().parent
    package_root = base / "package"
    if package_root.exists():
        return package_root
    return base


def _load_module(name: str, registry_root: Path) -> Any:
    try:
        return _hub.load_local_module(name, registry_root=str(registry_root))
    except Exception as exc:  # pragma: no cover
        logger.debug("local module %s not found: %s", name, exc)
        return optunahub.load_module(name)


def _instantiate_sampler(
    name: str,
    search_space: dict[str, BaseDistribution],
    seed: int,
    n_trials: int,
    hebo_module: Any,
    hebo_updated_module: Any,
    smac_module: Any,
    smac_mo_module: Any,
    autosampler_module: Any,
    moead_module: Any,
    speaii_module: Any,
    num_objectives: int,
    use_binary_constraint: bool = False,
) -> optuna.samplers.BaseSampler:
    def _binary_constraints(trial: optuna.trial.FrozenTrial) -> tuple[float]:
        raw = trial.user_attrs.get(_BINARY_CONSTRAINT_KEY)
        if raw is None:
            return (1.0,)
        try:
            value = float(raw)
        except Exception:
            return (1.0,)
        return (value,)

    effective_constraints_func = _binary_constraints if use_binary_constraint else None

    sampler: optuna.samplers.BaseSampler
    if name == "random":
        return RandomSampler(seed=seed)
    if name == "tpe":
        return TPESampler(seed=seed, constraints_func=effective_constraints_func)
    if name == "nsgaii":
        return NSGAIISampler(seed=seed, constraints_func=effective_constraints_func)
    if name == "nsgaiii":
        return NSGAIIISampler(seed=seed, constraints_func=effective_constraints_func)
    if name == "autosampler":
        if autosampler_module is None:
            raise ValueError("AutoSampler module is unavailable.")
        return autosampler_module.AutoSampler(seed=seed)
    if name == "hebo":
        if hebo_module is None:
            raise ValueError("HEBOSampler module is unavailable.")
        return hebo_module.HEBOSampler(
            search_space=search_space, seed=seed, num_obj=num_objectives
        )
    if name in _HEBO_UPDATED_VARIANTS:
        if hebo_updated_module is None:
            raise ValueError("HEBOSampler module is unavailable.")
        model_name = _HEBO_UPDATED_MODEL_NAME_BY_SAMPLER[name]
        sampler = hebo_updated_module.HEBOSampler(
            search_space=search_space,
            seed=seed,
            num_obj=num_objectives,
            model_name=model_name,
            track_surrogate_predictions=True,
            failure_constraint_model=use_binary_constraint,
            constraints_func=effective_constraints_func,
        )
        return sampler
    if name == "smac":
        if smac_module is None:
            raise ValueError("SMACSampler module is unavailable.")
        if num_objectives > 1:
            raise ValueError(
                "sampler='smac' supports only single-objective studies in this harness. "
                "Use sampler='smac_mo' for multi-objective studies."
            )
        return smac_module.SMACSampler(search_space=search_space, n_trials=n_trials, seed=seed)
    if name == "smac_mo":
        if smac_mo_module is None:
            raise ValueError("SMACMultiObjectiveSampler module is unavailable.")
        return smac_mo_module.SMACMultiObjectiveSampler(
            search_space=search_space,
            n_trials=n_trials,
            seed=seed,
            num_objectives=num_objectives,
        )
    if name == "moead":
        if moead_module is None:
            raise ValueError("MOEADSampler module is unavailable.")
        return moead_module.MOEADSampler(seed=seed)
    if name == "speaii":
        if speaii_module is None:
            raise ValueError("SPEAIISampler module is unavailable.")
        return speaii_module.SPEAIISampler(seed=seed)
    raise ValueError(f"Unsupported sampler name: {name}")


def _are_values_finite(values: Sequence[float] | None) -> bool:
    if values is None or len(values) == 0:
        return False
    return all(np.isfinite(float(v)) for v in values)


def _to_objective_values(value_or_values: float | Sequence[float]) -> list[float]:
    if isinstance(value_or_values, Sequence) and not isinstance(value_or_values, (str, bytes)):
        return [float(v) for v in value_or_values]
    return [float(value_or_values)]


def _compute_binary_constraint(
    problem: Any,
    params: dict[str, Any],
    value_or_values: float | Sequence[float],
) -> float:
    if hasattr(problem, "evaluate_constraints"):
        try:
            raw_constraints = problem.evaluate_constraints(params)
            constraints = np.asarray(raw_constraints, dtype=float).ravel()
            if constraints.size > 0:
                return 1.0 if np.any(constraints > 0.0) else -1.0
        except Exception:
            pass
    return -1.0 if _are_values_finite(_to_objective_values(value_or_values)) else 1.0


def _require_bbob_dim(dim: int, benchmark: str) -> int:
    if dim not in _BBOB_DIMS:
        choices = ", ".join(str(d) for d in _BBOB_DIMS)
        raise ValueError(f"{benchmark} requires dim in [{choices}], got {dim}.")
    return dim


def _require_dim_from_choices(dim: int, allowed_dims: tuple[int, ...], benchmark: str) -> int:
    if dim not in allowed_dims:
        choices = ", ".join(str(d) for d in allowed_dims)
        raise ValueError(f"{benchmark} requires dim in [{choices}], got {dim}.")
    return dim


def _bbob_ids_from_offset(
    *,
    offset: int,
    function_count: int,
    instance_base: int,
    instance_limit: int,
    function_start: int = 1,
) -> tuple[int, int]:
    function_id = function_start + (offset % function_count)
    instance_shift = offset // function_count
    instance_id = ((instance_base - 1 + instance_shift) % instance_limit) + 1
    return function_id, instance_id


def _make_problem_kwargs(
    args: argparse.Namespace, idx: int, benchmark_module: Any | None = None
) -> dict[str, Any]:
    offset = args.problem_offset + idx
    if _is_synthetic_benchmark(args.benchmark):
        options: dict[str, Any] = {
            "num_objectives": args.num_objectives,
            "seed_tag": args.benchmark,
        }
        options.update(_synthetic_forced_dynamics_from_alias(args.benchmark))
        return {
            "dim": args.dim,
            "seed": args.seed + offset,
            "data_type": args.data_type,
            "options": options,
        }
    if args.benchmark == "zdt":
        function_id = (offset % 6) + 1
        return {"function_id": function_id}
    if args.benchmark == "dtlz":
        function_id = (offset % 7) + 1
        return {
            "function_id": function_id,
            "n_objectives": args.num_objectives,
            "dimension": args.dim,
        }
    if args.benchmark == "dtlz_unknown_constraints":
        combo = _DTLZ_CONSTRAINED_COMBINATIONS[offset % len(_DTLZ_CONSTRAINED_COMBINATIONS)]
        return {
            "function_id": combo["function_id"],
            "constraint_type": combo["constraint_type"],
            "n_objectives": args.num_objectives,
            "dimension": args.dim,
        }
    if args.benchmark == "dtlz_constrained":
        combo = _DTLZ_CONSTRAINED_COMBINATIONS[offset % len(_DTLZ_CONSTRAINED_COMBINATIONS)]
        return {
            "function_id": combo["function_id"],
            "constraint_type": combo["constraint_type"],
            "n_objectives": args.num_objectives,
            "dimension": args.dim,
        }
    if args.benchmark == "wfg":
        function_id = (offset % 9) + 1
        return {
            "function_id": function_id,
            "n_objectives": args.num_objectives,
            "dimension": args.dim,
        }
    if args.benchmark == "bbob":
        function_id, instance_id = _bbob_ids_from_offset(
            offset=offset,
            function_count=_BBOB_FUNCTION_COUNT,
            instance_base=args.instance_id,
            instance_limit=_BBOB_INSTANCE_LIMIT,
        )
        dim = _require_bbob_dim(args.dim, "bbob")
        return {"function_id": function_id, "dimension": dim, "instance_id": instance_id}
    if args.benchmark == "bbob_biobj":
        function_id, instance_id = _bbob_ids_from_offset(
            offset=offset,
            function_count=_BBOB_BIOBJ_FUNCTION_COUNT,
            instance_base=args.instance_id,
            instance_limit=_BBOB_COCO_INSTANCE_LIMIT,
        )
        dim = _require_bbob_dim(args.dim, "bbob_biobj")
        return {"function_id": function_id, "dimension": dim, "instance_id": instance_id}
    if args.benchmark == "bbob_noisy":
        function_id, instance_id = _bbob_ids_from_offset(
            offset=offset,
            function_count=_BBOB_NOISY_FUNCTION_COUNT,
            function_start=_BBOB_NOISY_FUNCTION_START,
            instance_base=args.instance_id,
            instance_limit=_BBOB_COCO_INSTANCE_LIMIT,
        )
        dim = _require_bbob_dim(args.dim, "bbob_noisy")
        return {"function_id": function_id, "dimension": dim, "instance_id": instance_id}
    if args.benchmark == "bbob_mixint":
        function_id, instance_id = _bbob_ids_from_offset(
            offset=offset,
            function_count=_BBOB_FUNCTION_COUNT,
            instance_base=args.instance_id,
            instance_limit=_BBOB_COCO_INSTANCE_LIMIT,
        )
        dim = _require_dim_from_choices(args.dim, _BBOB_MIXINT_DIMS, "bbob_mixint")
        return {"function_id": function_id, "dimension": dim, "instance_id": instance_id}
    if args.benchmark == "bbob_largescale":
        function_id, instance_id = _bbob_ids_from_offset(
            offset=offset,
            function_count=_BBOB_FUNCTION_COUNT,
            instance_base=args.instance_id,
            instance_limit=_BBOB_COCO_INSTANCE_LIMIT,
        )
        dim = _require_dim_from_choices(args.dim, _BBOB_LARGESCALE_DIMS, "bbob_largescale")
        return {"function_id": function_id, "dimension": dim, "instance_id": instance_id}
    if args.benchmark == "bbob_constrained":
        function_id, instance_id = _bbob_ids_from_offset(
            offset=offset,
            function_count=_BBOB_CONSTRAINED_FUNCTION_COUNT,
            instance_base=args.instance_id,
            instance_limit=_BBOB_COCO_INSTANCE_LIMIT,
        )
        dim = _require_bbob_dim(args.dim, "bbob_constrained")
        return {"function_id": function_id, "dimension": dim, "instance_id": instance_id}
    if args.benchmark == "bbob_unknown_constraints":
        function_id, instance_id = _bbob_ids_from_offset(
            offset=offset,
            function_count=_BBOB_CONSTRAINED_FUNCTION_COUNT,
            instance_base=args.instance_id,
            instance_limit=_BBOB_COCO_INSTANCE_LIMIT,
        )
        dim = _require_bbob_dim(args.dim, "bbob_unknown_constraints")
        return {"function_id": function_id, "dimension": dim, "instance_id": instance_id}
    if args.benchmark == "bbob_biobj_mixint":
        function_id, instance_id = _bbob_ids_from_offset(
            offset=offset,
            function_count=_BBOB_BIOBJ_FUNCTION_COUNT,
            instance_base=args.instance_id,
            instance_limit=_BBOB_COCO_INSTANCE_LIMIT,
        )
        dim = _require_dim_from_choices(
            args.dim, _BBOB_MIXINT_DIMS, "bbob_biobj_mixint"
        )
        return {"function_id": function_id, "dimension": dim, "instance_id": instance_id}
    if args.benchmark == "hpolib":
        dataset_id = args.hpolib_dataset_id
        if dataset_id is None:
            available = None
            if benchmark_module is not None and hasattr(benchmark_module, "Problem"):
                available = getattr(benchmark_module.Problem, "available_dataset_names", None)
            if available:
                dataset_id = offset % len(available)
            else:
                dataset_id = offset
        return {"dataset_id": int(dataset_id), "seed": args.seed + offset}
    raise ValueError(f"Unsupported benchmark: {args.benchmark}")


def _benchmark_module_name(benchmark: str) -> str:
    if _is_synthetic_benchmark(benchmark):
        return "benchmarks/synthetic"
    return f"benchmarks/{benchmark}"


def _is_synthetic_benchmark(benchmark: str) -> bool:
    return benchmark == "synthetic" or benchmark.startswith("synthetic_")


def _synthetic_forced_dynamics_from_alias(benchmark: str) -> dict[str, bool]:
    if benchmark == "synthetic":
        return {
            "does_augmentations": False,
            "adds_noise": False,
            "has_failure_regions": False,
        }
    parts = benchmark.split("_")[1:]
    unknown = [part for part in parts if part not in _SYNTHETIC_ALIAS_COMPONENTS]
    if unknown:
        raise ValueError(
            f"Unsupported synthetic benchmark alias '{benchmark}'. Unknown components: {unknown}."
        )
    selected = set(parts)
    return {
        "does_augmentations": "aug" in selected,
        "adds_noise": "noise" in selected,
        "has_failure_regions": "fail" in selected,
    }


def _format_search_space(search_space: dict[str, BaseDistribution]) -> str:
    if not search_space:
        return "(empty search space)"
    lines = [f"- {name}: {dist}" for name, dist in search_space.items()]
    return "\n".join(lines)


def _problem_name(problem: Any, benchmark: str, idx: int) -> str:
    default_name = f"{benchmark}_{idx}"
    raw_name = getattr(problem, "name", None)
    if callable(raw_name):
        try:
            value = raw_name()
        except Exception:
            return default_name
        return str(value) if value is not None else default_name
    if raw_name is None:
        return default_name
    return str(raw_name)


def _suggest_from_distribution(
    trial: optuna.trial.Trial,
    name: str,
    dist: BaseDistribution,
) -> Any:
    if isinstance(dist, FloatDistribution):
        return trial.suggest_float(
            name,
            dist.low,
            dist.high,
            log=getattr(dist, "log", False),
            step=getattr(dist, "step", None),
        )
    if isinstance(dist, IntDistribution):
        return trial.suggest_int(
            name,
            dist.low,
            dist.high,
            log=getattr(dist, "log", False),
            step=getattr(dist, "step", None),
        )
    if isinstance(dist, CategoricalDistribution):
        return trial.suggest_categorical(name, dist.choices)
    raise ValueError(f"Unsupported distribution type for {name}: {type(dist).__name__}")


def _suggest_all_params(
    trial: optuna.trial.Trial,
    search_space: dict[str, BaseDistribution],
) -> None:
    for name, dist in search_space.items():
        _suggest_from_distribution(trial, name, dist)


def _normalize_score(
    value: float,
    direction: optuna.study.StudyDirection,
    y_min: float,
    y_max: float,
) -> float:
    span = max(1e-9, y_max - y_min)
    if direction == optuna.study.StudyDirection.MINIMIZE:
        score = (y_max - value) / span
    else:
        score = (value - y_min) / span
    return float(max(0.0, min(1.0, score)))


def _objective(
    problem: Any, trial: optuna.trial.Trial, use_binary_constraint: bool = False
) -> float | tuple[float, ...]:
    _suggest_all_params(trial, problem.search_space)
    value_or_values = problem.evaluate(trial.params)
    if not use_binary_constraint:
        return value_or_values
    binary_constraint = _compute_binary_constraint(problem, trial.params, value_or_values)
    trial.set_user_attr(_BINARY_CONSTRAINT_KEY, binary_constraint)
    if binary_constraint > 0.0:
        raise _ConstraintViolationError("Constraint violated.")
    return value_or_values


def _extract_best_values(study: optuna.Study, num_objectives: int) -> list[float]:
    if num_objectives <= 1:
        try:
            trial = study.best_trial
        except ValueError:
            return [float("nan")]
        if trial.value is None:
            return [float("nan")]
        return [float(trial.value)]
    best_trials = study.best_trials
    if not best_trials:
        return [float("nan")] * num_objectives
    trial = best_trials[0]
    if not trial.values:
        return [float("nan")] * num_objectives
    return [float(value) for value in trial.values]


def _normalize_objective_values(
    values: list[float],
    directions: list[optuna.study.StudyDirection],
    ranges: list[tuple[float, float]],
) -> list[float]:
    normalized: list[float] = []
    for idx, value in enumerate(values):
        if not np.isfinite(value):
            normalized.append(0.0)
            continue
        direction = directions[idx] if idx < len(directions) else directions[-1]
        y_min, y_max = ranges[idx] if idx < len(ranges) else ranges[-1]
        normalized.append(_normalize_score(value, direction, y_min, y_max))
    return normalized


def _objective_ranges(problem: Any) -> list[tuple[float, float]]:
    if hasattr(problem, "surfaces"):
        return [
            (surface.global_y_min, surface.global_y_max)
            for surface in problem.surfaces
        ]
    if hasattr(problem, "global_y_min") and hasattr(problem, "global_y_max"):
        return [(float(problem.global_y_min), float(problem.global_y_max))]
    return []


def _maximize_to_minimize(values: list[float]) -> list[float]:
    return [1.0 - v for v in values]


def _filter_non_dominated(points: list[list[float]]) -> list[list[float]]:
    nondominated: list[list[float]] = []
    for idx, point in enumerate(points):
        dominated = False
        for other_idx, other in enumerate(points):
            if idx == other_idx:
                continue
            if all(o <= p for o, p in zip(other, point)) and any(o < p for o, p in zip(other, point)):
                dominated = True
                break
        if not dominated:
            nondominated.append(point)
    return nondominated


def _study_hypervolume(
    study: optuna.Study,
    directions: list[optuna.study.StudyDirection],
    ranges: list[tuple[float, float]],
) -> float:
    if not directions:
        return 0.0
    normalized_front: list[list[float]] = []
    for trial in study.trials:
        values = trial.values
        if not values or len(values) != len(directions):
            continue
        normalized = _normalize_objective_values(list(values), directions, ranges)
        normalized_front.append(normalized)
    if not normalized_front:
        return 0.0
    minimization_front = [_maximize_to_minimize(point) for point in normalized_front]
    nondominated = _filter_non_dominated(minimization_front)
    if not nondominated:
        return 0.0
    reference_point = np.ones(len(directions), dtype=np.float64)
    hv = compute_hypervolume(
        np.asarray(nondominated, dtype=np.float64), reference_point, assume_pareto=True
    )
    return float(hv)


def _empirical_objective_ranges(
    studies: list[optuna.Study], directions: list[optuna.study.StudyDirection]
) -> list[tuple[float, float]]:
    if not directions:
        return []
    mins = [np.inf] * len(directions)
    maxs = [-np.inf] * len(directions)
    for study in studies:
        for trial in study.trials:
            values = trial.values
            if not values or len(values) != len(directions):
                continue
            for i, value in enumerate(values):
                mins[i] = min(mins[i], float(value))
                maxs[i] = max(maxs[i], float(value))
    ranges: list[tuple[float, float]] = []
    for i in range(len(directions)):
        if mins[i] == np.inf or maxs[i] == -np.inf:
            ranges.append((0.0, 1.0))
        else:
            ranges.append((float(mins[i]), float(maxs[i])))
    return ranges


def _history_callback(
    problem_name: str, sampler_name: str, enabled: bool
) -> Callable[[optuna.Study, optuna.trial.FrozenTrial], None]:
    def _callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if not enabled:
            return
        true_eval = (
            list(trial.values)
            if trial.values is not None
            else ([trial.value] if trial.value is not None else [])
        )
        pred_mean: float | None = None
        pred_std: float | None = None
        p_failure_model: float | None = None
        sampler = study.sampler
        if hasattr(sampler, "get_last_surrogate_predictions"):
            try:
                preds = sampler.get_last_surrogate_predictions()
                if preds is not None and not preds.empty:
                    row = preds.iloc[0]
                    pred_mean_value = row.get("_hebo_pred_mean")
                    pred_std_value = row.get("_hebo_pred_std")
                    pred_mean = float(pred_mean_value) if pred_mean_value is not None else None
                    pred_std = float(pred_std_value) if pred_std_value is not None else None
                    p_all = row.get("_hebo_pred_p_feasible_all")
                    if p_all is not None:
                        try:
                            p_failure_model = 1.0 - float(p_all)
                        except Exception:
                            pass
            except Exception:
                pred_mean = None
                pred_std = None
                p_failure_model = None

        logger.info(
            "[step] problem=%s sampler=%s trial=%d state=%s action=%s pred_mean=%s pred_std=%s p_failure_model=%s true_eval=%s",
            problem_name,
            sampler_name,
            trial.number,
            trial.state.name,
            trial.params,
            pred_mean,
            pred_std,
            p_failure_model,
            true_eval,
        )

    return _callback


@contextlib.contextmanager
def _maybe_suppress_sampler_output(enabled: bool) -> Any:
    if not enabled:
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def _configure_sampler_library_logging(hide_sampler_output: bool) -> None:
    if not hide_sampler_output:
        return
    # SMAC/ConfigSpace emit INFO logs via Python logging; keep them quiet unless explicitly requested.
    logging.getLogger("smac").setLevel(logging.WARNING)
    logging.getLogger("ConfigSpace").setLevel(logging.WARNING)


def _run_single_problem(
    *,
    idx: int,
    args: argparse.Namespace,
    registry_root: Path,
    benchmark_module_name: str,
) -> dict[str, Any]:
    """
    Worker function to process a single problem index (seed/function_id).
    Runs all requested samplers for this problem instance.
    """
    # Re-configure logging in worker
    optuna.logging.set_verbosity(getattr(optuna.logging, args.optuna_log_level))
    _configure_sampler_library_logging(args.sampler_output == "hide")
    
    # Reload modules locally in worker
    benchmark_module = _load_module(benchmark_module_name, registry_root)
    
    # Load sampler modules as needed
    hebo_module = None
    hebo_updated_module = None
    smac_module = None
    smac_mo_module = None
    autosampler_module = None
    moead_module = None
    speaii_module = None
    
    if "autosampler" in args.samplers:
        autosampler_module = _load_module("samplers/auto_sampler", registry_root)
    if "hebo" in args.samplers:
        hebo_module = _load_module("samplers/hebo", registry_root)
    if any(name in _HEBO_UPDATED_VARIANTS for name in args.samplers):
        hebo_updated_module = _load_module("samplers/hebo_updated", registry_root)
    if "smac" in args.samplers:
        smac_module = _load_module("samplers/smac_sampler", registry_root)
    if "smac_mo" in args.samplers:
        smac_mo_module = _load_module("samplers/smac_mo_sampler", registry_root)
    if "moead" in args.samplers:
        moead_module = _load_module("samplers/moead", registry_root)
    if "speaii" in args.samplers:
        speaii_module = _load_module("samplers/speaii", registry_root)

    # Initialize Problem
    problem_kwargs = _make_problem_kwargs(args, idx, benchmark_module)
    problem = benchmark_module.Problem(**problem_kwargs)
    
    use_binary_constraint = args.benchmark in {
        "bbob_unknown_constraints",
        "dtlz_unknown_constraints",
        "synthetic_fail",
        "synthetic_aug_fail",
        "synthetic_noise_fail",
        "synthetic_aug_noise_fail",
    }
    directions = problem.directions
    raw_objective_ranges = _objective_ranges(problem)
    problem_name = _problem_name(problem, args.benchmark, idx)
    
    info = {
        "problem": problem_name,
        "benchmark": args.benchmark,
        "problem_index": idx,
        "problem_kwargs": problem_kwargs,
        "directions": [direction.name for direction in directions],
        "samplers": [],
    }
    
    # Add metadata
    if raw_objective_ranges:
        info["objective_ranges"] = [
            {"min": obj_min, "max": obj_max}
            for obj_min, obj_max in raw_objective_ranges
        ]
    if hasattr(problem, "global_y_min") and hasattr(problem, "global_y_max"):
        info["range"] = {
            "min": float(problem.global_y_min),
            "max": float(problem.global_y_max),
        }
    if hasattr(problem, "num_objectives"):
        info["num_objectives"] = int(problem.num_objectives)
    if use_binary_constraint:
        info["constraints"] = True
    if len(directions) == 1:
        info["direction"] = directions[0].name

    # Run Samplers
    sampler_results: list[dict[str, Any]] = []
    studies: list[optuna.Study] = []
    
    for sampler_name in args.samplers:
        sampler_seed = args.seed + args.problem_offset + idx
        sampler = _instantiate_sampler(
            sampler_name,
            problem.search_space,
            sampler_seed,
            args.trials,
            hebo_module,
            hebo_updated_module,
            smac_module,
            smac_mo_module,
            autosampler_module,
            moead_module,
            speaii_module,
            len(directions),
            use_binary_constraint,
        )
        study_kwargs: dict[str, Any] = {"sampler": sampler}
        if len(directions) == 1:
            study_kwargs["direction"] = directions[0]
        else:
            study_kwargs["directions"] = directions

        study = optuna.create_study(**study_kwargs)
        with _maybe_suppress_sampler_output(args.sampler_output == "hide"):
            study.optimize(
                lambda trial: _objective(problem, trial, use_binary_constraint),
                n_trials=args.trials,
                catch=(_ConstraintViolationError,),
                callbacks=[_history_callback(problem_name, sampler_name, args.print_step_history)],
            )
        studies.append(study)
        best_values = _extract_best_values(study, len(directions))
        if len(directions) == 1:
            try:
                best_params = study.best_params
            except ValueError:
                best_params = {}
        else:
            pareto = study.best_trials
            best_params = pareto[0].params if pareto else {}
            
        sampler_results.append(
            {
                "name": sampler_name,
                "study": study, # Kept local for hypervolume calc, not returned in final Dict
                "best_values": best_values,
                "best_params": best_params,
                "trial_state_counts": dict(Counter(t.state.name for t in study.trials)),
            }
        )

    # Compute Hypervolumes and Ranges locally
    objective_ranges = (
        raw_objective_ranges
        if raw_objective_ranges and len(raw_objective_ranges) == len(directions)
        else _empirical_objective_ranges(studies, directions)
    )
    info["objective_ranges"] = [
        {"min": obj_min, "max": obj_max}
        for obj_min, obj_max in objective_ranges
    ]

    for result in sampler_results:
        normalized_scores = _normalize_objective_values(
            result["best_values"], directions, objective_ranges
        )
        hypervolume_score = _study_hypervolume(result["study"], directions, objective_ranges)
        
        sampler_entry: dict[str, Any] = {
            "name": result["name"],
            "best_values": result["best_values"],
            "normalized_scores": normalized_scores,
            "hypervolume": hypervolume_score,
            "best_params": result["best_params"],
            "trial_state_counts": result["trial_state_counts"],
        }
        if len(result["best_values"]) == 1:
            sampler_entry["best_value"] = result["best_values"][0]
            sampler_entry["normalized_score"] = normalized_scores[0]
        info["samplers"].append(sampler_entry)

    return info


def _run_single_problem_task(task: tuple) -> tuple[int, dict[str, Any]]:
    """
    Wrapper for multiprocessing submission.
    Unpacks picklable arguments and runs the worker.
    """
    idx, args_dict, registry_root_str, benchmark_module_name = task
    
    # Reconstruct namespace
    args = argparse.Namespace(**args_dict)
    registry_root = Path(registry_root_str)
    
    result = _run_single_problem(
        idx=idx, 
        args=args, 
        registry_root=registry_root, 
        benchmark_module_name=benchmark_module_name
    )
    return idx, result


def _force_stop_executor_workers(
    executor: concurrent.futures.ProcessPoolExecutor | None,
    *,
    grace_period_sec: float = 2.0,
) -> None:
    processes = _snapshot_executor_processes(executor)
    _force_stop_processes(processes, grace_period_sec=grace_period_sec)


def _snapshot_executor_processes(
    executor: concurrent.futures.ProcessPoolExecutor | None,
) -> list[Any]:
    if executor is None:
        return []
    raw_processes = getattr(executor, "_processes", None)
    if not raw_processes:
        return []
    return [process for process in raw_processes.values() if process is not None]


def _force_stop_processes(
    processes: Sequence[Any],
    *,
    grace_period_sec: float = 2.0,
) -> None:
    if not processes:
        return

    for process in processes:
        if not process.is_alive():
            continue
        try:
            process.terminate()
        except Exception as exc:
            logger.debug("Failed to terminate worker pid=%s: %s", process.pid, exc)

    if grace_period_sec > 0.0:
        deadline = time.monotonic() + grace_period_sec
        while time.monotonic() < deadline:
            if not any(process.is_alive() for process in processes):
                return
            time.sleep(0.05)

    for process in processes:
        if not process.is_alive():
            continue
        try:
            if hasattr(process, "kill"):
                process.kill()
            elif process.pid is not None:
                os.kill(process.pid, signal.SIGKILL)
        except Exception as exc:
            logger.debug("Failed to kill worker pid=%s: %s", process.pid, exc)


def _shutdown_executor(
    executor: concurrent.futures.ProcessPoolExecutor | None,
    *,
    cancel_futures: bool,
    force_kill: bool,
) -> None:
    if executor is None:
        return
    processes = _snapshot_executor_processes(executor)
    if force_kill:
        _force_stop_processes(processes)
    executor.shutdown(wait=False, cancel_futures=cancel_futures)
    if force_kill:
        _force_stop_processes(processes, grace_period_sec=0.0)


def main() -> None:
    args = parse_args()
    optuna.logging.set_verbosity(getattr(optuna.logging, args.optuna_log_level))
    
    if args.output_file is None:
        dim_suffix = f"{args.dim}d" if args.benchmark in _BENCHMARKS_USING_DIM else ""
        objective_suffix = (
            f"_{args.num_objectives}obj"
            if args.benchmark in _BENCHMARKS_USING_NUM_OBJECTIVES
            else ""
        )
        args.output_file = (
            Path("tests")
            / f"{args.benchmark}/{dim_suffix}{objective_suffix}_{args.n_problems}_{args.trials}problems.json"
        )
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    registry_root = _resolve_registry_root()
    benchmark_module_name = _benchmark_module_name(args.benchmark)
    
    # Prepare Tasks
    # We pass args as a dict to ensure picklability across processes
    tasks = [
        (idx, vars(args).copy(), str(registry_root), benchmark_module_name)
        for idx in range(args.n_problems)
    ]
    
    worker_count = max(1, min(args.workers, len(tasks)))
    logger.info(f"Starting comparison with {len(tasks)} problems using {worker_count} workers.")
    
    detailed_results: list[dict[str, Any]] = []
    # Temporary storage to reconstruct order if needed, though aggregation order doesn't matter
    results_by_idx: dict[int, dict[str, Any]] = {}
    
    executor = None
    try:
        if worker_count == 1:
            # Sequential execution
            with tqdm(total=len(tasks), desc="Problems", unit="prob") as pbar:
                for task in tasks:
                    idx, _ = task
                    try:
                        _, info = _run_single_problem_task(task)
                        results_by_idx[idx] = info
                    except KeyboardInterrupt:
                        raise
                    except Exception as exc:
                        logger.error(f"Problem index {idx} failed: {exc}")
                    pbar.update(1)
        else:
            # Parallel execution
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=worker_count)
            future_to_task = {
                executor.submit(_run_single_problem_task, task): task 
                for task in tasks
            }
            
            with tqdm(total=len(future_to_task), desc="Problems", unit="prob") as pbar:
                for future in concurrent.futures.as_completed(future_to_task):
                    task_data = future_to_task[future]
                    task_idx = task_data[0]
                    
                    try:
                        idx, info = future.result()
                        results_by_idx[idx] = info
                    except KeyboardInterrupt:
                        raise
                    except Exception as exc:
                        logger.error(f"Problem index {task_idx} failed in worker: {exc}")
                    
                    pbar.update(1)
            
            executor.shutdown(wait=True)

    except KeyboardInterrupt:
        logger.warning("\nReceived Ctrl+C. Stopping workers immediately...")
        _shutdown_executor(executor, cancel_futures=True, force_kill=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected global error: {e}")
        _shutdown_executor(executor, cancel_futures=True, force_kill=True)
        sys.exit(1)

    # Collect valid results
    detailed_results = [results_by_idx[idx] for idx in sorted(results_by_idx.keys())]
    
    if not detailed_results:
        logger.warning("No results to save.")
        return

    # Post-processing aggregation
    aggregate: dict[str, list[float]] = {name: [] for name in args.samplers}
    
    for info in detailed_results:
        for sampler_entry in info["samplers"]:
            name = sampler_entry["name"]
            hv = sampler_entry.get("hypervolume", 0.0)
            if name in aggregate:
                aggregate[name].append(hv)

    summary = {
        "aggregate": {
            name: {
                "mean_hypervolume": float(mean(scores)) if scores else 0.0,
                "runs": len(scores),
            }
            for name, scores in aggregate.items()
        },
        "details": detailed_results,
    }
    
    args.output_file.write_text(json.dumps(summary, indent=2))
    logger.info("Saved comparison summary to %s", args.output_file)


if __name__ == "__main__":
    main()
