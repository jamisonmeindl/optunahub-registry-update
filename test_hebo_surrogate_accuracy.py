"""Evaluate surrogate prediction accuracy across hebo_updated sampler variants."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Callable

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


import numpy as np
import optuna
import optunahub
import pandas as pd
from optuna.distributions import BaseDistribution, CategoricalDistribution, FloatDistribution, IntDistribution
from optunahub import hub as _hub

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

_BBOB_DIMS = (2, 3, 5, 10, 20, 40)
_BBOB_MIXINT_DIMS = (5, 10, 20, 40, 80, 160)
_BBOB_LARGESCALE_DIMS = (20, 40, 80, 160, 320, 640)
_BBOB_FUNCTION_COUNT = 24
_BBOB_BIOBJ_FUNCTION_COUNT = 92
_BBOB_NOISY_FUNCTION_START = 101
_BBOB_NOISY_FUNCTION_COUNT = 30
_BBOB_CONSTRAINED_FUNCTION_COUNT = 54
_BBOB_INSTANCE_LIMIT = 110
_BBOB_COCO_INSTANCE_LIMIT = 15
_DTLZ_CONSTRAINED_COMBINATIONS = (
    {"constraint_type": 1, "function_id": 1},
    {"constraint_type": 1, "function_id": 3},
    {"constraint_type": 2, "function_id": 2},
    {"constraint_type": 3, "function_id": 1},
    {"constraint_type": 3, "function_id": 4},
)
_SYNTHETIC_ALIAS_COMPONENTS = {"aug", "noise", "fail"}
_ALL_BENCHMARKS = (
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
)

_HEBO_UPDATED_MODEL_NAME_BY_SAMPLER = {
    "hebo_updated": None,
    "hebo_updated_gp": "gp",
    "hebo_updated_rf": "rf",
    "hebo_updated_deep_ensemble": "deep_ensemble",
    "hebo_updated_catboost": "catboost",
    "hebo_updated_psgld": "psgld",
    "hebo_updated_svidkl": "svidkl",
}
_HEBO_OPTION_ALIASES = {
    "default": "hebo_updated",
    "gp": "hebo_updated_gp",
    "rf": "hebo_updated_rf",
    "deep_ensemble": "hebo_updated_deep_ensemble",
    "catboost": "hebo_updated_catboost",
    "psgld": "hebo_updated_psgld",
    "svidkl": "hebo_updated_svidkl",
}
_HEBO_OPTION_CHOICES = tuple(_HEBO_UPDATED_MODEL_NAME_BY_SAMPLER.keys()) + tuple(
    _HEBO_OPTION_ALIASES.keys()
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hebo_updated variants and evaluate surrogate prediction accuracy."
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=list(_ALL_BENCHMARKS),
        default=list(_ALL_BENCHMARKS),
        help="Benchmark presets to evaluate.",
    )
    parser.add_argument(
        "--benchmark",
        choices=list(_ALL_BENCHMARKS),
        default=None,
        help="Deprecated single-benchmark alias for --benchmarks.",
    )
    parser.add_argument("--n-problems", type=int, default=10, help="Number of problem instances.")
    parser.add_argument("--problem-offset", type=int, default=0, help="Problem index offset.")
    parser.add_argument("--trials", type=int, default=50, help="Trials per study.")
    parser.add_argument("--dim", type=int, default=10, help="Problem dimensionality.")
    parser.add_argument(
        "--data-type",
        choices=["Mixed", "Continuous"],
        default="Mixed",
        help="Synthetic benchmark data type.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    parser.add_argument("--instance-id", type=int, default=1, help="Base BBOB instance id.")
    parser.add_argument(
        "--num-objectives",
        type=int,
        default=1,
        help="Number of objectives for presets that support it (e.g. synthetic/dtlz/wfg).",
    )
    parser.add_argument(
        "--hpolib-dataset-id",
        type=int,
        default=None,
        help="Fixed dataset_id for hpolib (omit to cycle datasets).",
    )
    parser.add_argument(
        "--hebo-options",
        nargs="+",
        default=[
            "hebo_updated_gp",
            "hebo_updated_rf",
            # "hebo_updated_deep_ensemble",
            "hebo_updated_catboost",
            "hebo_updated_psgld",
            "hebo_updated_svidkl",
        ],
        choices=list(_HEBO_OPTION_CHOICES),
        help=(
            "hebo_updated sampler variants to compare. "
            "Also accepts legacy aliases: default/gp/rf/deep_ensemble/catboost/psgld/svidkl."
        ),
    )
    parser.add_argument(
        "--rand-sample",
        type=int,
        default=None,
        help="Override HEBO random warmup trials (None = backend default).",
    )
    parser.add_argument(
        "--global-optimum-samples",
        type=int,
        default=20000,
        help=(
            "Fallback random samples to estimate global-optimum params when not exposed "
            "by the benchmark."
        ),
    )
    parser.add_argument(
        "--sampler-output",
        choices=["hide", "show"],
        default="hide",
        help="Whether to hide sampler stdout/stderr and sampler library INFO logs.",
    )
    parser.add_argument(
        "--print-step-history",
        action="store_true",
        help="Print trial-by-trial pred/true values.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Output JSON path.",
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


def _normalize_hebo_options(options: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for name in options:
        canonical = _HEBO_OPTION_ALIASES.get(name, name)
        if canonical not in _HEBO_UPDATED_MODEL_NAME_BY_SAMPLER:
            raise ValueError(f"Unsupported hebo option: {name}")
        if canonical in seen:
            continue
        seen.add(canonical)
        normalized.append(canonical)
    return normalized


def _normalize_benchmarks(
    benchmarks: list[str],
    benchmark: str | None,
) -> list[str]:
    selected = [benchmark] if benchmark is not None else benchmarks
    normalized: list[str] = []
    seen: set[str] = set()
    for name in selected:
        if name in seen:
            continue
        seen.add(name)
        normalized.append(name)
    return normalized


def _configure_sampler_library_logging(hide_sampler_output: bool) -> None:
    if not hide_sampler_output:
        return
    logging.getLogger("smac").setLevel(logging.WARNING)
    logging.getLogger("ConfigSpace").setLevel(logging.WARNING)


@contextlib.contextmanager
def _maybe_suppress_sampler_output(enabled: bool) -> Any:
    if not enabled:
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def _require_bbob_dim(dim: int, benchmark: str) -> int:
    if dim not in _BBOB_DIMS:
        allowed = ", ".join(str(d) for d in _BBOB_DIMS)
        raise ValueError(f"{benchmark} requires dim in [{allowed}], got {dim}.")
    return dim


def _require_dim_from_choices(dim: int, allowed_dims: tuple[int, ...], benchmark: str) -> int:
    if dim not in allowed_dims:
        allowed = ", ".join(str(d) for d in allowed_dims)
        raise ValueError(f"{benchmark} requires dim in [{allowed}], got {dim}.")
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
    args: argparse.Namespace, benchmark: str, idx: int, benchmark_module: Any | None = None
) -> dict[str, Any]:
    offset = args.problem_offset + idx
    if _is_synthetic_benchmark(benchmark):
        options: dict[str, Any] = {
            "num_objectives": args.num_objectives,
            "seed_tag": benchmark,
        }
        options.update(_synthetic_forced_dynamics_from_alias(benchmark))
        return {
            "dim": args.dim,
            "seed": args.seed + offset,
            "data_type": args.data_type,
            "options": options,
        }
    if benchmark == "zdt":
        function_id = (offset % 6) + 1
        return {"function_id": function_id}
    if benchmark == "dtlz":
        function_id = (offset % 7) + 1
        return {
            "function_id": function_id,
            "n_objectives": args.num_objectives,
            "dimension": args.dim,
        }
    if benchmark == "dtlz_unknown_constraints":
        combo = _DTLZ_CONSTRAINED_COMBINATIONS[offset % len(_DTLZ_CONSTRAINED_COMBINATIONS)]
        return {
            "function_id": combo["function_id"],
            "constraint_type": combo["constraint_type"],
            "n_objectives": args.num_objectives,
            "dimension": args.dim,
        }
    if benchmark == "dtlz_constrained":
        combo = _DTLZ_CONSTRAINED_COMBINATIONS[offset % len(_DTLZ_CONSTRAINED_COMBINATIONS)]
        return {
            "function_id": combo["function_id"],
            "constraint_type": combo["constraint_type"],
            "n_objectives": args.num_objectives,
            "dimension": args.dim,
        }
    if benchmark == "wfg":
        function_id = (offset % 9) + 1
        return {
            "function_id": function_id,
            "n_objectives": args.num_objectives,
            "dimension": args.dim,
        }
    if benchmark == "bbob":
        function_id, instance_id = _bbob_ids_from_offset(
            offset=offset,
            function_count=_BBOB_FUNCTION_COUNT,
            instance_base=args.instance_id,
            instance_limit=_BBOB_INSTANCE_LIMIT,
        )
        dim = _require_bbob_dim(args.dim, "bbob")
        return {"function_id": function_id, "dimension": dim, "instance_id": instance_id}
    if benchmark == "bbob_biobj":
        function_id, instance_id = _bbob_ids_from_offset(
            offset=offset,
            function_count=_BBOB_BIOBJ_FUNCTION_COUNT,
            instance_base=args.instance_id,
            instance_limit=_BBOB_COCO_INSTANCE_LIMIT,
        )
        dim = _require_bbob_dim(args.dim, "bbob_biobj")
        return {"function_id": function_id, "dimension": dim, "instance_id": instance_id}
    if benchmark == "bbob_noisy":
        function_id, instance_id = _bbob_ids_from_offset(
            offset=offset,
            function_count=_BBOB_NOISY_FUNCTION_COUNT,
            function_start=_BBOB_NOISY_FUNCTION_START,
            instance_base=args.instance_id,
            instance_limit=_BBOB_COCO_INSTANCE_LIMIT,
        )
        dim = _require_bbob_dim(args.dim, "bbob_noisy")
        return {"function_id": function_id, "dimension": dim, "instance_id": instance_id}
    if benchmark == "bbob_mixint":
        function_id, instance_id = _bbob_ids_from_offset(
            offset=offset,
            function_count=_BBOB_FUNCTION_COUNT,
            instance_base=args.instance_id,
            instance_limit=_BBOB_COCO_INSTANCE_LIMIT,
        )
        dim = _require_dim_from_choices(args.dim, _BBOB_MIXINT_DIMS, "bbob_mixint")
        return {"function_id": function_id, "dimension": dim, "instance_id": instance_id}
    if benchmark == "bbob_largescale":
        function_id, instance_id = _bbob_ids_from_offset(
            offset=offset,
            function_count=_BBOB_FUNCTION_COUNT,
            instance_base=args.instance_id,
            instance_limit=_BBOB_COCO_INSTANCE_LIMIT,
        )
        dim = _require_dim_from_choices(args.dim, _BBOB_LARGESCALE_DIMS, "bbob_largescale")
        return {"function_id": function_id, "dimension": dim, "instance_id": instance_id}
    if benchmark == "bbob_constrained":
        function_id, instance_id = _bbob_ids_from_offset(
            offset=offset,
            function_count=_BBOB_CONSTRAINED_FUNCTION_COUNT,
            instance_base=args.instance_id,
            instance_limit=_BBOB_COCO_INSTANCE_LIMIT,
        )
        dim = _require_bbob_dim(args.dim, "bbob_constrained")
        return {"function_id": function_id, "dimension": dim, "instance_id": instance_id}
    if benchmark == "bbob_unknown_constraints":
        function_id, instance_id = _bbob_ids_from_offset(
            offset=offset,
            function_count=_BBOB_CONSTRAINED_FUNCTION_COUNT,
            instance_base=args.instance_id,
            instance_limit=_BBOB_COCO_INSTANCE_LIMIT,
        )
        dim = _require_bbob_dim(args.dim, "bbob_unknown_constraints")
        return {"function_id": function_id, "dimension": dim, "instance_id": instance_id}
    if benchmark == "bbob_biobj_mixint":
        function_id, instance_id = _bbob_ids_from_offset(
            offset=offset,
            function_count=_BBOB_BIOBJ_FUNCTION_COUNT,
            instance_base=args.instance_id,
            instance_limit=_BBOB_COCO_INSTANCE_LIMIT,
        )
        dim = _require_dim_from_choices(args.dim, _BBOB_MIXINT_DIMS, "bbob_biobj_mixint")
        return {"function_id": function_id, "dimension": dim, "instance_id": instance_id}
    if benchmark == "hpolib":
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
    raise ValueError(f"Unsupported benchmark: {benchmark}")


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


def _objective(problem: Any, trial: optuna.trial.Trial) -> float:
    for name, dist in problem.search_space.items():
        _suggest_from_distribution(trial, name, dist)
    value = problem.evaluate(trial.params)
    if isinstance(value, tuple):
        return float(value[0])
    return float(value)


def _sample_from_distribution(
    rng: np.random.Generator,
    dist: BaseDistribution,
) -> Any:
    if isinstance(dist, FloatDistribution):
        if getattr(dist, "log", False):
            return float(np.exp(rng.uniform(np.log(dist.low), np.log(dist.high))))
        if getattr(dist, "step", None) is not None:
            step = float(dist.step)
            n_steps = int(round((dist.high - dist.low) / step))
            idx = int(rng.integers(0, n_steps + 1))
            return float(dist.low + idx * step)
        return float(rng.uniform(dist.low, dist.high))
    if isinstance(dist, IntDistribution):
        if getattr(dist, "log", False):
            raw = np.exp(rng.uniform(np.log(dist.low), np.log(dist.high)))
            return int(np.clip(int(round(raw)), int(dist.low), int(dist.high)))
        step = int(getattr(dist, "step", 1) or 1)
        n_steps = int((dist.high - dist.low) // step)
        idx = int(rng.integers(0, n_steps + 1))
        return int(dist.low + idx * step)
    if isinstance(dist, CategoricalDistribution):
        return rng.choice(dist.choices)
    raise ValueError(f"Unsupported distribution type: {type(dist).__name__}")


def _coerce_params_vector_to_dict(
    values: Any,
    search_space: dict[str, BaseDistribution],
) -> dict[str, Any] | None:
    if values is None:
        return None
    if isinstance(values, dict):
        params = {name: values[name] for name in search_space if name in values}
        return params if len(params) == len(search_space) else None
    array = np.asarray(values, dtype=object).ravel()
    if len(array) != len(search_space):
        return None
    return {name: array[idx].item() if isinstance(array[idx], np.generic) else array[idx] for idx, name in enumerate(search_space)}


def _lookup_callable_or_attr(obj: Any, names: list[str]) -> Any:
    for name in names:
        if not hasattr(obj, name):
            continue
        value = getattr(obj, name)
        if callable(value):
            try:
                value = value()
            except Exception:
                continue
        return value
    return None


def _estimate_global_optimum_from_sampling(
    problem: Any,
    *,
    seed: int,
    n_samples: int,
) -> tuple[dict[str, Any], float]:
    rng = np.random.default_rng(seed)
    search_space = problem.search_space
    direction = problem.directions[0]
    best_params: dict[str, Any] | None = None
    best_value: float | None = None

    for _ in range(max(1, n_samples)):
        params = {
            name: _sample_from_distribution(rng, dist) for name, dist in search_space.items()
        }
        value_raw = problem.evaluate(params)
        value = float(value_raw[0]) if isinstance(value_raw, tuple) else float(value_raw)
        if not np.isfinite(value):
            continue
        if (
            best_value is None
            or (direction == optuna.study.StudyDirection.MINIMIZE and value < best_value)
            or (direction == optuna.study.StudyDirection.MAXIMIZE and value > best_value)
        ):
            best_value = value
            best_params = params

    if best_params is None or best_value is None:
        raise RuntimeError("Failed to estimate a finite global optimum from random sampling.")
    return best_params, best_value


def _resolve_global_optimum(
    problem: Any,
    *,
    benchmark: str,
    seed: int,
    n_samples: int,
) -> tuple[dict[str, Any], float, str]:
    search_space = problem.search_space
    candidate_objs = [problem, getattr(problem, "_problem", None), getattr(problem, "_env", None)]

    known_params = None
    known_value = None
    for obj in candidate_objs:
        if obj is None:
            continue
        if known_params is None:
            known_params = _lookup_callable_or_attr(
                obj,
                ["xopt", "x_opt", "best_parameter", "optimal_parameter", "optimal_point"],
            )
        if known_value is None:
            known_value = _lookup_callable_or_attr(
                obj,
                ["fopt", "f_opt", "best_value", "optimal_value", "global_optimum"],
            )

    params_dict = _coerce_params_vector_to_dict(known_params, search_space)
    if params_dict is not None and known_value is not None:
        try:
            value = float(np.asarray(known_value, dtype=float).ravel()[0])
            return params_dict, value, "known"
        except Exception:
            pass

    if params_dict is not None:
        value_raw = problem.evaluate(params_dict)
        value = float(value_raw[0]) if isinstance(value_raw, tuple) else float(value_raw)
        return params_dict, value, "known_params"

    sampled_params, sampled_value = _estimate_global_optimum_from_sampling(
        problem, seed=seed, n_samples=n_samples
    )
    source = f"sampled_{benchmark}"
    return sampled_params, sampled_value, source


def _params_to_hebo_row(
    params: dict[str, Any],
    search_space: dict[str, BaseDistribution],
) -> pd.DataFrame:
    row: dict[str, Any] = {}
    for name, dist in search_space.items():
        value = params[name]
        if (
            isinstance(dist, (IntDistribution, FloatDistribution))
            and not dist.log
            and dist.step is not None
        ):
            row[name] = (float(value) - float(dist.low)) / float(dist.step)
        else:
            row[name] = value
    return pd.DataFrame([row])


def _predict_surrogate_at_params(
    study: optuna.Study,
    *,
    params: dict[str, Any],
    search_space: dict[str, BaseDistribution],
) -> tuple[float | None, float | None]:
    sampler = study.sampler
    hebo = getattr(sampler, "_hebo", None)
    compute_stats = getattr(sampler, "_compute_prediction_stats", None)
    if hebo is None or not callable(compute_stats):
        return None, None
    try:
        suggestions = _params_to_hebo_row(params, search_space)
        stats = compute_stats(hebo, suggestions)
    except Exception:
        return None, None
    if stats is None:
        return None, None
    mean_arr, std_arr = stats
    mean_values = np.asarray(mean_arr, dtype=float).ravel()
    std_values = np.asarray(std_arr, dtype=float).ravel()
    if len(mean_values) == 0 or len(std_values) == 0:
        return None, None
    pred_mean = float(mean_values[0])
    if study.direction == optuna.study.StudyDirection.MAXIMIZE:
        pred_mean = -pred_mean
    pred_std = float(np.abs(std_values[0]))
    return pred_mean, pred_std


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    return ranks


def _compute_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [
        r
        for r in records
        if r.get("pred_mean") is not None
        and r.get("true_value") is not None
        and np.isfinite(float(r["pred_mean"]))
        and np.isfinite(float(r["true_value"]))
    ]
    if not valid:
        return {
            "n_trials_total": len(records),
            "n_with_prediction": 0,
            "n_without_prediction": len(records),
        }

    y = np.asarray([float(r["true_value"]) for r in valid], dtype=float)
    yhat = np.asarray([float(r["pred_mean"]) for r in valid], dtype=float)
    err = yhat - y

    # Scale used for normalized errors. We prefer target magnitude, and
    # fall back to prediction spread/magnitude when the target is near zero.
    scale_candidates = [
        float(np.max(np.abs(y))) if len(y) > 0 else 0.0,
        float(np.std(yhat)) if len(yhat) > 0 else 0.0,
        float(np.mean(np.abs(yhat))) if len(yhat) > 0 else 0.0,
    ]
    scale = max([v for v in scale_candidates if np.isfinite(v)] + [1.0, 1e-12])

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    bias = float(np.mean(err))
    normalized_mae = float(mae / scale)
    normalized_rmse = float(rmse / scale)
    normalized_bias = float(bias / scale)

    pearson = None
    if len(y) > 1 and float(np.std(y)) > 0.0 and float(np.std(yhat)) > 0.0:
        pearson = float(np.corrcoef(y, yhat)[0, 1])

    spearman = None
    if len(y) > 1:
        ry = _rankdata(y)
        ryhat = _rankdata(yhat)
        if float(np.std(ry)) > 0.0 and float(np.std(ryhat)) > 0.0:
            spearman = float(np.corrcoef(ry, ryhat)[0, 1])

    with_std = [
        r
        for r in valid
        if r.get("pred_std") is not None and np.isfinite(float(r["pred_std"]))
    ]
    coverage_1sigma = None
    nll = None
    if with_std:
        y_std = np.asarray([float(r["pred_std"]) for r in with_std], dtype=float)
        y_true_std = np.asarray([float(r["true_value"]) for r in with_std], dtype=float)
        yhat_std = np.asarray([float(r["pred_mean"]) for r in with_std], dtype=float)
        abs_err_std = np.abs(yhat_std - y_true_std)
        coverage_1sigma = float(np.mean(abs_err_std <= y_std))
        sigma = np.maximum(y_std, 1e-8)
        sq_err = (yhat_std - y_true_std) ** 2
        nll_terms = 0.5 * np.log(2.0 * math.pi * sigma**2) + sq_err / (2.0 * sigma**2)
        nll = float(np.mean(nll_terms))

    return {
        "n_trials_total": len(records),
        "n_with_prediction": len(valid),
        "n_without_prediction": len(records) - len(valid),
        "prediction_coverage": float(len(valid) / max(len(records), 1)),
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "normalized_mae": normalized_mae,
        "normalized_rmse": normalized_rmse,
        "normalized_bias": normalized_bias,
        "normalization_scale": scale,
        "pearson": pearson,
        "spearman": spearman,
        "coverage_1sigma": coverage_1sigma,
        "nll": nll,
        "gaussian_nll": nll,
    }


def _average_metrics(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not metric_rows:
        return {}
    numeric_keys = sorted(
        {
            key
            for row in metric_rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and np.isfinite(float(value))
        }
    )
    averaged: dict[str, Any] = {"n_runs": len(metric_rows)}
    for key in numeric_keys:
        values = [
            float(row[key])
            for row in metric_rows
            if key in row and isinstance(row[key], (int, float)) and np.isfinite(float(row[key]))
        ]
        if values:
            averaged[key] = float(np.mean(values))
    return averaged


def _build_callback(
    *,
    problem_name: str,
    option_name: str,
    records: list[dict[str, Any]],
    global_optimum_params: dict[str, Any],
    global_optimum_value: float,
    search_space: dict[str, BaseDistribution],
    enabled: bool,
) -> Callable[[optuna.Study, optuna.trial.FrozenTrial], None]:
    def _callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        pred_mean, pred_std = _predict_surrogate_at_params(
            study,
            params=global_optimum_params,
            search_space=search_space,
        )

        records.append(
            {
                "trial": int(trial.number),
                "pred_mean": pred_mean,
                "pred_std": pred_std,
                "true_value": float(global_optimum_value),
            }
        )

        if enabled:
            logger.info(
                "[step] problem=%s option=%s trial=%d pred_mean=%s pred_std=%s global_optimum=%s",
                problem_name,
                option_name,
                trial.number,
                pred_mean,
                pred_std,
                global_optimum_value,
            )

    return _callback


def main() -> None:
    args = parse_args()
    args.hebo_options = _normalize_hebo_options(args.hebo_options)
    args.benchmarks = _normalize_benchmarks(args.benchmarks, args.benchmark)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _configure_sampler_library_logging(args.sampler_output == "hide")

    registry_root = _resolve_registry_root()
    hebo_updated_module = _load_module("samplers/hebo_updated", registry_root)
    benchmark_modules: dict[str, Any] = {
        benchmark: _load_module(_benchmark_module_name(benchmark), registry_root)
        for benchmark in args.benchmarks
    }

    if args.output_file is None:
        benchmark_tag = args.benchmarks[0] if len(args.benchmarks) == 1 else "all"
        args.output_file = (
            Path("experiment_results")
            / f"hebo_surrogate_accuracy_{benchmark_tag}_{args.dim}d_{args.n_problems}_{args.trials}trials.json"
        )
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    aggregate_records: dict[str, list[dict[str, Any]]] = {opt: [] for opt in args.hebo_options}
    aggregate_metric_rows: dict[str, list[dict[str, Any]]] = {opt: [] for opt in args.hebo_options}
    details: list[dict[str, Any]] = []

    for benchmark in args.benchmarks:
        benchmark_module = benchmark_modules[benchmark]
        logger.info("Running benchmark: %s", benchmark)
        for idx in range(args.n_problems):
            problem_kwargs = _make_problem_kwargs(args, benchmark, idx, benchmark_module)
            problem = benchmark_module.Problem(**problem_kwargs)

            p_name = _problem_name(problem, benchmark, idx)
            logger.info(
                "Running problem %d/%d [%s]: %s",
                idx + 1,
                args.n_problems,
                benchmark,
                p_name,
            )
            global_optimum_params, global_optimum_value, global_optimum_source = _resolve_global_optimum(
                problem,
                benchmark=benchmark,
                seed=args.seed + args.problem_offset + idx,
                n_samples=args.global_optimum_samples,
            )

            row: dict[str, Any] = {
                "problem": p_name,
                "benchmark": benchmark,
                "problem_index": idx,
                "problem_kwargs": problem_kwargs,
                "global_optimum": {
                    "source": global_optimum_source,
                    "params": global_optimum_params,
                    "value": float(global_optimum_value),
                },
                "options": [],
            }

            for option in args.hebo_options:
                model_name = _HEBO_UPDATED_MODEL_NAME_BY_SAMPLER[option]
                sampler_seed = args.seed + args.problem_offset + idx
                sampler_kwargs: dict[str, Any] = {
                    "search_space": problem.search_space,
                    "seed": sampler_seed,
                    "num_obj": 1,
                    "rand_sample": args.rand_sample,
                    "track_surrogate_predictions": True,
                }
                if model_name is not None:
                    sampler_kwargs["model_name"] = model_name
                sampler = hebo_updated_module.HEBOSampler(
                    **sampler_kwargs,
                )
                study = optuna.create_study(direction=problem.directions[0], sampler=sampler)

                records: list[dict[str, Any]] = []
                callback = _build_callback(
                    problem_name=p_name,
                    option_name=option,
                    records=records,
                    global_optimum_params=global_optimum_params,
                    global_optimum_value=global_optimum_value,
                    search_space=problem.search_space,
                    enabled=args.print_step_history,
                )
                with _maybe_suppress_sampler_output(args.sampler_output == "hide"):
                    study.optimize(
                        lambda trial: _objective(problem, trial),
                        n_trials=args.trials,
                        callbacks=[callback],
                    )

                metrics = _compute_metrics(records)
                aggregate_records[option].extend(records)
                aggregate_metric_rows[option].append(metrics)
                row["options"].append(
                    {
                        "name": option,
                        "model_name": model_name,
                        "best_value": float(study.best_value),
                        "metrics": metrics,
                    }
                )
                logger.info(
                    "  option=%s model=%s pred_points=%d mae=%s rmse=%s",
                    option,
                    model_name,
                    metrics.get("n_with_prediction", 0),
                    metrics.get("mae"),
                    metrics.get("rmse"),
                )

            details.append(row)

    summary = {
        "config": {
            "benchmarks": args.benchmarks,
            "n_problems": args.n_problems,
            "problem_offset": args.problem_offset,
            "trials": args.trials,
            "dim": args.dim,
            "data_type": args.data_type,
            "seed": args.seed,
            "instance_id": args.instance_id,
            "num_objectives": args.num_objectives,
            "hpolib_dataset_id": args.hpolib_dataset_id,
            "hebo_options": args.hebo_options,
            "rand_sample": args.rand_sample,
            "global_optimum_samples": args.global_optimum_samples,
        },
        "aggregate_average": {
            option: _average_metrics(metric_rows)
            for option, metric_rows in aggregate_metric_rows.items()
        },
        "aggregate_pooled": {
            option: _compute_metrics(records) for option, records in aggregate_records.items()
        },
        "details": details,
    }
    args.output_file.write_text(json.dumps(summary, indent=2))
    logger.info("Saved surrogate accuracy summary to %s", args.output_file)


if __name__ == "__main__":
    main()
