"""Run HEBO methods on selected synthetic functions and persist trajectories to pickle."""
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

import argparse
import concurrent.futures
import contextlib
import io
import json
import logging
import signal
from pathlib import Path
import time
from typing import Any

import optuna
import optunahub
import pandas as pd
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optunahub import hub as _hub
from tqdm import tqdm

logger = logging.getLogger(__name__)
# Enhanced logging format to include timestamps for long-running jobs
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)

_HEBO_UPDATED_VARIANTS = {
    "hebo_updated_gp",
    "hebo_updated_rf",
    "hebo_updated_deep_ensemble",
    "hebo_updated_catboost",
    "hebo_updated_psgld",
    "hebo_updated_svidkl",
}
_HEBO_UPDATED_MODEL_NAME_BY_SAMPLER = {
    "hebo_updated_gp": "gp",
    "hebo_updated_rf": "rf",
    "hebo_updated_deep_ensemble": "deep_ensemble",
    "hebo_updated_catboost": "catboost",
    "hebo_updated_psgld": "psgld",
    "hebo_updated_svidkl": "svidkl",
}
_SYNTHETIC_ALIAS_COMPONENTS = {"aug", "noise", "fail"}
_BINARY_CONSTRAINT_KEY = "_binary_constraint"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HEBO methods on selected synthetic benchmark functions."
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
        ],
        default="synthetic",
        help="Synthetic benchmark alias.",
    )
    parser.add_argument(
        "--function-ids",
        required=True,
        help="Comma-separated list/ranges of synthetic function IDs (used as problem seeds).",
    )
    parser.add_argument("--dim", type=int, default=10, help="Synthetic problem dimensionality.")
    parser.add_argument(
        "--data-type",
        choices=["Mixed", "Continuous"],
        default="Mixed",
        help="Synthetic search-space type.",
    )
    parser.add_argument(
        "--num-objectives",
        type=int,
        default=1,
        help="Number of synthetic objectives.",
    )
    parser.add_argument("--trials", type=int, default=25, help="Optuna trials per run.")
    parser.add_argument(
        "--samplers",
        nargs="+",
        default=[
            "hebo_updated_gp",
            "hebo_updated_rf",
            "hebo_updated_deep_ensemble",
            "hebo_updated_catboost",
            "hebo_updated_psgld",
            "hebo_updated_svidkl",
        ],
        help=(
            "HEBO methods to run: "
            "hebo, hebo_updated, hebo_updated_gp, hebo_updated_rf, "
            "hebo_updated_deep_ensemble, hebo_updated_catboost, "
            "hebo_updated_psgld, hebo_updated_svidkl."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "hebo_synthetic",
        help="Directory for output pickle files (one per sampler).",
    )
    parser.add_argument(
        "--trajectory-start",
        type=int,
        default=1,
        help="Inclusive first trial number to persist in each trajectory.",
    )
    parser.add_argument(
        "--trajectory-end",
        type=int,
        default=None,
        help="Inclusive final trial number to persist in each trajectory.",
    )
    parser.add_argument(
        "--extra-options",
        default="{}",
        help="JSON object merged into synthetic options.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of worker processes (1 disables multiprocessing).",
    )
    return parser.parse_args()


def _parse_function_ids(spec: str) -> list[int]:
    ids: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"Invalid range {part!r}: end < start")
            ids.extend(range(start, end + 1))
        else:
            ids.append(int(part))
    return ids


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


def _suggest_from_distribution(
    trial: optuna.trial.Trial, name: str, dist: BaseDistribution
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
    trial: optuna.trial.Trial, search_space: dict[str, BaseDistribution]
) -> None:
    for name, dist in search_space.items():
        _suggest_from_distribution(trial, name, dist)


def _instantiate_sampler(
    sampler_name: str,
    *,
    search_space: dict[str, BaseDistribution],
    seed: int,
    num_objectives: int,
    hebo_module: Any,
    hebo_updated_module: Any,
    use_binary_constraint: bool,
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

    if sampler_name == "hebo":
        return hebo_module.HEBOSampler(search_space=search_space, seed=seed, num_obj=num_objectives)
    if sampler_name in _HEBO_UPDATED_VARIANTS:
        model_name = _HEBO_UPDATED_MODEL_NAME_BY_SAMPLER[sampler_name]
        kwargs: dict[str, Any] = {
            "search_space": search_space,
            "seed": seed,
            "num_obj": num_objectives,
            "track_surrogate_predictions": True,
            "failure_constraint_model": use_binary_constraint,
            "constraints_func": _binary_constraints if use_binary_constraint else None,
        }
        if model_name is not None:
            kwargs["model_name"] = model_name
        return hebo_updated_module.HEBOSampler(**kwargs)
    raise ValueError(f"Unsupported sampler: {sampler_name}")


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_to_python_scalar(item) for item in value]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            converted = value.tolist()
        except Exception:
            converted = None
        if isinstance(converted, list):
            return [_to_python_scalar(item) for item in converted]
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _serialize_search_space(
    search_space: dict[str, BaseDistribution],
) -> dict[str, Any]:
    return {
        name: optuna.distributions.distribution_to_json(dist)
        for name, dist in search_space.items()
    }


def _build_trajectory_entries(
    study: optuna.Study,
    sampler: Any,
    *,
    start: int,
    end: int | None,
) -> list[dict[str, Any]]:
    history = None
    get_history = getattr(sampler, "get_surrogate_prediction_history", None)
    if callable(get_history):
        try:
            history = get_history()
        except Exception:
            history = None
    if history is not None and hasattr(history, "to_dict"):
        pred_by_trial: dict[int, dict[str, Any]] = {}
        for _, row in history.iterrows():
            trial_number = row.get("_trial_number")
            if trial_number is None:
                continue
            p_feasible_all = _to_python_scalar(row.get("_hebo_pred_p_feasible_all"))
            pred_by_trial[int(trial_number)] = {
                "predicted_mean": _to_python_scalar(row.get("_hebo_pred_mean")),
                "predicted_std": _to_python_scalar(row.get("_hebo_pred_std")),
                "predicted_failure_probability": (
                    None if p_feasible_all is None else _to_python_scalar(1.0 - p_feasible_all)
                ),
            }
    else:
        pred_by_trial = {}

    entries: list[dict[str, Any]] = []
    for trial in study.trials:
        if trial.number < start or (end is not None and trial.number > end):
            continue
        objective: Any = None
        if trial.values is not None:
            objective = _to_python_scalar(list(trial.values))
        elif trial.value is not None:
            objective = _to_python_scalar(trial.value)
        pred = pred_by_trial.get(trial.number, {})
        entries.append(
            {
                "trial_number": trial.number,
                "params": {k: _to_python_scalar(v) for k, v in trial.params.items()},
                "objective": objective,
                "trial_state": trial.state.name,
                "predicted_mean": pred.get("predicted_mean"),
                "predicted_std": pred.get("predicted_std"),
                "predicted_failure_probability": pred.get("predicted_failure_probability"),
                "failure": _to_python_scalar(trial.user_attrs.get("failure")),
                "initial_random": pred.get("predicted_mean") is None,
            }
        )
    return entries


def _build_output_path(
    output_dir: Path,
    benchmark: str,
    data_type: str,
    function_ids: list[int],
    dim: int,
    num_objectives: int,
    trials: int,
) -> Path:
    env_type = data_type.replace(" ", "_")
    lo = min(function_ids)
    hi = max(function_ids)
    filename = (
        f"hebo_samplers_{benchmark}_{env_type}_"
        f"fn{lo}_{hi}_dim{dim}_obj{num_objectives}_trials{trials}.pkl"
    )
    return output_dir / filename


def _run_single_function(
    *,
    function_id: int,
    args: argparse.Namespace,
    registry_root: Path,
    base_options: dict[str, Any],
) -> dict[str, Any]:
    synthetic_module = _load_module("benchmarks/synthetic", registry_root)
    hebo_module = None
    hebo_updated_module = None
    if "hebo" in args.samplers:
        hebo_module = _load_module("samplers/hebo", registry_root)
    if any(name in _HEBO_UPDATED_VARIANTS for name in args.samplers):
        hebo_updated_module = _load_module("samplers/hebo_updated", registry_root)

    problem = synthetic_module.Problem(
        dim=args.dim,
        seed=function_id,
        data_type=args.data_type,
        options=dict(base_options),
    )
    sampler_runs: dict[str, dict[str, Any]] = {}
    use_binary_constraint = bool(base_options.get("has_failure_regions", False))
    for sampler_name in args.samplers:
        sampler = _instantiate_sampler(
            sampler_name,
            search_space=problem.search_space,
            seed=function_id,
            num_objectives=problem.num_objectives,
            hebo_module=hebo_module,
            hebo_updated_module=hebo_updated_module,
            use_binary_constraint=use_binary_constraint,
        )

        def objective(trial: optuna.trial.Trial) -> float | tuple[float, ...]:
            _suggest_all_params(trial, problem.search_space)
            evaluate_with_failures = getattr(problem, "evaluate_with_failures", None)
            if callable(evaluate_with_failures):
                values, failure = evaluate_with_failures(trial.params)
                trial.set_user_attr("failure", bool(failure))
                trial.set_user_attr(_BINARY_CONSTRAINT_KEY, 1.0 if bool(failure) else -1.0)
                return values
            return problem.evaluate(trial.params)

        study = optuna.create_study(sampler=sampler, directions=problem.directions)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            study.optimize(objective, n_trials=args.trials)
        sampler_runs[sampler_name] = {
            "sampler_seed": function_id,
            "trajectory_entries": _build_trajectory_entries(
                study, sampler, start=args.trajectory_start, end=args.trajectory_end
            ),
        }

    return {
        "function_id": function_id,
        "problem_seed": function_id,
        "directions": [d.name for d in problem.directions],
        "samplers": sampler_runs,
        "failed": False,
    }


def _run_single_function_task(
    task: tuple[int, int, dict[str, Any], str, dict[str, Any]]
) -> tuple[int, dict[str, Any]]:
    """Worker task wrapper."""
    index, function_id, args_dict, registry_root_str, base_options = task
    # Silence Optuna inside workers to avoid console spam
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    
    args = argparse.Namespace(**args_dict)
    run = _run_single_function(
        function_id=function_id,
        args=args,
        registry_root=Path(registry_root_str),
        base_options=base_options,
    )
    return index, run


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
    processes: list[Any],
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
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    
    function_ids = _parse_function_ids(args.function_ids)
    if not function_ids:
        raise SystemExit("No valid function IDs were parsed.")
    try:
        extra_options = json.loads(args.extra_options)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Unable to parse --extra-options: {exc}")
    if not isinstance(extra_options, dict):
        raise SystemExit("--extra-options must be a JSON object.")

    registry_root = _resolve_registry_root()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    base_options: dict[str, Any] = {
        "num_objectives": args.num_objectives,
        "seed_tag": args.benchmark,
    }
    base_options.update(_synthetic_forced_dynamics_from_alias(args.benchmark))
    base_options.update(extra_options)

    runs_by_index: dict[int, dict[str, Any]] = {}
    tasks = [
        (idx, function_id, vars(args).copy(), str(registry_root), dict(base_options))
        for idx, function_id in enumerate(function_ids)
    ]
    
    worker_count = max(1, min(args.workers, len(tasks)))
    logger.info(f"Starting experiment with {len(tasks)} seeds using {worker_count} workers.")

    executor = None
    try:
        if worker_count == 1:
            # Sequential execution
            with tqdm(total=len(tasks), desc="Seeds", unit="seed") as pbar:
                for idx, function_id in enumerate(function_ids):
                    try:
                        runs_by_index[idx] = _run_single_function(
                            function_id=function_id,
                            args=args,
                            registry_root=registry_root,
                            base_options=base_options,
                        )
                    except KeyboardInterrupt:
                        raise
                    except Exception as exc:
                        logger.error(f"Seed {function_id} failed: {exc}")
                        runs_by_index[idx] = {
                            "function_id": function_id, 
                            "failed": True, 
                            "error": str(exc)
                        }
                    pbar.update(1)
        else:
            # Parallel execution
            # Manually manage executor to support immediate shutdown on Ctrl+C
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=worker_count)
            
            future_to_task = {
                executor.submit(_run_single_function_task, task): task 
                for task in tasks
            }
            
            with tqdm(total=len(future_to_task), desc="Seeds", unit="seed") as pbar:
                for future in concurrent.futures.as_completed(future_to_task):
                    task_data = future_to_task[future]
                    task_idx = task_data[0] # (idx, function_id, ...)
                    function_id = task_data[1]
                    
                    try:
                        idx, run = future.result()
                        runs_by_index[idx] = run
                    except KeyboardInterrupt:
                        raise
                    except Exception as exc:
                        logger.error(f"Seed {function_id} failed in worker: {exc}")
                        # Fill with failure placeholder to maintain list integrity
                        runs_by_index[task_idx] = {
                            "function_id": function_id,
                            "problem_seed": function_id,
                            "failed": True,
                            "error": str(exc),
                        }
                    
                    pbar.update(1)
            
            # Normal shutdown if loop completes successfully
            executor.shutdown(wait=True)

    except KeyboardInterrupt:
        logger.warning("\nReceived Ctrl+C. Stopping workers immediately...")
        _shutdown_executor(executor, cancel_futures=True, force_kill=True)
        if executor:
            logger.warning("Workers shut down. Exiting...")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Unexpected global error: {e}")
        _shutdown_executor(executor, cancel_futures=True, force_kill=True)
        sys.exit(1)

    # --- Save Results ---
    # Reconstruct list 0..N-1 to ensure data alignment with function_ids
    # If a run is missing for some reason, we insert a generic failure record.
    runs = []
    for idx in range(len(function_ids)):
        if idx in runs_by_index:
            runs.append(runs_by_index[idx])
        else:
            runs.append({
                "function_id": function_ids[idx],
                "failed": True,
                "error": "Missing result (likely interrupted)"
            })
    
    # Do not save if everything failed or list is empty
    if not runs:
        logger.warning("No runs completed. Nothing to save.")
        return

    output_path = _build_output_path(
        args.output_dir,
        args.benchmark,
        args.data_type,
        function_ids,
        args.dim,
        args.num_objectives,
        args.trials,
    )
    
    # Attempt to load metadata from the first successful run for schema consistency
    # If all failed, we construct a dummy object or empty dict
    space_json = {}
    try:
        # Find first non-failed run to grab metadata
        valid_run = next((r for r in runs if not r.get("failed")), None)
        if valid_run:
            seed_for_meta = valid_run["problem_seed"]
            synthetic_module = _load_module("benchmarks/synthetic", registry_root)
            problem_inst = synthetic_module.Problem(
                dim=args.dim,
                seed=seed_for_meta,
                data_type=args.data_type,
                options=dict(base_options),
            )
            space_json = _serialize_search_space(problem_inst.search_space)
    except Exception as e:
        logger.warning(f"Could not generate metadata schema: {e}")

    payload = {
        "metadata": {
            "benchmark": args.benchmark,
            "data_type": args.data_type,
            "dim": args.dim,
            "num_objectives": args.num_objectives,
            "function_ids": function_ids,
            "samplers": args.samplers,
            "trials": args.trials,
            "options": base_options,
            "search_space": space_json,
        },
        "runs": runs,
    }
    pd.to_pickle(payload, output_path)
    logger.info("Saved %s", output_path)


if __name__ == "__main__":
    main()
