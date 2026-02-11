from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any, Callable

import json
import numpy as np
import optuna
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.logging import get_logger
from optuna.samplers import BaseSampler
from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.search_space import IntersectionSearchSpace
from optuna.study import Study
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
import optunahub
import pandas as pd
import torch

from hebo.acquisitions.acq import Mean, Sigma
from hebo.design_space.design_space import DesignSpace
from hebo.models.base_model import BaseModel
from hebo.models.model_factory import get_model
from hebo.optimizers.general import GeneralBO
from hebo.optimizers.hebo import HEBO


_logger = get_logger(f"optuna.{__name__}")
_BINARY_CONSTRAINT_KEY = "_binary_constraint"


class HEBOSampler(optunahub.samplers.SimpleBaseSampler):
    """A sampler using `HEBO <https://github.com/huawei-noah/HEBO/tree/master/HEBO>__` as the backend.

    For further information about HEBO algorithm, please refer to the following paper:
    - `HEBO Pushing The Limits of Sample-Efficient Hyperparameter Optimisation <https://arxiv.org/abs/2012.03826>__`

    Args:
        search_space:
            By specifying search_space, the sampling speed at each iteration becomes slightly quicker, but this argument is not necessary to run this sampler. Default is :obj:`None`.

        seed:
            A seed for the initialization of ``HEBOSampler``. Default is :obj:`None`.
            Please note that the Bayesian optimization part is not deterministic even if seed is
            fixed due to the backend implementation.

        constant_liar:
            If :obj:`True`, penalize running trials to avoid suggesting parameter configurations
            nearby. Default is :obj:`False`.

            .. note::
                Abnormally terminated trials often leave behind a record with a state of
                ``RUNNING`` in the storage.
                Such "zombie" trial parameters will be avoided by the constant liar algorithm
                during subsequent sampling.
                When using an :class:`~optuna.storages.RDBStorage`, it is possible to enable the
                ``heartbeat_interval`` to change the records for abnormally terminated trials to
                ``FAIL``.
                (This note is quoted from `TPESampler <https://github.com/optuna/optuna/blob/v4.1.0/optuna/samplers/_tpe/sampler.py#L215-L222>__`.)

            .. note::
                It is recommended to set this value to :obj:`True` during distributed
                optimization to avoid having multiple workers evaluating similar parameter
                configurations. In particular, if each objective function evaluation is costly
                and the durations of the running states are significant, and/or the number of
                workers is high.
                (This note is quoted from `TPESampler <https://github.com/optuna/optuna/blob/v4.1.0/optuna/samplers/_tpe/sampler.py#L224-L229>__`.)

            .. note::
                HEBO algorithm involves multi-objective optimization of multiple acquisition functions.
                While `constant_liar` is a simple way to get diverse params for parallel optimization,
                it may not be the best approach for HEBO.

        independent_sampler:
            A :class:`~optuna.samplers.BaseSampler` instance that is used for independent
            sampling. The parameters not contained in the relative search space are sampled
            by this sampler. If :obj:`None` is specified, :class:`~optuna.samplers.RandomSampler`
            is used as the default.

        num_obj:
            If :obj:`>1`, use the multi-objective version of HEBO (GeneralBO) as the backend.
            Default is :obj:`1`.

        rand_sample:
            Number of initial random suggestions before HEBO relies on the surrogate model.
            If :obj:`None`, use HEBO's backend default.

        model_name:
            Override HEBO's surrogate backend model name (for example ``"gp"``,
            ``"rf"``, or ``"nn"``), if supported by the installed HEBO version.

        track_surrogate_predictions:
            If :obj:`True`, cache surrogate mean/std for proposed candidates
            in single-objective mode.

        failure_constraint_model:
            If :obj:`True`, enable constraint-aware backend mode.
            Constraint values are read from standard Optuna-integrated labels
            (``_CONSTRAINTS_KEY`` and ``trial.user_attrs['_binary_constraint']``).

        constraints_func:
            Optuna-style constraint callback. If provided, HEBO Updated will
            read constraints from this callback first, then fall back to trial attrs.
    """  # NOQA

    def __init__(
        self,
        search_space: dict[str, BaseDistribution] | None = None,
        *,
        seed: int | None = None,
        constant_liar: bool = False,
        independent_sampler: BaseSampler | None = None,
        num_obj: int = 1,
        rand_sample: int = 5,
        model_name: str | None = None,
        track_surrogate_predictions: bool = False,
        failure_constraint_model: bool = False,
        constraints_func: Callable[[FrozenTrial], Sequence[float]] | None = None,
    ) -> None:
        super().__init__(search_space, seed)
        if rand_sample is not None and rand_sample < 0:
            raise ValueError(f"`rand_sample` must be non-negative, but got {rand_sample}.")
        self._multi_objective = num_obj > 1
        self._constraints_func = constraints_func
        self._constraint_aware_mode = bool(
            failure_constraint_model or self._constraints_func is not None
        )
        self._rand_sample = rand_sample
        self._model_name = model_name
        self._multi_task_base_model_name: str | None = None
        use_internal_constraint_bo = self._constraint_aware_mode
        needs_multi_output_backend = self._multi_objective or use_internal_constraint_bo
        if needs_multi_output_backend and self._model_name is not None:
            if not self._supports_multi_output_model(self._model_name):
                # GeneralBO can still use single-output models through the multi_task wrapper.
                # See HEBO model_factory.MultiTaskModel(base_model_name=...).
                self._multi_task_base_model_name = self._model_name
                _logger.info(
                    "HEBO model_name=%r is single-output; using GeneralBO multi_task wrapper "
                    "with base_model_name=%r for multi-output optimization.",
                    self._model_name,
                    self._multi_task_base_model_name,
                )
        if search_space is not None and not constant_liar and not use_internal_constraint_bo:
            design_space = self._convert_to_hebo_design_space(search_space)
            if self._multi_objective:
                self._hebo = GeneralBO(design_space, num_obj=num_obj)
            else:
                self._hebo = HEBO(design_space, scramble_seed=seed)
            self._configure_hebo_backend(self._hebo)
        else:
            self._hebo = None
        self._intersection_search_space = IntersectionSearchSpace()
        self._independent_sampler = independent_sampler or optuna.samplers.RandomSampler(seed=seed)
        self._constant_liar = constant_liar
        self._rng = np.random.default_rng(seed)
        self._track_surrogate_predictions = track_surrogate_predictions
        self._last_surrogate_predictions: pd.DataFrame | None = None
        self._surrogate_prediction_history: list[pd.DataFrame] = []

    @staticmethod
    def _supports_multi_output_model(model_name: str) -> bool:
        # Probe model construction with a tiny shape to verify whether this HEBO model
        # supports num_out > 1 in the currently installed HEBO version.
        try:
            get_model(model_name, num_cont=1, num_enum=0, num_out=2)
            return True
        except Exception:
            return False

    def _configure_hebo_backend(self, hebo: HEBO | GeneralBO) -> None:
        if self._rand_sample is not None and hasattr(hebo, "rand_sample"):
            hebo.rand_sample = int(self._rand_sample)
        elif self._rand_sample is not None:
            _logger.warning(
                "HEBO backend does not expose `rand_sample`; warmup length cannot be overridden."
            )
        if self._model_name is not None and hasattr(hebo, "model_name"):
            if isinstance(hebo, GeneralBO) and self._multi_task_base_model_name is not None:
                hebo.model_name = "multi_task"
                if hasattr(hebo, "model_config"):
                    model_config = dict(getattr(hebo, "model_config", {}) or {})
                    model_config["base_model_name"] = self._multi_task_base_model_name
                    hebo.model_config = model_config
            else:
                hebo.model_name = self._model_name
        elif self._model_name is not None:
            _logger.warning(
                "HEBO backend does not expose `model_name`; surrogate model cannot be overridden."
            )

    def _suggest_and_transform_to_dict(
        self,
        hebo: HEBO | GeneralBO,
        search_space: dict[str, BaseDistribution],
        study: Study | None = None,
        trial_number: int | None = None,
    ) -> dict[str, Any]:
        # Keep the original HEBO behavior by default to avoid side effects from
        # extra surrogate-transform logic. Opt-in tracking is still available.
        try:
            suggestions = hebo.suggest()
        except Exception as e:
            _logger.warning(
                "HEBO suggest() failed (%s). Falling back to random space.sample(1).",
                e,
            )
            suggestions = hebo.space.sample(1)
        params = self._transform_suggestions_to_dict(suggestions, search_space)

        if self._track_surrogate_predictions and not self._multi_objective:
            real_predictions = self._convert_suggestions_to_real_space(suggestions, search_space)
            cached_predictions = self._update_prediction_cache(
                hebo, suggestions, real_predictions, study, search_space
            )
            self._record_prediction_history(cached_predictions, trial_number, search_space)

        return params

    @staticmethod
    def _transform_suggestions_to_dict(
        suggestions: pd.DataFrame, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for name, row in suggestions.items():
            if name not in search_space:
                continue

            dist = search_space[name]
            if (
                isinstance(dist, (IntDistribution, FloatDistribution))
                and not dist.log
                and dist.step is not None
            ):
                step_index = row.iloc[0]
                value = dist.low + step_index * dist.step
            else:
                value = row.iloc[0]

            if isinstance(value, np.generic):
                value = value.item()

            params[name] = value
        return params

    def get_last_surrogate_predictions(self) -> pd.DataFrame | None:
        """Return the last candidate proposals annotated with HEBO's mean/std."""
        return self._last_surrogate_predictions

    def get_surrogate_prediction_history(self) -> pd.DataFrame:
        """Return all cached surrogate predictions (params + mean/std) seen so far."""
        if not self._surrogate_prediction_history:
            return pd.DataFrame()
        return pd.concat(self._surrogate_prediction_history, ignore_index=True)

    def _update_prediction_cache(
        self,
        hebo: HEBO,
        suggestions: pd.DataFrame,
        real_predictions: pd.DataFrame,
        study: Study | None,
        search_space: dict[str, BaseDistribution],
    ) -> pd.DataFrame | None:
        try:
            stats = self._compute_prediction_stats(hebo, suggestions)
        except Exception as e:
            _logger.warning(
                "Failed to compute surrogate prediction stats (%s). Skipping prediction cache.",
                e,
            )
            self._last_surrogate_predictions = None
            return None
        if stats is None:
            self._last_surrogate_predictions = None
            return None

        mean, std = stats
        if study is not None and study.direction == StudyDirection.MAXIMIZE:
            # HEBO internally minimizes; convert predictions back to Optuna study scale.
            mean = -mean
        df = real_predictions.copy()
        df["_hebo_pred_mean"] = mean
        df["_hebo_pred_std"] = std
        try:
            constraint_stats = self._compute_constraint_prediction_stats(
                hebo=hebo,
                suggestions=suggestions,
                study=study,
                search_space=search_space,
            )
        except Exception as e:
            _logger.warning(
                "Failed to compute constraint prediction stats (%s). "
                "Continuing without constraint-model cache.",
                e,
            )
            constraint_stats = []
        if constraint_stats:
            all_prob = np.ones(len(df), dtype=float)
            for idx, (c_mean, c_std) in enumerate(constraint_stats):
                prob_feasible = self._probability_constraint_feasible(c_mean, c_std)
                df[f"_hebo_constraint_{idx}_pred_mean"] = c_mean
                df[f"_hebo_constraint_{idx}_pred_std"] = c_std
                df[f"_hebo_constraint_{idx}_pred_p_feasible"] = prob_feasible
                all_prob *= prob_feasible
            df["_hebo_pred_p_feasible_all"] = all_prob
        self._last_surrogate_predictions = df
        return df

    def _compute_constraint_prediction_stats(
        self,
        hebo: HEBO,
        suggestions: pd.DataFrame,
        study: Study | None,
        search_space: dict[str, BaseDistribution],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        if study is None or suggestions.empty:
            return []
        training = self._prepare_constraint_training_data(study, search_space, hebo)
        if training is None:
            return []
        X_train, Xe_train, constraint_vals = training
        min_train_size = max(getattr(hebo, "rand_sample", 1), 1)
        if X_train.shape[0] < min_train_size:
            return []

        transformed = hebo.space.transform(suggestions)
        X_cand = transformed[0]
        Xe_cand = transformed[1]
        if not self._is_valid_training_data(X_cand, Xe_cand):
            return []

        results: list[tuple[np.ndarray, np.ndarray]] = []
        n_constraints = constraint_vals.shape[1]
        for idx in range(n_constraints):
            y_tensor = torch.FloatTensor(constraint_vals[:, idx : idx + 1])
            if not self._is_valid_training_data(X_train, Xe_train, y_tensor):
                continue
            X_fit, Xe_fit, y_fit = self._filter_valid_training_data(X_train, Xe_train, y_tensor)
            if X_fit.shape[0] < min_train_size:
                continue
            model = get_model(
                hebo.model_name,
                hebo.space.num_numeric,
                hebo.space.num_categorical,
                1,
                **hebo.model_config,
            )
            model.fit(X_fit, Xe_fit, y_fit)
            mu = Mean(model)
            sig = Sigma(model, linear_a=-1.0)
            with torch.no_grad():
                c_mean = mu(X_cand, Xe_cand).squeeze().numpy()
                c_std = np.abs((-1 * sig(X_cand, Xe_cand).squeeze().numpy()))
            results.append((np.atleast_1d(c_mean), np.atleast_1d(c_std)))
        return results

    def _prepare_constraint_training_data(
        self,
        study: Study,
        search_space: dict[str, BaseDistribution],
        hebo: HEBO,
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray] | None:
        trials = study._get_trials(
            deepcopy=False, states=(TrialState.COMPLETE, TrialState.FAIL), use_cache=True
        )
        valid_trials: list[FrozenTrial] = []
        constraint_rows: list[list[float]] = []
        n_constraints: int | None = None
        for trial in trials:
            if not (set(search_space.keys()) <= set(trial.params.keys())):
                continue
            raw_constraints = self._get_standard_constraint_values(trial)
            if raw_constraints is None:
                continue
            row = [float(v) for v in raw_constraints]
            if n_constraints is None:
                n_constraints = len(row)
            if n_constraints != len(row):
                continue
            valid_trials.append(trial)
            constraint_rows.append(row)
        if not valid_trials or n_constraints is None or n_constraints == 0:
            return None

        params = self._trials_to_hebo_params_dataframe(valid_trials, search_space)
        if params.empty:
            return None
        X_train, Xe_train = hebo.space.transform(params)
        constraint_vals = np.asarray(constraint_rows, dtype=np.float32)
        if not np.isfinite(constraint_vals).all():
            return None
        return X_train, Xe_train, constraint_vals

    def _get_standard_constraint_values(self, trial: FrozenTrial) -> list[float] | None:
        if self._constraints_func is not None:
            try:
                raw_from_callback = self._constraints_func(trial)
                return [float(v) for v in raw_from_callback]
            except Exception:
                pass
        raw_constraints = trial.system_attrs.get(_CONSTRAINTS_KEY)
        if raw_constraints is not None:
            return [float(v) for v in raw_constraints]
        raw_binary = trial.user_attrs.get(_BINARY_CONSTRAINT_KEY)
        if raw_binary is None:
            return None
        try:
            return [float(raw_binary)]
        except Exception:
            return None

    @staticmethod
    def _trials_to_hebo_params_dataframe(
        trials: list[FrozenTrial],
        search_space: dict[str, BaseDistribution],
    ) -> pd.DataFrame:
        params = pd.DataFrame([t.params for t in trials])
        for name, dist in search_space.items():
            if (
                isinstance(dist, (IntDistribution, FloatDistribution))
                and not dist.log
                and dist.step is not None
            ):
                params[name] = (params[name] - dist.low) / dist.step
        return params

    @staticmethod
    def _probability_constraint_feasible(mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        safe_std = np.maximum(std, 1e-12)
        z = (0.0 - mean) / safe_std
        probs = 0.5 * (1.0 + np.vectorize(math.erf)(z / np.sqrt(2.0)))
        return np.asarray(probs, dtype=float)

    def _record_prediction_history(
        self,
        cached_predictions: pd.DataFrame | None,
        trial_number: int | None,
        search_space: dict[str, BaseDistribution] | None,
    ) -> None:
        if cached_predictions is None or cached_predictions.empty:
            return
        snapshot = cached_predictions.copy()
        if trial_number is not None:
            snapshot["_trial_number"] = trial_number
        if search_space is not None:
            snapshot["_search_space"] = self._serialize_search_space(search_space)
        self._surrogate_prediction_history.append(snapshot)

    def _compute_prediction_stats(
        self,
        hebo: HEBO,
        suggestions: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if suggestions.empty:
            return None

        model_result = self._build_surrogate_model(hebo)
        if model_result is None:
            return None

        model = model_result
        mu = Mean(model)
        sig = Sigma(model, linear_a=-1.0)
        with torch.no_grad():
            transformed = hebo.space.transform(suggestions)
            X_cand = transformed[0]
            Xe_cand = transformed[1]
            if not self._is_valid_training_data(X_cand, Xe_cand):
                return None
            py = mu(X_cand, Xe_cand).squeeze().numpy()
            ps = -1 * sig(X_cand, Xe_cand).squeeze().numpy()

        return py, np.abs(ps)

    def _build_surrogate_model(
        self, hebo: HEBO | GeneralBO
    ) -> BaseModel | None:
        min_train_size = max(getattr(hebo, "rand_sample", 1), 1)
        if hebo.X.shape[0] < min_train_size:
            return None

        X, Xe = hebo.space.transform(hebo.X)
        y_arr = np.asarray(hebo.y, dtype=np.float32)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        elif y_arr.ndim == 2 and y_arr.shape[1] > 1:
            # GeneralBO may contain objective + constraint columns.
            y_arr = y_arr[:, :1]
        if not np.isfinite(y_arr).all():
            return None
        y_tensor = torch.FloatTensor(y_arr)

        if not self._is_valid_training_data(X, Xe, y_tensor):
            return None
        X, Xe, y_tensor = self._filter_valid_training_data(X, Xe, y_tensor)
        if X.shape[0] < min_train_size:
            return None
        # print(hebo.model_name)
        model = get_model(
            hebo.model_name,
            hebo.space.num_numeric,
            hebo.space.num_categorical,
            1,
            **hebo.model_config,
        )
        model.fit(X, Xe, y_tensor)
        return model

    @staticmethod
    def _is_valid_training_data(
        X: torch.Tensor, Xe: torch.Tensor, y: torch.Tensor | None = None
    ) -> bool:
        if X.numel() == 0:
            return False
        if not torch.isfinite(X).all():
            return False
        if Xe.numel() > 0 and not torch.isfinite(Xe.float()).all():
            return False
        if y is not None and (y.numel() == 0 or not torch.isfinite(y).all()):
            return False
        return True

    @staticmethod
    def _filter_valid_training_data(
        X: torch.Tensor, Xe: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y_flat = y.view(-1)
        mask = torch.isfinite(X).all(dim=1)
        if Xe.numel() > 0:
            mask = mask & torch.isfinite(Xe.float()).all(dim=1)
        mask = mask & torch.isfinite(y_flat)
        return X[mask], Xe[mask], y[mask]

    def _transform_to_dict_and_observe(
        self,
        hebo: HEBO | GeneralBO,
        search_space: dict[str, BaseDistribution],
        study: Study,
        trials: list[FrozenTrial],
    ) -> None:
        expected = set(search_space.keys())
        for t in trials:
            if t.state != TrialState.COMPLETE:
                continue
            missing = expected - set(t.params.keys())
            if missing:
                raise ValueError(
                    "Incompatible search_space detected. "
                    "The following parameters are defined in the sampler's search_space "
                    f"but were missing from completed trial parameters: {missing}. "
                    "This indicates a mismatch between the sampler's search_space and "
                    "the parameters generated during optimization."
                )
        directions = study.directions if study._is_multi_objective() else [study.direction]
        sign = np.array([1 if d == StudyDirection.MINIMIZE else -1 for d in directions])

        def _trial_values_row(trial: FrozenTrial) -> list[float]:
            if trial.state == TrialState.COMPLETE:
                if trial.values is not None:
                    row_values = list(trial.values)
                elif trial.value is not None:
                    row_values = [trial.value]
                else:
                    row_values = [np.nan] * len(directions)
            else:
                row_values = [np.nan] * len(directions)
            if len(row_values) != len(directions):
                row_values = row_values[: len(directions)]
                if len(row_values) < len(directions):
                    row_values = row_values + [np.nan] * (len(directions) - len(row_values))
            return row_values

        values = np.array(
            [_trial_values_row(t) for t in trials],
            dtype=float,
        )

        worst_values = []
        for idx, direction in enumerate(directions):
            column = values[:, idx]
            worst_value = (
                np.nanmax(column) if direction == StudyDirection.MINIMIZE else np.nanmin(column)
            )
            worst_values.append(worst_value)
        worst_values = np.array(worst_values, dtype=float)

        for idx in range(len(directions)):
            nan_mask = np.isnan(values[:, idx])
            if nan_mask.any():
                values[nan_mask, idx] = worst_values[idx]

        # Assume that the back-end HEBO implementation aims to minimize.
        nan_padded_values = sign * values
        num_constr = int(getattr(hebo, "num_constr", 0) or 0)
        if num_constr > 0:
            constraint_values = self._extract_constraint_matrix(
                trials=trials,
                num_constr=num_constr,
            )
            nan_padded_values = np.concatenate([nan_padded_values, constraint_values], axis=1)
        params = pd.DataFrame([t.params for t in trials])
        for name, dist in search_space.items():
            if (
                isinstance(dist, (IntDistribution, FloatDistribution))
                and not dist.log
                and dist.step is not None
            ):
                # NOTE(nabenabe): We do not round here because HEBO treats params as float even if
                # the domain is defined on integer. By not rounding, HEBO can handle any changes in
                # the domain of these parameters such as changes in low, high, and step.
                params[name] = (params[name] - dist.low) / dist.step

        hebo.observe(params, nan_padded_values)

    def _extract_constraint_matrix(
        self,
        trials: list[FrozenTrial],
        num_constr: int,
    ) -> np.ndarray:
        rows: list[list[float]] = []
        for trial in trials:
            raw_constraints = self._get_standard_constraint_values(trial)
            if raw_constraints is None:
                rows.append([1.0] * num_constr)
                continue
            row = [float(v) for v in raw_constraints]
            if len(row) < num_constr:
                row = row + [1.0] * (num_constr - len(row))
            elif len(row) > num_constr:
                row = row[:num_constr]
            rows.append(row)
        constraints = np.asarray(rows, dtype=float)
        constraints[~np.isfinite(constraints)] = 1.0
        return constraints

    def _sample_relative_define_and_run(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        return self._suggest_and_transform_to_dict(
            self._hebo, search_space, study=study, trial_number=trial.number
        )

    def _sample_relative_stateless(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        if self._constant_liar:
            target_states = [TrialState.COMPLETE, TrialState.RUNNING]
        else:
            target_states = [TrialState.COMPLETE, TrialState.FAIL]

        use_cache = not self._constant_liar
        trials = study._get_trials(deepcopy=False, states=target_states, use_cache=use_cache)
        is_complete = np.array([t.state == TrialState.COMPLETE for t in trials])
        if not np.any(is_complete):
            # note: The backend HEBO implementation uses Sobol sampling here.
            # This sampler does not call `hebo.suggest()` here because
            # Optuna needs to know search space by running the first trial in Define-by-Run.
            return {}

        trials = [t for t in trials if set(search_space.keys()) <= set(t.params.keys())]
        num_constraints = self._infer_num_constraints_from_trials(trials)
        seed = int(self._rng.integers(low=1, high=(1 << 31)))
        design_space = self._convert_to_hebo_design_space(search_space)
        if self._multi_objective or num_constraints > 0:
            hebo = GeneralBO(
                design_space,
                num_obj=len(study.directions),
                num_constr=num_constraints,
            )
        else:
            hebo = HEBO(design_space, scramble_seed=seed)
        self._configure_hebo_backend(hebo)
        self._transform_to_dict_and_observe(
            hebo,
            search_space,
            study,
            trials,
        )
        return self._suggest_and_transform_to_dict(
            hebo, search_space, study=study, trial_number=trial.number
        )

    def _infer_num_constraints_from_trials(self, trials: list[FrozenTrial]) -> int:
        max_n_constraints = 0
        for trial in trials:
            row = self._get_standard_constraint_values(trial)
            if row is None:
                continue
            max_n_constraints = max(max_n_constraints, len(row))
        return max_n_constraints

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        if study._is_multi_objective() and not self._multi_objective:
            raise ValueError(
                f"To use {self.__class__.__name__} for multi-objective optimization, please specify the 'num_obj' parameter."
            )
        if self._hebo is None or self._constant_liar is True:
            return self._sample_relative_stateless(study, trial, search_space)
        else:
            return self._sample_relative_define_and_run(study, trial, search_space)

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        if self._hebo is not None and values is not None:
            # Note that trial.values is None and trial.state is RUNNNING here.
            trial_ = optuna.trial.create_trial(
                state=state,  # updated
                params=trial.params,
                distributions=trial.distributions,
                values=values,  # updated
            )
            self._transform_to_dict_and_observe(
                hebo=self._hebo,
                search_space=trial.distributions,
                study=study,
                trials=[trial_],
            )

    def _convert_to_hebo_design_space(
        self, search_space: dict[str, BaseDistribution]
    ) -> DesignSpace:
        design_space = []
        for name, distribution in search_space.items():
            config: dict[str, Any] = {"name": name}
            if isinstance(distribution, (FloatDistribution, IntDistribution)):
                if not distribution.log and distribution.step is not None:
                    config["type"] = "int"
                    # NOTE(nabenabe): high is adjusted in Optuna so that below is divisable.
                    n_steps = int(
                        np.round((distribution.high - distribution.low) / distribution.step + 1)
                    )
                    config["lb"] = 0
                    config["ub"] = n_steps - 1
                else:
                    config["lb"] = distribution.low
                    config["ub"] = distribution.high
                    if distribution.log:
                        config["type"] = (
                            "pow_int" if isinstance(distribution, IntDistribution) else "pow"
                        )
                    else:
                        assert not isinstance(distribution, IntDistribution)
                        config["type"] = "num"
            elif isinstance(distribution, CategoricalDistribution):
                config["type"] = "cat"
                config["categories"] = distribution.choices
            else:
                raise NotImplementedError(f"Unsupported distribution: {distribution}")

            design_space.append(config)

        return DesignSpace().parse(design_space)

    def _serialize_search_space(
        self, search_space: dict[str, BaseDistribution]
    ) -> str:
        serialized: dict[str, dict[str, Any]] = {}
        for name, distribution in search_space.items():
            data: dict[str, Any] = {"type": distribution.__class__.__name__}
            if isinstance(distribution, (FloatDistribution, IntDistribution)):
                data.update(
                    {
                        "low": distribution.low,
                        "high": distribution.high,
                        "step": distribution.step,
                        "log": distribution.log,
                    }
                )
            elif isinstance(distribution, CategoricalDistribution):
                data["choices"] = distribution.choices
            serialized[name] = data
        return json.dumps(serialized, sort_keys=True)

    def _convert_suggestions_to_real_space(
        self,
        suggestions: pd.DataFrame,
        search_space: dict[str, BaseDistribution],
    ) -> pd.DataFrame:
        if suggestions.empty:
            return pd.DataFrame(columns=search_space.keys())

        rows: list[dict[str, float]] = []
        for _, row in suggestions.iterrows():
            converted: dict[str, float] = {}
            for name, dist in search_space.items():
                if name not in row:
                    continue
                value = row[name]
                if (
                    isinstance(dist, (IntDistribution, FloatDistribution))
                    and not dist.log
                    and dist.step is not None
                ):
                    step_index = value
                    converted_value = dist.low + step_index * dist.step
                    converted[name] = int(converted_value) if isinstance(dist, IntDistribution) else converted_value
                else:
                    if isinstance(value, bool):
                        converted[name] = value
                    else:
                        converted[name] = (
                            float(value) if isinstance(value, (int, float)) else value
                        )
            rows.append(converted)
        return pd.DataFrame(rows)

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        if self.search_space is not None:
            return self.search_space
        return optuna.search_space.intersection_search_space(
            study._get_trials(deepcopy=False, use_cache=True)
        )

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        states = (TrialState.COMPLETE,)
        trials = study._get_trials(deepcopy=False, states=states, use_cache=True)
        if any(param_name in trial.params for trial in trials):
            _logger.warn(f"Use `RandomSampler` for {param_name} due to dynamic search space.")

        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )
