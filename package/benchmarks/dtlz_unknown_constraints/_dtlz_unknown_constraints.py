from __future__ import annotations

import math
from typing import Any

import optuna
import optunahub


try:
    import optproblems.dtlz
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Please run `pip install optproblems diversipy` to use `dtlz_unknown_constraints`."
    )


class Problem(optunahub.benchmarks.BaseProblem):
    """Unknown-constraint variant of C-DTLZ.

    Constraint values are not exposed to optimizers.
    Infeasible points are returned as NaN objective values.
    """

    available_combinations = [
        {"constraint_type": 1, "function_id": 1},  # C1-DTLZ1
        {"constraint_type": 1, "function_id": 3},  # C1-DTLZ3
        {"constraint_type": 2, "function_id": 2},  # C2-DTLZ2
        {"constraint_type": 3, "function_id": 1},  # C3-DTLZ1
        {"constraint_type": 3, "function_id": 4},  # C3-DTLZ4
    ]

    def __init__(
        self,
        function_id: int,
        n_objectives: int,
        constraint_type: int,
        dimension: int | None = None,
        **kwargs: Any,
    ) -> None:
        assert 1 <= function_id <= 4, "function_id must be in [1, 4]"
        if dimension is None:
            dimension = n_objectives + (4 if function_id in [1, 4] else 9)
        self._dtlz_type = {"constraint_type": constraint_type, "function_id": function_id}

        assert (
            self._dtlz_type in self.available_combinations
        ), f"Invalid combination of constraint_type and function_id: {self._dtlz_type}. Available combinations are: {self.available_combinations}"
        self._problem = optproblems.dtlz.DTLZ(n_objectives, dimension, **kwargs)[function_id - 1]

        self._search_space = {
            f"x{i}": optuna.distributions.FloatDistribution(
                self._problem.min_bounds[i], self._problem.max_bounds[i]
            )
            for i in range(dimension)
        }

    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        return self._search_space.copy()

    @property
    def directions(self) -> list[optuna.study.StudyDirection]:
        direction = (
            optuna.study.StudyDirection.MAXIMIZE
            if self._problem.do_maximize
            else optuna.study.StudyDirection.MINIMIZE
        )
        return [direction] * self._problem.num_objectives

    def evaluate(self, params: dict[str, float]) -> list[float]:
        objective_values = self._problem.objective_function(
            [params[name] for name in self._search_space]
        )
        constraints = self._evaluate_constraints_from_objectives(objective_values)
        if any(c > 0.0 for c in constraints):
            return [float("nan")] * len(objective_values)
        return [float(v) for v in objective_values]

    def _evaluate_constraints_from_objectives(self, objective_values: list[float]) -> list[float]:
        if self._dtlz_type == {"constraint_type": 1, "function_id": 1}:
            return [objective_values[-1] / 0.6 + sum(objective_values[:-1]) / 0.5 - 1.0]
        if self._dtlz_type == {"constraint_type": 1, "function_id": 3}:
            sum_squares = sum(x**2 for x in objective_values)
            r = 9.0 if (m := len(objective_values)) < 5 else 12.5 if m < 10 else 15.0
            return [-(sum_squares - 16) * (sum_squares - r**2)]
        if self._dtlz_type == {"constraint_type": 2, "function_id": 2}:
            sum_squares = sum(x**2 for x in objective_values)
            m = len(objective_values)
            r = 0.3 if m == 3 else 0.5
            return [
                -max(
                    sum_squares - r**2 + 1.0 - 2.0 * max(objective_values),
                    sum((v - 1 / math.sqrt(m)) ** 2 for v in objective_values) - r**2,
                )
            ]
        if self._dtlz_type == {"constraint_type": 3, "function_id": 1}:
            sum_values = sum(objective_values)
            return [-sum_values - v + 1.0 for v in objective_values]
        if self._dtlz_type == {"constraint_type": 3, "function_id": 4}:
            squares = [x**2 for x in objective_values]
            return [-sum(squares) + v * 0.75 + 1.0 for v in objective_values]
        raise ValueError(f"Unsupported DTLZ type: {self._dtlz_type}")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._problem, name)
