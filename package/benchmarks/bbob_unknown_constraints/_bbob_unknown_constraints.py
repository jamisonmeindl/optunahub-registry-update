from __future__ import annotations

from typing import Any

import optuna
import optunahub


try:
    import cocoex as ex
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Please run `pip install coco-experiment` to use `bbob_unknown_constraints`."
    )


class Problem(optunahub.benchmarks.BaseProblem):
    """Unknown-constraint variant of COCO bbob-constrained.

    Infeasible evaluations are treated as failed evaluations by returning NaN,
    while constraint values are not exposed to optimizers.
    """

    def __init__(self, function_id: int, dimension: int, instance_id: int = 1):
        self._valid_arguments = False
        assert 1 <= function_id <= 54, "function_id must be in [1, 54]"
        assert dimension in [2, 3, 5, 10, 20, 40], "dimension must be in [2, 3, 5, 10, 20, 40]"
        assert 1 <= instance_id <= 15, "instance_id must be in [1, 15]"
        self._valid_arguments = True

        self._problem = ex.Suite(
            "bbob-constrained", "", ""
        ).get_problem_by_function_dimension_instance(
            function=function_id, dimension=dimension, instance=instance_id
        )
        self._search_space = {
            f"x{i}": optuna.distributions.FloatDistribution(
                low=self._problem.lower_bounds[i],
                high=self._problem.upper_bounds[i],
            )
            for i in range(self._problem.dimension)
        }

    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        return self._search_space.copy()

    @property
    def directions(self) -> list[optuna.study.StudyDirection]:
        return [optuna.study.StudyDirection.MINIMIZE]

    def evaluate(self, params: dict[str, float]) -> float:
        x = [params[name] for name in self._search_space]
        constraint_values = self._problem.constraint(x)
        if any(c > 0.0 for c in constraint_values):
            return float("nan")
        return float(self._problem(x))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._problem, name)

    def __del__(self) -> None:
        if not self._valid_arguments:
            return

        self._problem.free()
