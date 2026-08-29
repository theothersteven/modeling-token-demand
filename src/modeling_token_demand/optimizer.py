"""Numerical solution of the user's policy problem.

The retry cap is discrete, so the optimizer enumerates it exactly.  For each
retry cap it searches over delegation horizon and inference intensity in log
space.  A coarse grid supplies several starting points to a bounded local
optimizer.  This is deterministic and robust enough for comparative statics
without coupling the economic model to one particular solver.
"""

from dataclasses import dataclass
import math
from typing import Iterable, Optional

import numpy as np
from scipy.optimize import minimize

from .model import IndustryModel, Policy, PolicyOutcome, Scenario


@dataclass(frozen=True)
class OptimizationSettings:
    """Bounds and numerical effort used to solve the policy problem.

    The economic optimum is conditional on these bounds.  They should describe
    the range of actions genuinely available to a user, not merely convenient
    numerical cutoffs.
    """

    # Smallest and largest amount of work that can be delegated per checkpoint.
    min_delegation_hours: float = 0.02
    max_delegation_hours: float = 80.0

    # Smallest and largest physical token budget per work-hour, per attempt.
    min_tokens_per_work_hour: float = 2_000.0
    max_tokens_per_work_hour: float = 1_200_000.0

    # Largest retry cap the user can select. Every integer from 1 is evaluated.
    max_attempts: int = 8

    # Grid resolution and number of promising grid points refined for each k.
    grid_points_per_dimension: int = 15
    local_starts_per_attempt: int = 3

    def __post_init__(self) -> None:
        if self.min_delegation_hours <= 0:
            raise ValueError("min_delegation_hours must be positive")
        if self.max_delegation_hours <= self.min_delegation_hours:
            raise ValueError("max_delegation_hours must exceed its minimum")
        if self.min_tokens_per_work_hour <= 0:
            raise ValueError("min_tokens_per_work_hour must be positive")
        if self.max_tokens_per_work_hour <= self.min_tokens_per_work_hour:
            raise ValueError("max_tokens_per_work_hour must exceed its minimum")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        if self.grid_points_per_dimension < 2:
            raise ValueError("grid_points_per_dimension must be at least two")
        if self.local_starts_per_attempt < 1:
            raise ValueError("local_starts_per_attempt must be at least one")


class PolicyOptimizer:
    """Solve for the surplus-maximizing user policy in a given scenario."""

    def __init__(self, settings: Optional[OptimizationSettings] = None):
        self.settings = settings or OptimizationSettings()

    def solve(self, model: IndustryModel, scenario: Scenario) -> PolicyOutcome:
        """Return the best policy found over continuous (s, x) and integer k.

        Users maximize expected surplus per work-hour. Adoption and aggregate
        token demand are consequences of that choice; they are not themselves
        part of the user's objective.
        """

        settings = self.settings
        log_s_bounds = (
            math.log(settings.min_delegation_hours),
            math.log(settings.max_delegation_hours),
        )
        log_x_bounds = (
            math.log(settings.min_tokens_per_work_hour),
            math.log(settings.max_tokens_per_work_hour),
        )
        log_s_grid = np.linspace(*log_s_bounds, settings.grid_points_per_dimension)
        log_x_grid = np.linspace(*log_x_bounds, settings.grid_points_per_dimension)

        best: Optional[PolicyOutcome] = None

        for max_attempts in range(1, settings.max_attempts + 1):
            candidates = []
            for log_s in log_s_grid:
                for log_x in log_x_grid:
                    outcome = self._evaluate_logs(
                        model, scenario, log_s, log_x, max_attempts
                    )
                    candidates.append(outcome)
                    best = self._better(best, outcome)

            candidates.sort(key=lambda item: item.surplus_per_work_hour, reverse=True)
            for start in candidates[: settings.local_starts_per_attempt]:
                initial = np.log(
                    [
                        start.policy.delegation_hours,
                        start.policy.tokens_per_work_hour,
                    ]
                )
                result = minimize(
                    self._negative_surplus,
                    initial,
                    args=(model, scenario, max_attempts),
                    method="L-BFGS-B",
                    bounds=(log_s_bounds, log_x_bounds),
                )
                # L-BFGS-B can report a line-search warning even when its last
                # feasible point is useful, so always evaluate the returned point.
                outcome = self._evaluate_logs(
                    model, scenario, result.x[0], result.x[1], max_attempts
                )
                best = self._better(best, outcome)

        if best is None:  # Defensive: validation ensures at least one k exists.
            raise RuntimeError("optimizer evaluated no candidate policies")
        return best

    def solve_many(
        self,
        model: IndustryModel,
        scenarios: Iterable[Scenario],
    ) -> list[PolicyOutcome]:
        """Solve a sequence of scenarios in the order supplied."""

        return [self.solve(model, scenario) for scenario in scenarios]

    @staticmethod
    def _better(
        current: Optional[PolicyOutcome], candidate: PolicyOutcome
    ) -> PolicyOutcome:
        if current is None:
            return candidate
        if candidate.surplus_per_work_hour > current.surplus_per_work_hour:
            return candidate
        return current

    @staticmethod
    def _evaluate_logs(
        model: IndustryModel,
        scenario: Scenario,
        log_s: float,
        log_x: float,
        max_attempts: int,
    ) -> PolicyOutcome:
        policy = Policy(
            delegation_hours=math.exp(float(log_s)),
            tokens_per_work_hour=math.exp(float(log_x)),
            max_attempts=max_attempts,
        )
        return model.evaluate(policy, scenario)

    @classmethod
    def _negative_surplus(
        cls,
        logs: np.ndarray,
        model: IndustryModel,
        scenario: Scenario,
        max_attempts: int,
    ) -> float:
        outcome = cls._evaluate_logs(
            model, scenario, logs[0], logs[1], max_attempts
        )
        return -outcome.surplus_per_work_hour
