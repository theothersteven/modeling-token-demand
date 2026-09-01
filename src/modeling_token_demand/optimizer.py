"""Numerical solution of the user's policy problem.

The optimizer searches over delegation horizon and model effort level in log
space. A coarse grid supplies several starting points to a bounded local
optimizer.  This is deterministic and robust enough for comparative statics
without coupling the economic model to one particular solver.
"""

from dataclasses import dataclass, replace
import math
from typing import Iterable, Optional

import numpy as np
from scipy.optimize import brentq, minimize

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

    # Smallest and largest normalized model effort level per work-hour.
    min_tokens_per_work_hour: float = 0.02
    max_tokens_per_work_hour: float = 20.0

    # Grid resolution and number of promising grid points refined locally.
    grid_points_per_dimension: int = 15
    local_starts: int = 3

    def __post_init__(self) -> None:
        if self.min_delegation_hours <= 0:
            raise ValueError("min_delegation_hours must be positive")
        if self.max_delegation_hours <= self.min_delegation_hours:
            raise ValueError("max_delegation_hours must exceed its minimum")
        if self.min_tokens_per_work_hour <= 0:
            raise ValueError("min_tokens_per_work_hour must be positive")
        if self.max_tokens_per_work_hour <= self.min_tokens_per_work_hour:
            raise ValueError("max_tokens_per_work_hour must exceed its minimum")
        if self.grid_points_per_dimension < 2:
            raise ValueError("grid_points_per_dimension must be at least two")
        if self.local_starts < 1:
            raise ValueError("local_starts must be at least one")


@dataclass(frozen=True)
class ReservationPriceResult:
    """Token price that leaves the optimized user at a target value."""

    token_price: float
    outcome: PolicyOutcome
    objective_gap: float
    iterations: int
    function_calls: int


class PolicyOptimizer:
    """Solve for the surplus-maximizing user policy in a given scenario."""

    def __init__(self, settings: Optional[OptimizationSettings] = None):
        self.settings = settings or OptimizationSettings()

    def solve(self, model: IndustryModel, scenario: Scenario) -> PolicyOutcome:
        """Return the best policy found over delegation and inference (s, x).

        The objective is supplied by ``objective_value``. The base optimizer
        maximizes expected surplus per work-hour; subclasses can retain the
        same numerical search while representing a different economic regime.
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

        candidates = []
        best: Optional[PolicyOutcome] = None
        for log_s in log_s_grid:
            for log_x in log_x_grid:
                outcome = self._evaluate_logs(
                    model, scenario, log_s, log_x
                )
                candidates.append(outcome)
                best = self._better(best, outcome)

        candidates.sort(key=self.objective_value, reverse=True)
        for start in candidates[: settings.local_starts]:
            initial = np.log(
                [start.policy.delegation_hours, start.policy.tokens_per_work_hour]
            )
            result = minimize(
                self._negative_objective,
                initial,
                args=(model, scenario),
                method="L-BFGS-B",
                bounds=(log_s_bounds, log_x_bounds),
            )
            # A line-search warning can still leave a useful feasible point.
            outcome = self._evaluate_logs(
                model, scenario, result.x[0], result.x[1]
            )
            best = self._better(best, outcome)

        if best is None:  # Defensive: the grid is nonempty.
            raise RuntimeError("optimizer evaluated no candidate policies")
        return best

    def solve_many(
        self,
        model: IndustryModel,
        scenarios: Iterable[Scenario],
    ) -> list[PolicyOutcome]:
        """Solve a sequence of scenarios in the order supplied."""

        return [self.solve(model, scenario) for scenario in scenarios]

    def objective_value(self, outcome: PolicyOutcome) -> float:
        """Economic objective maximized by this optimizer."""

        return outcome.surplus_per_work_hour

    def _better(
        self,
        current: Optional[PolicyOutcome], candidate: PolicyOutcome
    ) -> PolicyOutcome:
        if current is None:
            return candidate
        if self.objective_value(candidate) > self.objective_value(current):
            return candidate
        return current

    @staticmethod
    def _evaluate_logs(
        model: IndustryModel,
        scenario: Scenario,
        log_s: float,
        log_x: float,
    ) -> PolicyOutcome:
        policy = Policy(
            delegation_hours=math.exp(float(log_s)),
            tokens_per_work_hour=math.exp(float(log_x)),
        )
        return model.evaluate(policy, scenario)

    def _negative_objective(
        self,
        logs: np.ndarray,
        model: IndustryModel,
        scenario: Scenario,
    ) -> float:
        outcome = self._evaluate_logs(
            model, scenario, logs[0], logs[1]
        )
        return -self.objective_value(outcome)


class AttentionConstrainedOptimizer(PolicyOptimizer):
    """Optimize policy when useful work is abundant and attention is scarce.

    For a policy ``(s, x)``, one chunk creates expected net surplus
    ``s * surplus_per_work_hour`` and consumes ``h(s)`` human
    verification hours. The optimizer therefore maximizes their ratio. The
    size of the attention endowment scales throughput and token demand but does
    not affect the optimal policy in this single-industry polar case.
    """

    def objective_value(self, outcome: PolicyOutcome) -> float:
        """Return expected net surplus per human verification hour."""

        return outcome.surplus_per_attention_hour

    def solve_reservation_price(
        self,
        model: IndustryModel,
        scenario: Scenario,
        target_surplus_per_attention_hour: float,
        *,
        log_price_tolerance: float = 1e-6,
        max_bracket_steps: int = 48,
    ) -> ReservationPriceResult:
        """Find the token price that delivers a target optimized user value.

        Every price evaluation reoptimizes delegation scope and inference
        effort with :meth:`solve_interior`. The root is solved in log price,
        which preserves positivity and gives a relative rather than absolute
        price tolerance. Starting at ``scenario.token_price``, the method
        expands a factor-of-two bracket in whichever direction is required.
        """

        if not math.isfinite(target_surplus_per_attention_hour):
            raise ValueError("target surplus must be finite")
        if log_price_tolerance <= 0:
            raise ValueError("log_price_tolerance must be positive")
        if max_bracket_steps < 1:
            raise ValueError("max_bracket_steps must be positive")

        evaluations: dict[float, PolicyOutcome] = {}

        def gap(log_price: float) -> float:
            price = math.exp(float(log_price))
            outcome = self.solve_interior(
                model, replace(scenario, token_price=price)
            )
            evaluations[float(log_price)] = outcome
            return (
                outcome.surplus_per_attention_hour
                - target_surplus_per_attention_hour
            )

        log_initial = math.log(scenario.token_price)
        initial_gap = gap(log_initial)
        value_tolerance = 1e-10 * max(
            1.0, abs(target_surplus_per_attention_hour)
        )
        if abs(initial_gap) <= value_tolerance:
            outcome = evaluations[log_initial]
            return ReservationPriceResult(
                token_price=scenario.token_price,
                outcome=outcome,
                objective_gap=initial_gap,
                iterations=0,
                function_calls=1,
            )

        step = math.log(2.0)
        if initial_gap > 0:
            lower, upper = log_initial, log_initial + step
            for _ in range(max_bracket_steps):
                if gap(upper) <= 0:
                    break
                upper += step
            else:
                raise ValueError("could not bracket a finite reservation price")
        else:
            lower, upper = log_initial - step, log_initial
            for _ in range(max_bracket_steps):
                if gap(lower) >= 0:
                    break
                lower -= step
            else:
                raise ValueError(
                    "target value is unattainable at a positive token price"
                )

        root, details = brentq(
            gap,
            lower,
            upper,
            xtol=log_price_tolerance,
            rtol=4 * np.finfo(float).eps,
            maxiter=64,
            full_output=True,
        )
        final_gap = gap(root)
        outcome = evaluations[root]
        return ReservationPriceResult(
            token_price=math.exp(root),
            outcome=outcome,
            objective_gap=final_gap,
            iterations=details.iterations,
            function_calls=details.function_calls + 1,
        )

    def solve_interior(
        self, model: IndustryModel, scenario: Scenario
    ) -> PolicyOutcome:
        """Solve the exact scalar first-order equation to numerical tolerance.

        With positive fixed verification overhead and verification elasticity
        in [0, 1], the unconstrained attention optimum is unique in (s, x),
        with one review per chunk. Eliminating x from the first-order
        conditions leaves a strictly increasing equation in log(s). This
        gives an independent check on the general grid/local-search solver.

        The reduction assumes an interior policy. Reject unsupported
        verification technologies and optima outside the configured bounds;
        callers should use solve() for those cases. Like solve(), this returns
        a policy conditional on operating, not a participation decision.
        """

        p = model.industry
        if p.verification_fixed_hours <= 0 or p.verification_elasticity > 1:
            raise ValueError(
                "Interior reduction requires positive fixed verification "
                "overhead and verification elasticity at most one; use solve()"
            )

        alpha = p.inference_returns
        nu = p.capability_shape
        log_capability_horizon = math.log(
            p.capability_horizon_hours * scenario.model_capability
        )
        log_execution_scale = math.log(
            p.execution_scale * scenario.model_capability
        )
        cost_constant = alpha * math.log(
            scenario.token_price
            / (alpha * p.value_per_work_hour * scenario.token_efficiency)
        )

        def exponents(log_s: float) -> tuple[float, float]:
            # A = -log(q); B = -log(r), implied by the horizon FOC after
            # substituting the inference FOC cx = alpha * B * b * q * r.
            capability_exponent = math.exp(nu * (log_s - log_capability_horizon))
            s = math.exp(log_s)
            review_growth = math.exp(
                p.verification_elasticity * math.log1p(s)
            )
            variable_review = p.verification_scale * (review_growth - 1.0)
            review_hours = p.verification_fixed_hours + variable_review
            review_elasticity = (
                p.verification_elasticity
                * p.verification_scale
                * s
                * review_growth
                / ((1.0 + s) * review_hours)
            )
            delta = 1.0 - review_elasticity
            execution_exponent = (
                (delta - nu * capability_exponent) / (1.0 + alpha * delta)
            )
            return capability_exponent, execution_exponent

        def residual(log_s: float) -> float:
            capability_exponent, execution_exponent = exponents(log_s)
            if execution_exponent <= 0:
                return math.inf
            return (
                cost_constant + log_s - log_execution_scale
                + alpha * capability_exponent
                - (alpha + 1.0) * math.log(execution_exponent)
                + alpha * execution_exponent
            )

        # At nu*A = 1, B <= 0. Toward s=0, the residual tends to -infinity.
        upper = log_capability_horizon - math.log(nu) / nu
        lower = upper - 8.0
        while residual(lower) >= 0:
            lower -= 8.0
        log_s = brentq(residual, lower, upper, xtol=1e-12, rtol=1e-12)
        _, execution_exponent = exponents(log_s)
        log_x = -math.log(scenario.token_efficiency) + (
            log_s - log_execution_scale - math.log(execution_exponent)
        ) / alpha
        policy = Policy(
            delegation_hours=math.exp(log_s),
            tokens_per_work_hour=math.exp(log_x),
        )
        settings = self.settings
        if not (
            settings.min_delegation_hours <= policy.delegation_hours
            <= settings.max_delegation_hours
            and settings.min_tokens_per_work_hour <= policy.tokens_per_work_hour
            <= settings.max_tokens_per_work_hour
        ):
            raise ValueError(
                "Interior optimum lies outside configured policy bounds; use solve()"
            )
        return model.evaluate(policy, scenario)
