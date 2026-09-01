"""Core economic model.

The objects in this module contain no numerical optimization logic.  They map
an industry, a technology/cost scenario, and a candidate user policy into
reliability, surplus, adoption, and token demand.  Keeping this layer pure makes
it easy to swap numerical solvers or run comparative statics later.
"""

from dataclasses import dataclass
import math
from typing import Optional


@dataclass(frozen=True)
class Industry:
    """Parameters that remain fixed when an industry's scenario changes.

    Parameter names are descriptive versions of the paper's Greek symbols.
    Monetary quantities are dollars and time quantities are human-hours.
    """

    name: str

    # lambda: scope scale of capability loss at m=1.
    capability_horizon_hours: float
    # nu: how sharply the solvable share falls as delegation horizon grows.
    capability_shape: float

    # a: industry-specific ease of executing a task that is technically solvable.
    execution_scale: float
    # alpha: diminishing growth of the execution horizon in effective inference.
    inference_returns: float

    # h_0: fixed human time required at every checkpoint.
    verification_fixed_hours: float
    # h_1: scale of the human review time that grows with delegated scope.
    verification_scale: float
    # beta: curvature of the variable review component as scope grows.
    # The elasticity of total review time is computed at the chosen scope.
    verification_elasticity: float

    # b: value of successfully completing one human-hour-equivalent of work.
    value_per_work_hour: float
    # w: opportunity cost of one hour of human review time.
    human_cost_per_hour: float

    # mu: location of the logistic distribution of adoption hurdles.
    adoption_location: float
    # sigma: dispersion of per-unit adoption hurdles across industry work.
    adoption_scale: float

    # W: total potential industry work during the modeled period.
    potential_work_hours: float = 1_000_000.0
    # H: human review hours available. None means attention does not bind.
    human_attention_hours: Optional[float] = None

    def __post_init__(self) -> None:
        positive = {
            "capability_horizon_hours": self.capability_horizon_hours,
            "capability_shape": self.capability_shape,
            "execution_scale": self.execution_scale,
            "inference_returns": self.inference_returns,
            "verification_scale": self.verification_scale,
            "value_per_work_hour": self.value_per_work_hour,
            "human_cost_per_hour": self.human_cost_per_hour,
            "adoption_scale": self.adoption_scale,
            "potential_work_hours": self.potential_work_hours,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.inference_returns >= 1:
            raise ValueError("inference_returns must be below one")
        if self.verification_fixed_hours < 0:
            raise ValueError("verification_fixed_hours cannot be negative")
        if self.verification_elasticity < 0:
            raise ValueError("verification_elasticity cannot be negative")
        if self.verification_fixed_hours == 0 and self.verification_elasticity == 0:
            raise ValueError(
                "verification_fixed_hours and verification_elasticity cannot "
                "both be zero"
            )
        if self.human_attention_hours is not None and self.human_attention_hours <= 0:
            raise ValueError("human_attention_hours must be positive when supplied")


@dataclass(frozen=True)
class Scenario:
    """Technology and price variables varied in comparative statics.

    model_capability is m, token_efficiency is eta, and
    verification_time_multiplier is v in the paper.
    """

    model_capability: float = 1.0
    token_efficiency: float = 1.0
    # c: price of one normalized token unit. The reference scenario sets c=1.
    token_price: float = 1.0
    verification_time_multiplier: float = 1.0

    def __post_init__(self) -> None:
        for name, value in (
            ("model_capability", self.model_capability),
            ("token_efficiency", self.token_efficiency),
            ("token_price", self.token_price),
            ("verification_time_multiplier", self.verification_time_multiplier),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class Policy:
    """A user's policy for one delegated chunk and one review.

    delegation_hours (s) is work delegated before a checkpoint,
    and tokens_per_work_hour (x) is the model effort level in normalized token
    units. The normalization sets the minimum viable effort to one. Every
    chunk consumes sx token units and h(s) review hours, whether it succeeds
    or fails. Failed work produces no output in this model.
    """

    delegation_hours: float
    tokens_per_work_hour: float

    def __post_init__(self) -> None:
        if self.delegation_hours <= 0:
            raise ValueError("delegation_hours must be positive")
        if self.tokens_per_work_hour < 1:
            raise ValueError("tokens_per_work_hour must be at least one")


@dataclass(frozen=True)
class PolicyOutcome:
    """Economic and technical outcomes produced by one candidate policy."""

    policy: Policy
    capability_share: float
    conditional_success: float
    success_probability: float
    verification_hours_per_chunk: float
    cost_per_work_hour: float
    surplus_per_work_hour: float
    # Expected net surplus created per human verification hour. This is the
    # relevant policy objective when useful work is abundant and attention is
    # the binding resource.
    surplus_per_attention_hour: float
    adoption_share: float
    work_limited_tokens: float
    attention_limited_tokens: Optional[float]
    realized_tokens: float


class IndustryModel:
    """Pure evaluator for the economic model of one industry."""

    def __init__(self, industry: Industry):
        self.industry = industry

    def capability_share(self, policy: Policy, scenario: Scenario) -> float:
        """q: share of tasks at this horizon inside the capability frontier."""

        p = self.industry
        relative_horizon = (
            policy.delegation_hours
            / (p.capability_horizon_hours * scenario.model_capability)
        )
        return math.exp(-(relative_horizon ** p.capability_shape))

    def conditional_success(self, policy: Policy, scenario: Scenario) -> float:
        """r: single-attempt success, conditional on the task being solvable."""

        p = self.industry
        effective_inference = (
            scenario.token_efficiency * policy.tokens_per_work_hour
        )
        execution_horizon = (
            p.execution_scale
            * scenario.model_capability
            * effective_inference ** p.inference_returns
        )
        exponent = -policy.delegation_hours / execution_horizon
        # exp(-745) is near the smallest useful positive IEEE float.  Returning
        # zero below that point is numerically harmless and avoids underflow.
        return 0.0 if exponent < -745.0 else math.exp(exponent)

    def success_probability(self, policy: Policy, scenario: Scenario) -> float:
        """P = qr: probability that the delegated chunk is completed."""

        q = self.capability_share(policy, scenario)
        r = self.conditional_success(policy, scenario)
        return q * r

    def verification_hours(self, policy: Policy, scenario: Scenario) -> float:
        """Human review time required for every delegated chunk."""

        p = self.industry
        base_time = (
            p.verification_fixed_hours
            + p.verification_scale
            * math.expm1(
                p.verification_elasticity
                * math.log1p(policy.delegation_hours)
            )
        )
        return scenario.verification_time_multiplier * base_time

    def verification_scope_elasticity(self, policy: Policy) -> float:
        """Return d log h(s) / d log s for the shifted review function."""

        p = self.industry
        s = policy.delegation_hours
        growth = math.exp(p.verification_elasticity * math.log1p(s))
        review_hours = (
            p.verification_fixed_hours
            + p.verification_scale * (growth - 1.0)
        )
        return (
            p.verification_elasticity
            * p.verification_scale
            * s
            * growth
            / ((1.0 + s) * review_hours)
        )

    def adoption_share(self, surplus_per_work_hour: float) -> float:
        """A: work-weighted adoption under a logistic hurdle distribution.

        Hurdles may be negative, allowing adoption at negative measured surplus
        when users value being AI-forward or face an organizational mandate.
        """

        p = self.industry
        z = (surplus_per_work_hour - p.adoption_location) / p.adoption_scale
        # Stable logistic evaluation for very high or low adoption surplus.
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        exp_z = math.exp(z)
        return exp_z / (1.0 + exp_z)

    def evaluate(self, policy: Policy, scenario: Scenario) -> PolicyOutcome:
        """Evaluate reliability, surplus, adoption, and demand for a policy."""

        p = self.industry
        q = self.capability_share(policy, scenario)
        r = self.conditional_success(policy, scenario)
        success = q * r
        verification_hours = self.verification_hours(policy, scenario)

        token_cost = scenario.token_price * policy.tokens_per_work_hour
        review_cost = (
            p.human_cost_per_hour
            * verification_hours
            / policy.delegation_hours
        )
        # Neither cost is conditional on success: failed work consumes the
        # same tokens and review time as successful work at this policy.
        cost = token_cost + review_cost
        surplus = p.value_per_work_hour * success - cost
        surplus_per_attention_hour = (
            policy.delegation_hours
            * surplus
            / verification_hours
        )
        adoption = self.adoption_share(surplus)

        work_limited_tokens = (
            p.potential_work_hours * adoption * policy.tokens_per_work_hour
        )

        attention_limited_tokens: Optional[float]
        if p.human_attention_hours is None:
            attention_limited_tokens = None
            realized_tokens = work_limited_tokens
        else:
            # Throughput implied by this policy when the human-attention
            # constraint binds. Policy selection remains the optimizer's job.
            attention_limited_tokens = (
                p.human_attention_hours
                * policy.delegation_hours
                * policy.tokens_per_work_hour
                / verification_hours
            )
            realized_tokens = min(work_limited_tokens, attention_limited_tokens)

        return PolicyOutcome(
            policy=policy,
            capability_share=q,
            conditional_success=r,
            success_probability=success,
            verification_hours_per_chunk=verification_hours,
            cost_per_work_hour=cost,
            surplus_per_work_hour=surplus,
            surplus_per_attention_hour=surplus_per_attention_hour,
            adoption_share=adoption,
            work_limited_tokens=work_limited_tokens,
            attention_limited_tokens=attention_limited_tokens,
            realized_tokens=realized_tokens,
        )
