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

    # lambda: task horizon at which the capability frontier begins to bind.
    capability_horizon_hours: float
    # nu: how sharply the solvable share falls as delegation horizon grows.
    capability_shape: float

    # a: industry-specific ease of executing a task that is technically solvable.
    execution_scale: float
    # alpha: diminishing returns to inference intensity; normally between 0 and 1.
    inference_returns: float

    # h_0: fixed human time required at every checkpoint.
    verification_fixed_hours: float
    # h_1: scale of the human review time that grows with delegated scope.
    verification_scale: float
    # beta: elasticity of verification time with respect to delegated scope.
    verification_elasticity: float

    # b: value of successfully completing one human-hour-equivalent of work.
    value_per_work_hour: float
    # w: opportunity cost of one hour of human review time.
    human_cost_per_hour: float

    # mu: surplus per work-hour at which half of potential work adopts AI.
    adoption_midpoint: float
    # sigma: dispersion of per-unit adoption hurdles across industry work.
    adoption_scale: float

    # W: total potential industry work during the modeled period.
    potential_work_hours: float = 1_000_000.0
    # H: human review hours available. None means attention does not bind.
    human_attention_hours: Optional[float] = None

    # Numerical normalization used inside the execution-reliability function.
    # The policy variable itself always remains tokens per work-hour.
    token_reference: float = 100_000.0

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
            "token_reference": self.token_reference,
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
    token_price_per_million: float = 10.0
    verification_time_multiplier: float = 1.0

    def __post_init__(self) -> None:
        for name, value in (
            ("model_capability", self.model_capability),
            ("token_efficiency", self.token_efficiency),
            ("token_price_per_million", self.token_price_per_million),
            ("verification_time_multiplier", self.verification_time_multiplier),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def token_price(self) -> float:
        """Dollar price of one token."""

        return self.token_price_per_million / 1_000_000.0


@dataclass(frozen=True)
class Policy:
    """A user's interaction policy.

    delegation_hours (s) is work delegated before a checkpoint,
    tokens_per_work_hour (x) is inference intensity per attempt, and
    max_attempts (k) is the retry cap.
    """

    delegation_hours: float
    tokens_per_work_hour: float
    max_attempts: int

    def __post_init__(self) -> None:
        if self.delegation_hours <= 0:
            raise ValueError("delegation_hours must be positive")
        if self.tokens_per_work_hour <= 0:
            raise ValueError("tokens_per_work_hour must be positive")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least one")


@dataclass(frozen=True)
class PolicyOutcome:
    """Economic and technical outcomes produced by one candidate policy."""

    policy: Policy
    capability_share: float
    conditional_success: float
    eventual_success: float
    expected_attempts: float
    verification_hours_per_attempt: float
    expected_cost_per_work_hour: float
    surplus_per_work_hour: float
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
        normalized_tokens = policy.tokens_per_work_hour / p.token_reference
        effective_inference = scenario.token_efficiency * normalized_tokens
        execution_horizon = (
            p.execution_scale
            * scenario.model_capability
            * effective_inference ** p.inference_returns
        )
        exponent = -policy.delegation_hours / execution_horizon
        # exp(-745) is near the smallest useful positive IEEE float.  Returning
        # zero below that point is numerically harmless and avoids underflow.
        return 0.0 if exponent < -745.0 else math.exp(exponent)

    def eventual_success(self, policy: Policy, scenario: Scenario) -> float:
        """P: success after at most k attempts."""

        q = self.capability_share(policy, scenario)
        r = self.conditional_success(policy, scenario)
        return q * (1.0 - (1.0 - r) ** policy.max_attempts)

    def expected_attempts(self, policy: Policy, scenario: Scenario) -> float:
        """E: attempts consumed when the user stops after the first success."""

        q = self.capability_share(policy, scenario)
        r = self.conditional_success(policy, scenario)
        k = policy.max_attempts

        # This finite geometric sum is stable even when r underflows to zero.
        solvable_attempts = sum((1.0 - r) ** j for j in range(k))
        return (1.0 - q) * k + q * solvable_attempts

    def verification_hours(self, policy: Policy, scenario: Scenario) -> float:
        """Human review time required after one attempt."""

        p = self.industry
        base_time = (
            p.verification_fixed_hours
            + p.verification_scale
            * policy.delegation_hours ** p.verification_elasticity
        )
        return scenario.verification_time_multiplier * base_time

    def adoption_share(self, surplus_per_work_hour: float) -> float:
        """A: work-volume-weighted share of industry work delegated to AI."""

        p = self.industry
        z = (surplus_per_work_hour - p.adoption_midpoint) / p.adoption_scale
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
        success = q * (1.0 - (1.0 - r) ** policy.max_attempts)
        attempts = self.expected_attempts(policy, scenario)
        verification_hours = self.verification_hours(policy, scenario)

        token_cost = scenario.token_price * policy.tokens_per_work_hour
        review_cost = (
            p.human_cost_per_hour
            * verification_hours
            / policy.delegation_hours
        )
        expected_cost = attempts * (token_cost + review_cost)
        surplus = p.value_per_work_hour * success - expected_cost
        adoption = self.adoption_share(surplus)

        tokens_per_work_hour = policy.tokens_per_work_hour * attempts
        work_limited_tokens = (
            p.potential_work_hours * adoption * tokens_per_work_hour
        )

        attention_limited_tokens: Optional[float]
        if p.human_attention_hours is None:
            attention_limited_tokens = None
            realized_tokens = work_limited_tokens
        else:
            # Fixed-policy throughput when the human-attention constraint binds.
            # This aggregate cap does not feed a scarcity price back into the
            # user's objective; a full attention-market equilibrium would.
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
            eventual_success=success,
            expected_attempts=attempts,
            verification_hours_per_attempt=verification_hours,
            expected_cost_per_work_hour=expected_cost,
            surplus_per_work_hour=surplus,
            adoption_share=adoption,
            work_limited_tokens=work_limited_tokens,
            attention_limited_tokens=attention_limited_tokens,
            realized_tokens=realized_tokens,
        )
