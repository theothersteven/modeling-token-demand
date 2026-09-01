"""Economic model and numerical tools for studying AI token demand."""

from .model import (
    Industry,
    IndustryModel,
    Policy,
    PolicyOutcome,
    Scenario,
)
from .optimizer import (
    AttentionConstrainedOptimizer,
    OptimizationSettings,
    PolicyOptimizer,
    ReservationPriceResult,
)
from .calibrations import (
    HARD_EXECUTION,
    HIGH_ADOPTION_HURDLE,
    HIGH_CAPABILITY_REQUIREMENT,
    LOW_INFERENCE_RETURNS,
    LOW_ADOPTION_HURDLE,
    PROPORTIONAL_REVIEW,
    REFERENCE_INDUSTRY,
    SLOW_REVIEW_GROWTH,
    SUBLINEAR_VERIFICATION,
    illustrative_industries,
)

__all__ = [
    "AttentionConstrainedOptimizer",
    "HARD_EXECUTION",
    "HIGH_ADOPTION_HURDLE",
    "HIGH_CAPABILITY_REQUIREMENT",
    "Industry",
    "IndustryModel",
    "LOW_INFERENCE_RETURNS",
    "LOW_ADOPTION_HURDLE",
    "OptimizationSettings",
    "Policy",
    "PolicyOptimizer",
    "PolicyOutcome",
    "PROPORTIONAL_REVIEW",
    "REFERENCE_INDUSTRY",
    "ReservationPriceResult",
    "Scenario",
    "SLOW_REVIEW_GROWTH",
    "SUBLINEAR_VERIFICATION",
    "illustrative_industries",
]
