"""Economic model and numerical tools for studying AI token demand."""

from .model import (
    Industry,
    IndustryModel,
    Policy,
    PolicyOutcome,
    Scenario,
)
from .optimizer import OptimizationSettings, PolicyOptimizer
from .calibrations import illustrative_industries

__all__ = [
    "Industry",
    "IndustryModel",
    "OptimizationSettings",
    "Policy",
    "PolicyOptimizer",
    "PolicyOutcome",
    "Scenario",
    "illustrative_industries",
]
