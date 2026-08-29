"""Illustrative industry regimes used in examples and plots.

These are deliberately stylized, not empirical estimates.  Each calibration
changes only industry primitives; technology and token prices belong in a
separate :class:`Scenario` so comparative statics remain transparent.
"""

from dataclasses import replace

from .model import Industry


SOFTWARE = Industry(
    name="Software development",
    capability_horizon_hours=15.0,
    capability_shape=1.25,
    execution_scale=5.0,
    inference_returns=0.55,
    verification_fixed_hours=0.035,
    verification_scale=0.025,
    verification_elasticity=0.35,
    value_per_work_hour=125.0,
    human_cost_per_hour=100.0,
    adoption_midpoint=112.0,
    adoption_scale=6.0,
)

HIGH_REVIEW = Industry(
    name="High-review professional work",
    capability_horizon_hours=12.0,
    capability_shape=1.15,
    execution_scale=4.0,
    inference_returns=0.50,
    verification_fixed_hours=0.025,
    verification_scale=0.13,
    verification_elasticity=0.95,
    value_per_work_hour=125.0,
    human_cost_per_hour=100.0,
    adoption_midpoint=100.0,
    adoption_scale=7.0,
)

ROUTINE_AUTOMATION = Industry(
    name="Routine automation",
    capability_horizon_hours=40.0,
    capability_shape=1.35,
    execution_scale=8.0,
    inference_returns=0.45,
    verification_fixed_hours=0.025,
    verification_scale=0.012,
    verification_elasticity=0.25,
    value_per_work_hour=100.0,
    human_cost_per_hour=75.0,
    adoption_midpoint=30.0,
    adoption_scale=8.0,
)

FRONTIER_WORK = Industry(
    name="Capability-limited frontier work",
    capability_horizon_hours=2.0,
    capability_shape=1.40,
    execution_scale=1.5,
    inference_returns=0.60,
    verification_fixed_hours=0.04,
    verification_scale=0.08,
    verification_elasticity=0.50,
    value_per_work_hour=220.0,
    human_cost_per_hour=150.0,
    adoption_midpoint=152.0,
    adoption_scale=10.0,
)

# A software-like industry with many tasks clustered near a sharp adoption
# threshold.  It is useful for illustrating how efficiency can unlock enough
# new work to outweigh the tokens saved on each existing task.
ADOPTION_THRESHOLD = replace(
    SOFTWARE,
    name="Near an adoption threshold",
    adoption_midpoint=111.6,
    adoption_scale=0.4,
)


def illustrative_industries(include_threshold: bool = True) -> tuple[Industry, ...]:
    """Return the standard non-empirical calibrations in display order."""

    industries = (SOFTWARE, HIGH_REVIEW, ROUTINE_AUTOMATION, FRONTIER_WORK)
    if include_threshold:
        return industries + (ADOPTION_THRESHOLD,)
    return industries
