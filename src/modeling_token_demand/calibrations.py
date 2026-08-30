"""Illustrative parameter regimes used in plots.

These are deliberately stylized, not empirical estimates.  Each calibration
changes only model primitives; technology and token prices belong in a
separate :class:`Scenario` so comparative statics remain transparent.
"""

from dataclasses import replace

from .model import Industry


SUBLINEAR_VERIFICATION = Industry(
    name="Low verification growth",
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

NEAR_LINEAR_VERIFICATION = Industry(
    name="High verification burden",
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

LOW_COST_VERIFICATION = Industry(
    name="Low verification cost",
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

TIGHT_CAPABILITY_FRONTIER = Industry(
    name="Tight capability frontier",
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

# A regime with many tasks clustered near a sharp adoption threshold. It is
# useful for showing how efficiency can unlock enough work to outweigh the
# tokens saved on each existing task.
ADOPTION_THRESHOLD = replace(
    SUBLINEAR_VERIFICATION,
    name="Sharp adoption threshold",
    adoption_midpoint=111.6,
    adoption_scale=0.4,
)


def illustrative_industries(include_threshold: bool = True) -> tuple[Industry, ...]:
    """Return the standard non-empirical parameter regimes in display order."""

    industries = (
        SUBLINEAR_VERIFICATION,
        NEAR_LINEAR_VERIFICATION,
        LOW_COST_VERIFICATION,
        TIGHT_CAPABILITY_FRONTIER,
    )
    if include_threshold:
        return industries + (ADOPTION_THRESHOLD,)
    return industries


# Backward-compatible aliases for earlier notebooks and imports.
SOFTWARE = SUBLINEAR_VERIFICATION
HIGH_REVIEW = NEAR_LINEAR_VERIFICATION
ROUTINE_AUTOMATION = LOW_COST_VERIFICATION
FRONTIER_WORK = TIGHT_CAPABILITY_FRONTIER
