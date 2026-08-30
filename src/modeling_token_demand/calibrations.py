"""Controlled reference calibrations used in the numerical illustrations.

The main experiment starts from one reference industry and changes one related
parameter group at a time. The high/low labels describe the named feature over
the illustrated policy range, not a global ordering of shape parameters.
These are stylized comparative statics, not empirical industry estimates.
"""

from dataclasses import replace

from .model import Industry


REFERENCE_INDUSTRY = Industry(
    name="Reference industry",
    capability_horizon_hours=12.0,
    capability_shape=1.25,
    execution_scale=4.0,
    inference_returns=0.50,
    verification_fixed_hours=0.03,
    verification_scale=0.05,
    verification_elasticity=0.50,
    value_per_work_hour=125.0,
    human_cost_per_hour=100.0,
    # This is close to optimized baseline surplus, so baseline adoption is
    # approximately one half rather than near zero or one.
    adoption_midpoint=107.0,
    adoption_scale=8.0,
)

# A tighter frontier combines a shorter capability horizon with more failures
# at modest task sizes. Changing nu can reverse the ordering at extreme s/m;
# the notebook checks the ordering at the policies used in the figures.
HIGH_CAPABILITY_CONSTRAINT = replace(
    REFERENCE_INDUSTRY,
    name="Capability constraint: high",
    capability_horizon_hours=6.0,
    capability_shape=1.00,
)
LOW_CAPABILITY_CONSTRAINT = replace(
    REFERENCE_INDUSTRY,
    name="Capability constraint: low",
    capability_horizon_hours=24.0,
    capability_shape=1.50,
)

# Harder execution combines lower baseline ease with weaker returns to extra
# inference. The joint change lowers reliability over the illustrated range.
HIGH_EXECUTION_DIFFICULTY = replace(
    REFERENCE_INDUSTRY,
    name="Execution difficulty: high",
    execution_scale=2.0,
    inference_returns=0.35,
)
LOW_EXECUTION_DIFFICULTY = replace(
    REFERENCE_INDUSTRY,
    name="Execution difficulty: low",
    execution_scale=8.0,
    inference_returns=0.65,
)

# One verification group combines the level of review time with how quickly
# review grows as the user delegates larger tasks.
HIGH_VERIFICATION_BURDEN = replace(
    REFERENCE_INDUSTRY,
    name="Verification burden: high",
    verification_fixed_hours=0.06,
    verification_scale=0.10,
    verification_elasticity=0.95,
)
LOW_VERIFICATION_BURDEN = replace(
    REFERENCE_INDUSTRY,
    name="Verification burden: low",
    verification_fixed_hours=0.015,
    verification_scale=0.025,
    verification_elasticity=0.25,
)

# Vary output value alone; the opportunity cost of human attention stays fixed.
HIGH_ECONOMIC_VALUE = replace(
    REFERENCE_INDUSTRY,
    name="Economic value: high",
    value_per_work_hour=150.0,
)
LOW_ECONOMIC_VALUE = replace(
    REFERENCE_INDUSTRY,
    name="Economic value: low",
    value_per_work_hour=100.0,
)

# Higher typical hurdles are also more dispersed. These CDFs cross at surplus
# 75, below the surplus range in the adoption comparisons; dispersion alone
# would not define an unambiguously harder adoption environment.
HIGH_ADOPTION_HURDLE = replace(
    REFERENCE_INDUSTRY,
    name="Adoption hurdle: high",
    adoption_midpoint=115.0,
    adoption_scale=10.0,
)
LOW_ADOPTION_HURDLE = replace(
    REFERENCE_INDUSTRY,
    name="Adoption hurdle: low",
    adoption_midpoint=99.0,
    adoption_scale=6.0,
)

# Optional shape case retained for robustness checks, but excluded from the
# main controlled comparison.
SHARP_ADOPTION_THRESHOLD = replace(
    REFERENCE_INDUSTRY,
    name="Adoption threshold: sharp",
    adoption_midpoint=107.0,
    adoption_scale=0.4,
)


def illustrative_industries(include_threshold: bool = False) -> tuple[Industry, ...]:
    """Return the reference and five joint high/low parameter-group pairs."""

    industries = (
        REFERENCE_INDUSTRY,
        HIGH_CAPABILITY_CONSTRAINT,
        LOW_CAPABILITY_CONSTRAINT,
        HIGH_EXECUTION_DIFFICULTY,
        LOW_EXECUTION_DIFFICULTY,
        HIGH_VERIFICATION_BURDEN,
        LOW_VERIFICATION_BURDEN,
        HIGH_ECONOMIC_VALUE,
        LOW_ECONOMIC_VALUE,
        HIGH_ADOPTION_HURDLE,
        LOW_ADOPTION_HURDLE,
    )
    if include_threshold:
        return industries + (SHARP_ADOPTION_THRESHOLD,)
    return industries


# Historical calibrations remain available for reproducibility, but are not
# returned by illustrative_industries or used in the current figures.
SUBLINEAR_VERIFICATION = replace(
    REFERENCE_INDUSTRY,
    name="Low verification growth",
    capability_horizon_hours=15.0,
    execution_scale=5.0,
    inference_returns=0.55,
    verification_fixed_hours=0.035,
    verification_scale=0.025,
    verification_elasticity=0.35,
    adoption_midpoint=112.0,
    adoption_scale=6.0,
)
NEAR_LINEAR_VERIFICATION = replace(
    REFERENCE_INDUSTRY,
    name="High verification burden",
    capability_shape=1.15,
    verification_fixed_hours=0.025,
    verification_scale=0.13,
    verification_elasticity=0.95,
    adoption_midpoint=100.0,
    adoption_scale=7.0,
)
LOW_COST_VERIFICATION = replace(
    REFERENCE_INDUSTRY,
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
)
TIGHT_CAPABILITY_FRONTIER = replace(
    REFERENCE_INDUSTRY,
    name="Tight capability frontier",
    capability_horizon_hours=2.0,
    capability_shape=1.40,
    execution_scale=1.5,
    inference_returns=0.60,
    verification_fixed_hours=0.04,
    verification_scale=0.08,
    value_per_work_hour=220.0,
    human_cost_per_hour=150.0,
    adoption_midpoint=152.0,
    adoption_scale=10.0,
)
ADOPTION_THRESHOLD = replace(
    SUBLINEAR_VERIFICATION,
    name="Sharp adoption threshold",
    adoption_midpoint=111.6,
    adoption_scale=0.4,
)
SOFTWARE = SUBLINEAR_VERIFICATION
HIGH_REVIEW = NEAR_LINEAR_VERIFICATION
ROUTINE_AUTOMATION = LOW_COST_VERIFICATION
FRONTIER_WORK = TIGHT_CAPABILITY_FRONTIER
