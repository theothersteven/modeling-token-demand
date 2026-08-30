"""Controlled reference calibrations used in the numerical illustrations.

The main experiment starts from one reference industry and changes one
economically ordered parameter group at a time. These cases are deliberately
stylized comparative statics, not empirical industry estimates.
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

# Capability level: lambda moves the capability frontier while its shape nu
# remains fixed, so the two cases are ordered at every delegation horizon.
HIGH_CAPABILITY = replace(
    REFERENCE_INDUSTRY,
    name="Capability: high",
    capability_horizon_hours=24.0,
)
LOW_CAPABILITY = replace(
    REFERENCE_INDUSTRY,
    name="Capability: low",
    capability_horizon_hours=6.0,
)

# Execution level: a changes reliability on solvable tasks while alpha, the
# curvature of returns to inference, remains fixed.
HIGH_EXECUTION = replace(
    REFERENCE_INDUSTRY,
    name="Execution: high",
    execution_scale=8.0,
)
LOW_EXECUTION = replace(
    REFERENCE_INDUSTRY,
    name="Execution: low",
    execution_scale=2.0,
)

# Verification burden: fixed and variable review time move together. Beta is
# held fixed because changing it alters the shape rather than the level of the
# verification technology.
HIGH_VERIFICATION_BURDEN = replace(
    REFERENCE_INDUSTRY,
    name="Verification burden: high",
    verification_fixed_hours=0.06,
    verification_scale=0.10,
)
LOW_VERIFICATION_BURDEN = replace(
    REFERENCE_INDUSTRY,
    name="Verification burden: low",
    verification_fixed_hours=0.015,
    verification_scale=0.025,
)

# Economic surplus moves value and human cost in opposite directions. This
# changes the attractiveness of AI while leaving technical performance fixed.
HIGH_ECONOMIC_SURPLUS = replace(
    REFERENCE_INDUSTRY,
    name="Economic surplus: high",
    value_per_work_hour=150.0,
    human_cost_per_hour=75.0,
)
LOW_ECONOMIC_SURPLUS = replace(
    REFERENCE_INDUSTRY,
    name="Economic surplus: low",
    value_per_work_hour=100.0,
    human_cost_per_hour=125.0,
)

# Adoption hurdle: shifting mu by one baseline sigma yields approximately
# 27%, 50%, and 73% adoption at the reference industry's baseline surplus.
HIGH_ADOPTION_HURDLE = replace(
    REFERENCE_INDUSTRY,
    name="Adoption hurdle: high",
    adoption_midpoint=115.0,
)
LOW_ADOPTION_HURDLE = replace(
    REFERENCE_INDUSTRY,
    name="Adoption hurdle: low",
    adoption_midpoint=99.0,
)

# Optional shape case retained for robustness checks, but excluded from the
# main controlled comparison.
SHARP_ADOPTION_THRESHOLD = replace(
    REFERENCE_INDUSTRY,
    name="Adoption threshold: sharp",
    adoption_midpoint=107.0,
    adoption_scale=0.4,
)


def illustrative_industries(
    include_threshold: bool = False,
) -> tuple[Industry, ...]:
    """Return the reference case and one high/low pair for each group."""

    industries = (
        REFERENCE_INDUSTRY,
        HIGH_CAPABILITY,
        LOW_CAPABILITY,
        HIGH_EXECUTION,
        LOW_EXECUTION,
        HIGH_VERIFICATION_BURDEN,
        LOW_VERIFICATION_BURDEN,
        HIGH_ECONOMIC_SURPLUS,
        LOW_ECONOMIC_SURPLUS,
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
