"""Reference calibration and one-parameter comparative-static cases.

Every industry used in the main figures starts from ``REFERENCE_INDUSTRY`` and
changes exactly one industry parameter.  This keeps the qualitative mechanism
behind each line identifiable.  The cases are stylized comparative statics,
not empirical industry estimates.
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
    value_per_work_hour=100.0,
    human_cost_per_hour=100.0,
    # Single-attempt baseline surplus is about 80.3. Place the common adoption
    # location nearby to illustrate takeoff and saturation, not to fit data.
    adoption_location=81.0,
    adoption_scale=4.0,
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
    value_per_work_hour=200.0,
)
LOW_ECONOMIC_VALUE = replace(
    REFERENCE_INDUSTRY,
    name="Economic value: low",
    value_per_work_hour=50.0,
)

# Work-limited comparisons move only the location of the adoption hurdle.  The
# low-hurdle case is nearly saturated at the reference technology, while the
# high-hurdle case still has a large adoption margin.
HIGH_ADOPTION_HURDLE = replace(
    REFERENCE_INDUSTRY,
    name="High adoption hurdle",
    adoption_location=95.0,
)
LOW_ADOPTION_HURDLE = replace(
    REFERENCE_INDUSTRY,
    name="Low adoption hurdle",
    adoption_location=40.0,
)

# Attention-limited comparisons isolate execution and review mechanisms one
# parameter at a time.
HARD_EXECUTION = replace(
    REFERENCE_INDUSTRY,
    name="Hard execution",
    execution_scale=1.0,
)
HIGH_CAPABILITY_REQUIREMENT = replace(
    REFERENCE_INDUSTRY,
    name="High capability requirement",
    capability_horizon_hours=3.0,
)
PROPORTIONAL_REVIEW = replace(
    REFERENCE_INDUSTRY,
    name="Nearly proportional review",
    verification_elasticity=0.95,
)
SLOW_REVIEW_GROWTH = replace(
    REFERENCE_INDUSTRY,
    name="Slow-growing review",
    verification_elasticity=0.15,
)
LOW_INFERENCE_RETURNS = replace(
    REFERENCE_INDUSTRY,
    name="Low inference returns",
    inference_returns=0.20,
)

# The main adoption comparison varies dispersion alone. A high concentration
# means a SMALL sigma, not higher hurdles. The untruncated CDFs cross at mu;
# dispersion therefore changes the shape of adoption, not its value at mu.
HIGH_ADOPTION_CONCENTRATION = replace(
    REFERENCE_INDUSTRY, name="Adoption concentration: high", adoption_scale=1.0,
)
LOW_ADOPTION_CONCENTRATION = replace(
    REFERENCE_INDUSTRY, name="Adoption concentration: low", adoption_scale=16.0,
)

# Optional shape case retained for robustness checks, but excluded from the
# main controlled comparison.
SHARP_ADOPTION_THRESHOLD = replace(
    REFERENCE_INDUSTRY,
    name="Adoption threshold: sharp",
    adoption_location=107.0,
    adoption_scale=0.4,
)


def illustrative_industries(
    include_threshold: bool = False, *, include_singletons: bool = False,
) -> tuple[Industry, ...]:
    """Return the eight one-at-a-time cases used in the numerical gallery."""

    industries = (
        REFERENCE_INDUSTRY,
        LOW_ADOPTION_HURDLE,
        HIGH_ADOPTION_HURDLE,
        HARD_EXECUTION,
        HIGH_CAPABILITY_REQUIREMENT,
        LOW_INFERENCE_RETURNS,
        SLOW_REVIEW_GROWTH,
        PROPORTIONAL_REVIEW,
    )
    if include_singletons:
        industries += singleton_industries()
    if include_threshold:
        industries += (SHARP_ADOPTION_THRESHOLD,)
    return industries


# Singleton rows in the parameter table. These are explicitly not high/low
# comparisons: concentration is not an ordering of hurdle difficulty, and
# the other shapes require combinations of parameter groups.
# Compatibility name for the previously separate focused example.
CONCENTRATED_ADOPTION = HIGH_ADOPTION_CONCENTRATION
EARLY_SATURATION = replace(
    CONCENTRATED_ADOPTION,
    name="Early saturation",
    capability_horizon_hours=36.0,
    execution_scale=8.0,
)
SUPERVISORY_LEVERAGE = replace(
    REFERENCE_INDUSTRY,
    name="Supervisory leverage",
    verification_elasticity=0.25,
)
REVIEW_BOTTLENECK = PROPORTIONAL_REVIEW
CAPABILITY_VALLEY = replace(
    REFERENCE_INDUSTRY,
    name="Capability valley",
    capability_horizon_hours=36.0,
    inference_returns=0.25,
    verification_fixed_hours=0.001,
    verification_elasticity=0.90,
)
OFFSETTING_EFFICIENCY = replace(
    REFERENCE_INDUSTRY,
    name="Offsetting efficiency",
    capability_horizon_hours=36.0,
    execution_scale=2.0,
    verification_elasticity=0.80,
)


def singleton_industries() -> tuple[Industry, ...]:
    return EARLY_SATURATION, CAPABILITY_VALLEY, OFFSETTING_EFFICIENCY


def work_paradigms() -> tuple[Industry, ...]:
    return (
        REFERENCE_INDUSTRY,
        LOW_ADOPTION_HURDLE,
        HIGH_ADOPTION_HURDLE,
        HARD_EXECUTION,
        HIGH_CAPABILITY_REQUIREMENT,
    )


def attention_paradigms() -> tuple[Industry, ...]:
    """Return one reference and four cases that each change one parameter."""
    return (
        REFERENCE_INDUSTRY,
        HARD_EXECUTION,
        LOW_INFERENCE_RETURNS,
        SLOW_REVIEW_GROWTH,
        PROPORTIONAL_REVIEW,
    )


def calibration_tables_markdown() -> str:
    """Plot lines as rows and industry parameters as columns.

    Bold cells are the only cells that differ from the reference.  The
    Execution ease ``a`` is defined at the code's fixed reference model effort
    level, so it remains comparable when inference returns change.
    """

    fields = (
        ("capability_horizon_hours", r"$\lambda$"),
        ("capability_shape", r"$\nu$"),
        ("execution_scale", r"$a$"),
        ("inference_returns", r"$\alpha$"),
        ("verification_fixed_hours", r"$h_0$"),
        ("verification_scale", r"$h_1$"),
        ("verification_elasticity", r"$\beta$"),
        ("value_per_work_hour", r"$b$"),
        ("human_cost_per_hour", r"$w$"),
        ("adoption_location", r"$\mu$"),
        ("adoption_scale", r"$\sigma$"),
    )

    def displayed_value(industry: Industry, field: str) -> float:
        return getattr(industry, field)

    def block(title: str, industries: tuple[Industry, ...]) -> str:
        rows = [
            f"**{title}**",
            "",
            "| Plot line | " + " | ".join(label for _, label in fields) + " |",
            "|---|" + "---:|" * len(fields),
        ]
        for industry in industries:
            values = []
            for field, _ in fields:
                value = displayed_value(industry, field)
                reference = displayed_value(REFERENCE_INDUSTRY, field)
                formatted = f"{value:.5g}"
                values.append(f"**{formatted}**" if value != reference else formatted)
            rows.append(f"| {industry.name} | " + " | ".join(values) + " |")
        return "\n".join(rows)

    return "\n\n".join((
        block("Work-limited plot lines", work_paradigms()),
        block("Attention-limited plot lines", attention_paradigms()),
    ))


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
    adoption_location=112.0,
    adoption_scale=6.0,
)
NEAR_LINEAR_VERIFICATION = replace(
    REFERENCE_INDUSTRY,
    name="High verification burden",
    capability_shape=1.15,
    verification_fixed_hours=0.025,
    verification_scale=0.13,
    verification_elasticity=0.95,
    adoption_location=100.0,
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
    adoption_location=30.0,
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
    adoption_location=152.0,
    adoption_scale=10.0,
)
ADOPTION_THRESHOLD = replace(
    SUBLINEAR_VERIFICATION,
    name="Sharp adoption threshold",
    adoption_location=111.6,
    adoption_scale=0.4,
)
SOFTWARE = SUBLINEAR_VERIFICATION
HIGH_REVIEW = NEAR_LINEAR_VERIFICATION
ROUTINE_AUTOMATION = LOW_COST_VERIFICATION
FRONTIER_WORK = TIGHT_CAPABILITY_FRONTIER
