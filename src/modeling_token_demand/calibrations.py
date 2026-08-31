"""Reference, paired comparisons, and singleton demand archetypes.

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
    value_per_work_hour=100.0,
    human_cost_per_hour=100.0,
    # Single-attempt baseline surplus is about 75.3. Place the common adoption
    # location nearby to illustrate takeoff and saturation, not to fit data.
    adoption_location=76.0,
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

# Higher typical hurdles are also more dispersed. These CDFs cross at surplus
# 75, below the surplus range in the adoption comparisons; dispersion alone
# would not define an unambiguously harder adoption environment.
HIGH_ADOPTION_HURDLE = replace(
    REFERENCE_INDUSTRY,
    name="Adoption hurdle: high",
    adoption_location=115.0,
    adoption_scale=10.0,
)
LOW_ADOPTION_HURDLE = replace(
    REFERENCE_INDUSTRY,
    name="Adoption hurdle: low",
    adoption_location=99.0,
    adoption_scale=6.0,
)

# The main adoption comparison varies dispersion alone. A high concentration
# means a SMALL sigma, not higher hurdles. The untruncated CDFs cross at mu;
# after truncation at zero, crossings remain nearby, not exactly at mu.
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
    include_threshold: bool = False, *, include_singletons: bool = True,
) -> tuple[Industry, ...]:
    """Return the Section 3 cases, optionally restricted to paired comparisons."""

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
        HIGH_ADOPTION_CONCENTRATION,
        LOW_ADOPTION_CONCENTRATION,
    )
    if include_singletons:
        industries += singleton_industries()
    if include_threshold:
        industries += (SHARP_ADOPTION_THRESHOLD,)
    return industries


# Singleton rows in the Section 3 table. These are explicitly not high/low
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
REVIEW_BOTTLENECK = replace(
    REFERENCE_INDUSTRY,
    name="Review bottleneck",
    verification_elasticity=0.95,
)
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
    return REFERENCE_INDUSTRY, CONCENTRATED_ADOPTION, EARLY_SATURATION


def attention_paradigms() -> tuple[Industry, ...]:
    """Focus panels use cases from the main table, not a second calibration set."""
    return LOW_VERIFICATION_BURDEN, HIGH_VERIFICATION_BURDEN, CAPABILITY_VALLEY


def calibration_tables_markdown() -> str:
    """Parameter rows and condition columns, shared by paper and notebook.

    Smaller blocks avoid a fourteen-condition-wide matrix. Every numeric
    cell is explicit; bold cells differ from the reference in that row.
    """
    parameters = {
        "capability_horizon_hours": r"Capability horizon $\lambda$",
        "capability_shape": r"Frontier shape $\nu$",
        "execution_scale": r"Execution ease $a$",
        "inference_returns": r"Inference returns $\alpha$",
        "verification_fixed_hours": r"Fixed review $h_0$ (hours)",
        "verification_scale": r"Review scale $h_1$",
        "verification_elasticity": r"Review growth $\beta$",
        "value_per_work_hour": r"Work value $b$ (dollars)",
        "adoption_location": r"Hurdle location $\mu$ (dollars)",
        "adoption_scale": r"Hurdle spread $\sigma$ (dollars)",
    }
    groups = (
        ("Technical conditions", (
            ("Capability low", LOW_CAPABILITY_CONSTRAINT),
            ("Capability high", HIGH_CAPABILITY_CONSTRAINT),
            ("Execution low", LOW_EXECUTION_DIFFICULTY),
            ("Execution high", HIGH_EXECUTION_DIFFICULTY),
            ("Review low", LOW_VERIFICATION_BURDEN),
            ("Review high", HIGH_VERIFICATION_BURDEN),
        ), tuple(parameters)[:7]),
        ("Economic and adoption conditions", (
            ("Value low", LOW_ECONOMIC_VALUE),
            ("Value high", HIGH_ECONOMIC_VALUE),
            ("Concentration low", LOW_ADOPTION_CONCENTRATION),
            ("Concentration high", HIGH_ADOPTION_CONCENTRATION),
        ), ("value_per_work_hour", "adoption_scale", "adoption_location")),
        ("Singleton conditions", tuple((i.name, i) for i in singleton_industries()), (
            "capability_horizon_hours", "execution_scale", "inference_returns",
            "verification_fixed_hours", "verification_elasticity", "adoption_scale",
        )),
    )
    blocks = []
    for title, variants, fields in groups:
        columns = (("Reference", REFERENCE_INDUSTRY),) + variants
        rows = [f"**{title}**", "",
                "| Parameter | " + " | ".join(label for label, _ in columns) + " |",
                "|---|" + "---:|" * len(columns)]
        for field in fields:
            reference = getattr(REFERENCE_INDUSTRY, field)
            values = []
            for _, industry in columns:
                value = getattr(industry, field)
                formatted = f"{value:g}"
                values.append(f"**{formatted}**" if value != reference else formatted)
            rows.append("| " + parameters[field] + " | " + " | ".join(values) + " |")
        blocks.append("\n".join(rows))
    return "\n\n".join(blocks)


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
