from dataclasses import replace
import math

from modeling_token_demand import (
    AttentionConstrainedOptimizer,
    IndustryModel,
    OptimizationSettings,
    Policy,
    PolicyOptimizer,
    Scenario,
)
from modeling_token_demand.calibrations import SOFTWARE, illustrative_industries


def test_one_attempt_is_always_consumed() -> None:
    model = IndustryModel(SOFTWARE)
    outcome = model.evaluate(
        Policy(delegation_hours=1.0, tokens_per_work_hour=100_000.0, max_attempts=1),
        Scenario(),
    )

    assert outcome.expected_attempts == 1.0


def test_more_effective_inference_improves_conditional_reliability() -> None:
    model = IndustryModel(SOFTWARE)
    policy = Policy(
        delegation_hours=1.0,
        tokens_per_work_hour=100_000.0,
        max_attempts=3,
    )

    baseline = model.conditional_success(policy, Scenario(token_efficiency=1.0))
    efficient = model.conditional_success(policy, Scenario(token_efficiency=2.0))

    assert efficient > baseline


def test_optimizer_returns_a_feasible_policy() -> None:
    settings = OptimizationSettings(
        grid_points_per_dimension=7,
        local_starts_per_attempt=1,
        max_attempts=4,
    )
    optimizer = PolicyOptimizer(settings)
    model = IndustryModel(SOFTWARE)
    scenario = Scenario()
    outcome = optimizer.solve(model, scenario)
    outcomes_by_attempts = optimizer.solve_by_attempts(model, scenario)
    policy = outcome.policy

    assert (
        settings.min_delegation_hours
        <= policy.delegation_hours
        <= settings.max_delegation_hours
    )
    assert (
        settings.min_tokens_per_work_hour
        <= policy.tokens_per_work_hour
        <= settings.max_tokens_per_work_hour
    )
    assert 1 <= policy.max_attempts <= settings.max_attempts
    assert 0.0 <= outcome.adoption_share <= 1.0
    assert set(outcomes_by_attempts) == set(range(1, settings.max_attempts + 1))
    assert outcome.surplus_per_work_hour == max(
        item.surplus_per_work_hour for item in outcomes_by_attempts.values()
    )


def test_attention_optimizer_maximizes_surplus_per_attention_hour() -> None:
    settings = OptimizationSettings(
        grid_points_per_dimension=7,
        local_starts_per_attempt=1,
        max_attempts=4,
    )
    optimizer = AttentionConstrainedOptimizer(settings)
    model = IndustryModel(SOFTWARE)
    outcome = optimizer.solve(model, Scenario())
    outcomes_by_attempts = optimizer.solve_by_attempts(model, Scenario())

    expected_ratio = (
        outcome.policy.delegation_hours
        * outcome.surplus_per_work_hour
        / (outcome.expected_attempts * outcome.verification_hours_per_attempt)
    )
    assert outcome.surplus_per_attention_hour == expected_ratio
    assert outcome.surplus_per_attention_hour == max(
        item.surplus_per_attention_hour for item in outcomes_by_attempts.values()
    )
    assert outcome.policy.max_attempts == 1


def test_scalar_attention_solution_matches_general_optimizer() -> None:
    settings = OptimizationSettings(
        max_tokens_per_work_hour=20_000_000.0,
        grid_points_per_dimension=11,
        local_starts_per_attempt=2,
        max_attempts=4,
    )
    optimizer = AttentionConstrainedOptimizer(settings)

    for industry in illustrative_industries(include_threshold=False):
        model = IndustryModel(industry)
        for capability in (0.35, 1.0, 5.0):
            scenario = Scenario(model_capability=capability)
            general = optimizer.solve(model, scenario)
            scalar = optimizer.solve_interior(model, scenario)

            assert math.isclose(
                general.surplus_per_attention_hour,
                scalar.surplus_per_attention_hour,
                rel_tol=1e-9,
            )
            assert scalar.policy.max_attempts == 1


def test_attention_shadow_price_capability_elasticity() -> None:
    settings = OptimizationSettings(max_tokens_per_work_hour=20_000_000.0)
    optimizer = AttentionConstrainedOptimizer(settings)
    epsilon = 1e-4

    for industry in illustrative_industries(include_threshold=False):
        model = IndustryModel(industry)
        center = optimizer.solve_interior(model, Scenario(model_capability=1.0))
        lower = optimizer.solve_interior(
            model, Scenario(model_capability=math.exp(-epsilon))
        )
        upper = optimizer.solve_interior(
            model, Scenario(model_capability=math.exp(epsilon))
        )
        gross_shadow_price = (
            center.surplus_per_attention_hour + industry.human_cost_per_hour
        )
        lower_shadow_price = (
            lower.surplus_per_attention_hour + industry.human_cost_per_hour
        )
        upper_shadow_price = (
            upper.surplus_per_attention_hour + industry.human_cost_per_hour
        )
        s = center.policy.delegation_hours
        variable_review = industry.verification_scale * (
            s ** industry.verification_elasticity
        )
        review_elasticity = (
            industry.verification_elasticity * variable_review
            / (industry.verification_fixed_hours + variable_review)
        )
        numerical_elasticity = math.log(
            upper_shadow_price / lower_shadow_price
        ) / (2.0 * epsilon)

        assert gross_shadow_price > 0
        assert math.isclose(
            numerical_elasticity,
            1.0 - review_elasticity,
            rel_tol=1e-7,
            abs_tol=1e-7,
        )


def test_uniform_verification_speed_preserves_attention_policy() -> None:
    industry = replace(SOFTWARE, human_attention_hours=100_000.0)
    model = IndustryModel(industry)
    optimizer = AttentionConstrainedOptimizer(
        OptimizationSettings(max_tokens_per_work_hour=20_000_000.0)
    )
    multiplier = 0.4

    baseline = optimizer.solve_interior(
        model, Scenario(verification_time_multiplier=1.0)
    )
    faster = optimizer.solve_interior(
        model, Scenario(verification_time_multiplier=multiplier)
    )

    assert math.isclose(
        faster.policy.delegation_hours,
        baseline.policy.delegation_hours,
        rel_tol=1e-12,
    )
    assert math.isclose(
        faster.policy.tokens_per_work_hour,
        baseline.policy.tokens_per_work_hour,
        rel_tol=1e-12,
    )
    assert faster.policy.max_attempts == baseline.policy.max_attempts == 1
    assert faster.attention_limited_tokens is not None
    assert baseline.attention_limited_tokens is not None
    assert math.isclose(
        faster.attention_limited_tokens,
        baseline.attention_limited_tokens / multiplier,
        rel_tol=1e-12,
    )
