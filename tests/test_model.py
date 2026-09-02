from dataclasses import fields, replace
import math

import pytest

from modeling_token_demand import (
    AttentionConstrainedOptimizer,
    IndustryModel,
    OptimizationSettings,
    Policy,
    PolicyOptimizer,
    Scenario,
)
from modeling_token_demand.calibrations import REFERENCE_INDUSTRY, SOFTWARE, illustrative_industries


def test_policy_has_only_scope_and_model_effort() -> None:
    assert [field.name for field in fields(Policy)] == [
        "delegation_hours", "tokens_per_work_hour"
    ]


def test_normalized_model_effort_has_a_minimum_of_one() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Policy(delegation_hours=1, tokens_per_work_hour=.99)
    with pytest.raises(ValueError, match="minimum viable effort"):
        OptimizationSettings(min_tokens_per_work_hour=.99)
    assert OptimizationSettings().min_tokens_per_work_hour == 1


def test_failed_work_still_pays_tokens_and_consumes_scarce_attention() -> None:
    industry = replace(
        REFERENCE_INDUSTRY, verification_fixed_hours=.1,
        verification_scale=.2, verification_elasticity=1,
        human_attention_hours=1_000,
    )
    model = IndustryModel(industry)
    policy = Policy(delegation_hours=2, tokens_per_work_hour=1)
    failed = model.evaluate(policy, Scenario(model_capability=1e-9))
    reliable = model.evaluate(policy, Scenario(model_capability=1e9))

    # Each two-hour chunk spends $2 on tokens and $50 on half an hour of
    # review. With no output, both resource objectives must be negative.
    assert failed.success_probability == 0
    assert failed.cost_per_work_hour == pytest.approx(26)
    assert failed.surplus_per_work_hour == pytest.approx(-26)
    assert failed.surplus_per_attention_hour == pytest.approx(-104)
    assert reliable.cost_per_work_hour == failed.cost_per_work_hour
    assert reliable.surplus_per_work_hour - failed.surplus_per_work_hour == pytest.approx(100)
    # This is capacity conditional on operating, not a recommendation to
    # operate at negative value. Failure does not free a review slot.
    assert failed.attention_limited_tokens == pytest.approx(4_000)
    assert reliable.attention_limited_tokens == pytest.approx(4_000)
    assert 0 < failed.adoption_share < 1e-10
    assert failed.work_limited_tokens == pytest.approx(
        industry.potential_work_hours
        * failed.adoption_share
        * policy.tokens_per_work_hour
    )


@pytest.mark.parametrize("location, spread", [(76, 16), (0, 4), (-20, 4)])
def test_adoption_is_an_untruncated_logistic_distribution(location, spread) -> None:
    model = IndustryModel(replace(
        REFERENCE_INDUSTRY, adoption_location=location, adoption_scale=spread,
    ))
    assert model.adoption_share(-1e6) == 0
    assert model.adoption_share(location) == pytest.approx(.5)
    assert model.adoption_share(1e6) == 1
    values = [model.adoption_share(u) for u in (-100, -1, 0, 1, 100)]
    assert values == sorted(values)
    expected = 1 / (1 + math.exp((location - 10) / spread))
    assert model.adoption_share(10) == pytest.approx(expected)


def test_success_combines_feasibility_and_execution_without_discounting_costs() -> None:
    model = IndustryModel(REFERENCE_INDUSTRY)
    policy = Policy(delegation_hours=12, tokens_per_work_hour=1)
    outcome = model.evaluate(policy, Scenario())
    # s=lambda makes q=exp(-1), and normalized inference=1 makes r=exp(-3).
    assert outcome.capability_share == pytest.approx(math.exp(-1))
    assert outcome.conditional_success == pytest.approx(math.exp(-3))
    assert outcome.success_probability == pytest.approx(math.exp(-4))
    assert model.success_probability(policy, Scenario()) == outcome.success_probability
    assert outcome.success_probability < outcome.capability_share


def test_more_effective_inference_improves_conditional_reliability() -> None:
    model = IndustryModel(SOFTWARE)
    policy = Policy(
        delegation_hours=1.0,
        tokens_per_work_hour=1.0,
    )

    baseline = model.conditional_success(policy, Scenario(token_efficiency=1.0))
    efficient = model.conditional_success(policy, Scenario(token_efficiency=2.0))

    assert efficient > baseline


def test_shifted_review_cost_is_fixed_at_zero_and_lower_beta_is_always_cheaper() -> None:
    base = replace(
        REFERENCE_INDUSTRY,
        verification_fixed_hours=.1,
        verification_scale=.2,
    )
    policy = Policy(delegation_hours=.25, tokens_per_work_hour=1)
    model = IndustryModel(replace(base, verification_elasticity=.5))
    assert model.verification_hours(policy, Scenario()) == pytest.approx(
        .1 + .2 * (math.sqrt(1.25) - 1)
    )
    for scope in (.01, .25, 1, 10):
        policy = replace(policy, delegation_hours=scope)
        low = IndustryModel(replace(base, verification_elasticity=.1)) \
            .verification_hours(policy, Scenario())
        high = IndustryModel(replace(base, verification_elasticity=.9)) \
            .verification_hours(policy, Scenario())
        assert .1 < low < high


def test_optimizer_returns_a_feasible_policy() -> None:
    settings = OptimizationSettings(
        grid_points_per_dimension=7,
        local_starts=1,
    )
    optimizer = PolicyOptimizer(settings)
    model = IndustryModel(SOFTWARE)
    scenario = Scenario()
    outcome = optimizer.solve(model, scenario)
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
    assert 0.0 <= outcome.adoption_share <= 1.0


def test_work_reservation_price_reoptimizes_to_baseline_surplus() -> None:
    optimizer = PolicyOptimizer(
        OptimizationSettings(max_tokens_per_work_hour=2_000)
    )
    model = IndustryModel(REFERENCE_INDUSTRY)
    baseline = optimizer.solve(model, Scenario())
    capable_at_baseline_price = optimizer.solve(
        model, Scenario(model_capability=5)
    )

    result = optimizer.solve_reservation_price(
        model,
        Scenario(model_capability=5),
        baseline.surplus_per_work_hour,
    )

    assert result.token_price == pytest.approx(9.48206, rel=2e-4)
    assert result.outcome.surplus_per_work_hour == pytest.approx(
        baseline.surplus_per_work_hour, rel=1e-6
    )
    assert result.outcome.policy != capable_at_baseline_price.policy
    assert result.outcome.policy.tokens_per_work_hour == pytest.approx(1)
    assert result.iterations > 0


def test_attention_reservation_price_reoptimizes_to_the_target_value() -> None:
    optimizer = AttentionConstrainedOptimizer()
    industry = replace(REFERENCE_INDUSTRY, human_attention_hours=100_000)
    model = IndustryModel(industry)
    baseline = optimizer.solve_interior(model, Scenario())
    capable_at_baseline_price = optimizer.solve_interior(
        model, Scenario(model_capability=2)
    )

    result = optimizer.solve_reservation_price(
        model,
        Scenario(model_capability=2),
        baseline.surplus_per_attention_hour,
    )

    assert result.token_price > 1
    assert result.outcome.surplus_per_attention_hour == pytest.approx(
        baseline.surplus_per_attention_hour, rel=1e-6
    )
    assert abs(result.objective_gap) < 1e-3
    assert result.iterations > 0
    assert result.outcome.policy != capable_at_baseline_price.policy
    assert (
        result.outcome.policy.tokens_per_work_hour
        < capable_at_baseline_price.policy.tokens_per_work_hour
    )


def test_attention_reservation_price_returns_the_starting_price_at_target() -> None:
    optimizer = AttentionConstrainedOptimizer()
    industry = replace(REFERENCE_INDUSTRY, human_attention_hours=100_000)
    model = IndustryModel(industry)
    scenario = Scenario(model_capability=3, token_price=2)
    target = optimizer.solve_interior(model, scenario)

    result = optimizer.solve_reservation_price(
        model, scenario, target.surplus_per_attention_hour
    )

    assert result.token_price == pytest.approx(2)
    assert result.iterations == 0
    assert result.function_calls == 1


def test_attention_reservation_price_respects_the_effort_floor() -> None:
    optimizer = AttentionConstrainedOptimizer(
        OptimizationSettings(max_tokens_per_work_hour=2_000)
    )
    industry = replace(REFERENCE_INDUSTRY, human_attention_hours=100_000)
    model = IndustryModel(industry)
    target = optimizer.solve(model, Scenario()).surplus_per_attention_hour

    result = optimizer.solve_reservation_price(
        model,
        Scenario(model_capability=5),
        target,
    )

    assert result.token_price == pytest.approx(31.2133, rel=2e-4)
    assert result.outcome.policy.tokens_per_work_hour == pytest.approx(1)
    assert result.outcome.surplus_per_attention_hour == pytest.approx(
        target, rel=1e-6
    )
    execution_exponent = (
        result.outcome.policy.delegation_hours
        / (
            industry.execution_scale
            * 5
            * result.outcome.policy.tokens_per_work_hour ** industry.inference_returns
        )
    )
    marginal_value_of_effort = (
        industry.value_per_work_hour
        * result.outcome.success_probability
        * industry.inference_returns
        * execution_exponent
        / result.outcome.policy.tokens_per_work_hour
    )
    assert marginal_value_of_effort <= result.token_price


def test_each_regime_selects_policy_for_its_own_scarce_resource() -> None:
    settings = OptimizationSettings(
        grid_points_per_dimension=7,
        local_starts=1,
    )
    optimizer = AttentionConstrainedOptimizer(settings)
    model = IndustryModel(REFERENCE_INDUSTRY)
    outcome = optimizer.solve(model, Scenario())
    work = PolicyOptimizer(settings).solve(model, Scenario())
    assert outcome.surplus_per_attention_hour > work.surplus_per_attention_hour
    assert work.surplus_per_work_hour > outcome.surplus_per_work_hour
    assert outcome.policy.delegation_hours > work.policy.delegation_hours


def test_attention_policy_is_work_policy_at_the_scarcity_price() -> None:
    industry = REFERENCE_INDUSTRY
    attention = AttentionConstrainedOptimizer().solve(IndustryModel(industry), Scenario())
    scarcity_price = attention.surplus_per_attention_hour
    assert scarcity_price > 0
    priced = replace(industry, human_cost_per_hour=industry.human_cost_per_hour + scarcity_price)
    work = PolicyOptimizer().solve(IndustryModel(priced), Scenario())
    assert work.surplus_per_work_hour == pytest.approx(0, abs=1e-8)
    assert work.policy.delegation_hours == pytest.approx(attention.policy.delegation_hours, rel=2e-5)
    assert work.policy.tokens_per_work_hour == pytest.approx(attention.policy.tokens_per_work_hour, rel=2e-5)


def test_scalar_attention_solution_matches_general_optimizer() -> None:
    settings = OptimizationSettings(
        max_tokens_per_work_hour=200.0,
        grid_points_per_dimension=11,
        local_starts=2,
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


def test_attention_shadow_price_capability_elasticity() -> None:
    settings = OptimizationSettings(max_tokens_per_work_hour=200.0)
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
        review_elasticity = model.verification_scope_elasticity(center.policy)
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
        OptimizationSettings(max_tokens_per_work_hour=200.0)
    )
    multiplier = 0.4

    # Exercise the general optimizer used by the comparative-statics notebook.
    # The scalar solver omits the uniform multiplier by construction, so using
    # it here would not catch a regression in the production search path.
    baseline = optimizer.solve(model, Scenario(verification_time_multiplier=1.0))
    faster = optimizer.solve(
        model, Scenario(verification_time_multiplier=multiplier)
    )

    assert math.isclose(
        faster.policy.delegation_hours,
        baseline.policy.delegation_hours,
        rel_tol=1e-6,
    )
    assert math.isclose(
        faster.policy.tokens_per_work_hour,
        baseline.policy.tokens_per_work_hour,
        rel_tol=1e-6,
    )
    assert faster.attention_limited_tokens is not None
    assert baseline.attention_limited_tokens is not None
    assert math.isclose(
        faster.attention_limited_tokens,
        baseline.attention_limited_tokens / multiplier,
        rel_tol=1e-6,
    )


@pytest.mark.parametrize("optimizer_type", [PolicyOptimizer, AttentionConstrainedOptimizer])
def test_interior_policy_satisfies_economic_marginal_conditions(optimizer_type) -> None:
    industry, scenario = REFERENCE_INDUSTRY, Scenario()
    outcome = optimizer_type().solve(IndustryModel(industry), scenario)
    s, x = outcome.policy.delegation_hours, outcome.policy.tokens_per_work_hour
    frontier = (s / industry.capability_horizon_hours) ** industry.capability_shape
    execution = s / (industry.execution_scale * x ** industry.inference_returns)
    benefit = industry.value_per_work_hour * outcome.success_probability
    review = outcome.verification_hours_per_chunk
    elasticity = IndustryModel(industry).verification_scope_elasticity(outcome.policy)
    assert scenario.token_price * x == pytest.approx(
        industry.inference_returns * execution * benefit, rel=1e-5
    )
    reliability_loss = benefit * (industry.capability_shape * frontier + execution)
    if optimizer_type is PolicyOptimizer:
        scope_benefit = industry.human_cost_per_hour * review / s * (1 - elasticity)
    else:
        scope_benefit = (benefit - scenario.token_price * x) * (1 - elasticity)
    assert reliability_loss == pytest.approx(scope_benefit, rel=1e-5)


def test_attention_endowment_scales_output_without_changing_policy() -> None:
    optimizer = AttentionConstrainedOptimizer()
    low = replace(REFERENCE_INDUSTRY, human_attention_hours=100_000)
    high = replace(low, human_attention_hours=300_000)
    first = optimizer.solve(IndustryModel(low), Scenario())
    second = optimizer.solve(IndustryModel(high), Scenario())
    assert first.policy == second.policy
    assert second.attention_limited_tokens == pytest.approx(3 * first.attention_limited_tokens)


@pytest.mark.parametrize("optimizer_type, demand", [
    (PolicyOptimizer, "work_limited_tokens"),
    (AttentionConstrainedOptimizer, "attention_limited_tokens"),
])
def test_efficiency_equals_an_effective_price_cut_and_token_rescaling(optimizer_type, demand) -> None:
    industry = replace(REFERENCE_INDUSTRY, human_attention_hours=100_000)
    model = IndustryModel(industry)
    optimizer = optimizer_type(OptimizationSettings(max_tokens_per_work_hour=200))
    efficient = optimizer.solve(model, Scenario(
        model_capability=1.7, token_efficiency=4,
        token_price=1.3, verification_time_multiplier=.6,
    ))
    cheaper = optimizer.solve(model, Scenario(
        model_capability=1.7, token_efficiency=1,
        token_price=1.3 / 4, verification_time_multiplier=.6,
    ))
    assert efficient.policy.delegation_hours == pytest.approx(cheaper.policy.delegation_hours, rel=1e-5)
    assert efficient.success_probability == pytest.approx(cheaper.success_probability, rel=1e-5)
    assert efficient.adoption_share == pytest.approx(cheaper.adoption_share, rel=1e-5)
    assert getattr(efficient, demand) == pytest.approx(getattr(cheaper, demand) / 4, rel=1e-5)


def test_work_jevons_condition_decomposes_adoption_and_effort_responses() -> None:
    model = IndustryModel(REFERENCE_INDUSTRY)
    optimizer = PolicyOptimizer(OptimizationSettings(max_tokens_per_work_hour=200))
    epsilon = 1e-4
    center = optimizer.solve(model, Scenario(token_price=1))
    cheaper = optimizer.solve(model, Scenario(token_price=math.exp(-epsilon)))
    dearer = optimizer.solve(model, Scenario(token_price=math.exp(epsilon)))

    effort_elasticity = math.log(
        dearer.policy.tokens_per_work_hour / cheaper.policy.tokens_per_work_hour
    ) / (2 * epsilon)
    revenue_elasticity = math.log(
        (math.exp(epsilon) * dearer.work_limited_tokens)
        / (math.exp(-epsilon) * cheaper.work_limited_tokens)
    ) / (2 * epsilon)
    adoption_response = (
        (1 - center.adoption_share)
        * center.policy.tokens_per_work_hour
        / REFERENCE_INDUSTRY.adoption_scale
    )

    assert adoption_response == pytest.approx(.38749, rel=2e-4)
    assert effort_elasticity == pytest.approx(-.77, abs=.003)
    assert revenue_elasticity == pytest.approx(
        1 - adoption_response + effort_elasticity, abs=2e-4
    )
    assert adoption_response + abs(effort_elasticity) > 1
    assert revenue_elasticity < 0  # A small price cut raises spending locally.
