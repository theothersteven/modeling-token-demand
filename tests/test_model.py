from modeling_token_demand import (
    IndustryModel,
    OptimizationSettings,
    Policy,
    PolicyOptimizer,
    Scenario,
)
from modeling_token_demand.calibrations import SOFTWARE


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
    outcome = PolicyOptimizer(settings).solve(IndustryModel(SOFTWARE), Scenario())
    policy = outcome.policy

    assert settings.min_delegation_hours <= policy.delegation_hours <= settings.max_delegation_hours
    assert settings.min_tokens_per_work_hour <= policy.tokens_per_work_hour <= settings.max_tokens_per_work_hour
    assert 1 <= policy.max_attempts <= settings.max_attempts
    assert 0.0 <= outcome.adoption_share <= 1.0
