"""Economic behavior checks for the configuration-only scenario gallery."""

from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pytest

from modeling_token_demand import AttentionConstrainedOptimizer, IndustryModel, PolicyOptimizer, Scenario
from modeling_token_demand.calibrations import (
    CAPABILITY_VALLEY, CONCENTRATED_ADOPTION, EARLY_SATURATION,
    HARD_EXECUTION, HIGH_ADOPTION_HURDLE, HIGH_CAPABILITY_REQUIREMENT,
    LOW_ADOPTION_HURDLE, LOW_INFERENCE_RETURNS,
    PROPORTIONAL_REVIEW,
    SLOW_REVIEW_GROWTH,
    HIGH_ADOPTION_CONCENTRATION, LOW_ADOPTION_CONCENTRATION, REFERENCE_INDUSTRY,
    HIGH_ECONOMIC_VALUE, LOW_ECONOMIC_VALUE, calibration_tables_markdown,
    attention_paradigms, illustrative_industries, singleton_industries, work_paradigms,
)
from modeling_token_demand.paradigms import boundary_hits, gallery_settings, shape


@pytest.mark.parametrize("values, expected", [
    ([1, 2, 3], "rising"), ([3, 2, 1], "falling"),
    ([1, 3, 2], "hump"), ([3, 1, 2], "U-shape"),
    ([3, 1, 2, 1], "fall-rise-fall"),
    ([1, 1.02, .99, 1.01], "approximately flat"),
])
def test_shape_detects_material_not_tiny_reversals(values, expected):
    assert shape(values) == expected


@pytest.mark.parametrize("values", ([1], [0, 1], [-1, 2], [1, float("nan")]))
def test_shape_rejects_invalid_inputs(values):
    with pytest.raises(ValueError):
        shape(values)


def test_focus_views_are_subsets_of_the_main_configuration_table():
    assert len(illustrative_industries()) == 8
    assert len(illustrative_industries(include_singletons=True)) == 11
    assert len(singleton_industries()) == 3
    assert CONCENTRATED_ADOPTION == HIGH_ADOPTION_CONCENTRATION
    assert CONCENTRATED_ADOPTION not in singleton_industries()
    assert set(work_paradigms() + attention_paradigms()) <= set(illustrative_industries())
    assert len(work_paradigms()) == 5
    assert len(attention_paradigms()) == 5
    assert work_paradigms() == (
        REFERENCE_INDUSTRY, LOW_ADOPTION_HURDLE, HIGH_ADOPTION_HURDLE,
        HARD_EXECUTION, HIGH_CAPABILITY_REQUIREMENT,
    )
    assert attention_paradigms() == (
        REFERENCE_INDUSTRY, HARD_EXECUTION, LOW_INFERENCE_RETURNS,
        SLOW_REVIEW_GROWTH, PROPORTIONAL_REVIEW,
    )


def test_every_active_alternative_changes_exactly_one_industry_parameter():
    reference = asdict(REFERENCE_INDUSTRY)
    expected = {
        LOW_ADOPTION_HURDLE.name: {"name", "adoption_location"},
        HIGH_ADOPTION_HURDLE.name: {"name", "adoption_location"},
        HARD_EXECUTION.name: {"name", "execution_scale"},
        HIGH_CAPABILITY_REQUIREMENT.name: {"name", "capability_horizon_hours"},
        LOW_INFERENCE_RETURNS.name: {"name", "inference_returns"},
        SLOW_REVIEW_GROWTH.name: {"name", "verification_elasticity"},
        PROPORTIONAL_REVIEW.name: {"name", "verification_elasticity"},
    }
    for industry in illustrative_industries():
        if industry == REFERENCE_INDUSTRY:
            continue
        changed = {
            field for field, value in asdict(industry).items()
            if value != reference[field]
        }
        assert changed == expected[industry.name]


def test_concentration_pair_changes_only_spread_and_has_crossing_cdfs():
    reference = asdict(REFERENCE_INDUSTRY)
    for industry in (HIGH_ADOPTION_CONCENTRATION, LOW_ADOPTION_CONCENTRATION):
        difference = {key for key, value in asdict(industry).items() if value != reference[key]}
        assert difference == {"name", "adoption_scale"}
    assert HIGH_ADOPTION_CONCENTRATION.adoption_scale < REFERENCE_INDUSTRY.adoption_scale \
        < LOW_ADOPTION_CONCENTRATION.adoption_scale
    models = [IndustryModel(industry) for industry in (
        HIGH_ADOPTION_CONCENTRATION, REFERENCE_INDUSTRY, LOW_ADOPTION_CONCENTRATION)]
    location = REFERENCE_INDUSTRY.adoption_location
    assert all(model.adoption_share(location) == pytest.approx(.5) for model in models)
    # Logistic CDFs with a common location cross exactly at their median.
    below = [model.adoption_share(location - 1) for model in models]
    above = [model.adoption_share(location + 1) for model in models]
    assert below[0] < below[1] < below[2]
    assert above[0] > above[1] > above[2]


def test_hurdle_cases_order_the_extensive_margin_around_the_reference():
    assert [industry.value_per_work_hour for industry in (
        LOW_ECONOMIC_VALUE, REFERENCE_INDUSTRY, HIGH_ECONOMIC_VALUE)] == [50, 100, 200]
    assert [industry.adoption_scale for industry in (
        HIGH_ADOPTION_CONCENTRATION, REFERENCE_INDUSTRY, LOW_ADOPTION_CONCENTRATION)] == [1, 4, 16]
    assert [industry.adoption_location for industry in work_paradigms()] == [81, 40, 95, 81, 81]
    assert all(industry.adoption_scale == 4 for industry in work_paradigms())


def test_parameter_row_tables_match_the_calibrations():
    block = calibration_tables_markdown()
    headers = [line for line in block.splitlines() if line.startswith("| Plot line |")]
    assert len(headers) == 2
    assert all("$\\lambda$" in line and "$\\sigma$" in line for line in headers)
    assert "| Low adoption hurdle | 12 | 1.25 | 4 |" in block
    assert "| Hard execution | 12 | 1.25 | **1** |" in block
    assert "| High capability requirement | **3** | 1.25 |" in block
    assert "| Low inference returns | 12 | 1.25 | 4 | **0.2** |" in block
    assert "| Slow-growing review |" in block and "**0.15**" in block
    assert "| Nearly proportional review |" in block and "**0.95**" in block


def test_zero_overhead_makes_shifted_review_prefer_the_smallest_scope():
    industry = replace(CAPABILITY_VALLEY, verification_fixed_hours=0,
                       human_attention_hours=100_000)
    optimizer = AttentionConstrainedOptimizer(gallery_settings())
    outcomes = [optimizer.solve(IndustryModel(industry), Scenario(model_capability=m))
                for m in (1, 2, 4)]
    assert boundary_hits(outcomes, gallery_settings()) == ["s"]
    assert all(outcome.policy.delegation_hours == pytest.approx(
        gallery_settings().min_delegation_hours
    ) for outcome in outcomes)


def test_economic_effort_floor_is_not_reported_as_a_numerical_bound_hit():
    model = IndustryModel(REFERENCE_INDUSTRY)
    settings = gallery_settings()
    outcome = PolicyOptimizer(settings).solve(
        model, Scenario(model_capability=30)
    )
    assert outcome.policy.tokens_per_work_hour == pytest.approx(1)
    assert boundary_hits([outcome], settings) == []


@pytest.fixture(scope="module")
def work_outcomes():
    model = IndustryModel(CONCENTRATED_ADOPTION)
    optimizer = PolicyOptimizer(gallery_settings())
    prices = {price: optimizer.solve(model, Scenario(token_price=price))
              for price in (.1, .2, .5, 1, 2)}
    efficient = {eta: optimizer.solve(model, Scenario(token_efficiency=eta)) for eta in (2, 5, 10)}
    return prices, efficient


def test_concentrated_adoption_unlocks_large_market_and_revenue(work_outcomes):
    prices, _ = work_outcomes
    assert .2 < prices[1].adoption_share < .5
    assert prices[.5].adoption_share > .75
    assert prices[.1].adoption_share > .95
    ratio = prices[.5].work_limited_tokens / prices[1].work_limited_tokens
    assert ratio > 2  # Adoption expansion more than offsets the price cut.
    assert ratio / 2 > 1.5
    assert prices[.1].work_limited_tokens / prices[1].work_limited_tokens > 10
    assert not boundary_hits(list(prices.values()), gallery_settings())
    demand = [prices[p].work_limited_tokens for p in sorted(prices)]
    assert np.all(np.diff(demand) <= 0)


def test_efficiency_rebound_is_real_not_just_a_price_plot(work_outcomes):
    prices, efficient = work_outcomes
    for eta, outcome in efficient.items():
        # z = eta*x maps a technology improvement to an effective price cut.
        # Interior solutions should reproduce this identity independently.
        assert outcome.work_limited_tokens == pytest.approx(
            prices[1 / eta].work_limited_tokens / eta, rel=1e-5)
    assert efficient[5].work_limited_tokens > prices[1].work_limited_tokens * 1.3


@pytest.mark.parametrize("field, factor", [
    ("verification_fixed_hours", .9), ("verification_fixed_hours", 1.1),
    ("inference_returns", .9), ("inference_returns", 1.1),
    ("capability_horizon_hours", .9), ("capability_horizon_hours", 1.1),
])
def test_capability_valley_survives_nearby_parameters(field, factor):
    industry = replace(CAPABILITY_VALLEY, human_attention_hours=100_000,
                       **{field: getattr(CAPABILITY_VALLEY, field) * factor})
    model = IndustryModel(industry)
    optimizer = AttentionConstrainedOptimizer(gallery_settings())
    outcomes = [optimizer.solve_interior(model, Scenario(model_capability=m)) for m in (.1, 2, 30)]
    demand = [o.attention_limited_tokens for o in outcomes]
    assert demand[0] > demand[1] * 1.08
    assert demand[-1] > demand[1] * 1.08
    assert all(o.surplus_per_attention_hour > 0 for o in outcomes)
    assert np.all(np.diff([o.surplus_per_attention_hour for o in outcomes]) > 0)
    assert not boundary_hits(outcomes, gallery_settings())


def test_generated_gallery_matches_configuration_and_model_accounting():
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "figures/paradigms.json").read_text())
    assert report['settings'] == asdict(gallery_settings())
    assert report['audit']['boundary_hits'] == []
    assert report['audit']['independent_checks'] >= 36
    assert report['audit']['max_relative_objective_error'] < 1e-8
    expected = {i.name: i for i in (*work_paradigms(), *attention_paradigms())}
    for curve in report['curves']:
        config = expected[curve['industry']['name']]
        if curve['regime'] == 'attention':
            config = replace(config, human_attention_hours=100_000)
        assert curve['industry'] == asdict(config)
        demand = np.array(curve['demand'])
        assert demand == pytest.approx(np.array(curve['assigned_work']) *
                                       np.array(curve['tokens_per_assigned_work']))
        if curve['regime'] == 'work':
            assert curve['assigned_work'] == pytest.approx(
                np.array(curve['adoption']) * config.potential_work_hours)
        else:
            assert demand == pytest.approx(100_000 * np.array(curve['leverage']) *
                                           np.array(curve['x']))


def test_main_curves_include_all_cases_and_audited_paradigms():
    root = Path(__file__).resolve().parents[1]
    main = json.loads((root / "figures/paradigms.json").read_text())["main"]
    assert main["audit"]["independent_checks"] >= 90
    assert main["audit"]["max_relative_objective_error"] < 1e-8
    assert main["audit"]["boundary_hits"] == []
    cases = {industry.name: asdict(replace(industry, human_attention_hours=100_000))
             for industry in illustrative_industries()}
    assert len(main["curves"]) == len(cases) * 6
    indexed = {}
    for curve in main["curves"]:
        assert curve["industry"] == cases[curve["industry"]["name"]]
        indexed[curve["industry"]["name"], curve["regime"], curve["axis"]] = curve
        assert curve["demand"] == pytest.approx(np.array(curve["assigned_work"]) *
                                               np.array(curve["tokens_per_assigned_work"]))
    assert indexed[LOW_ADOPTION_HURDLE.name, "work", "capability"]["shape"] == "falling"
    assert indexed[REFERENCE_INDUSTRY.name, "work", "capability"]["shape"] == "hump"
    assert indexed[HIGH_ADOPTION_HURDLE.name, "work", "capability"]["shape"] == "rising"
    assert indexed[LOW_ADOPTION_HURDLE.name, "work", "efficiency"]["shape"] == "falling"
    assert indexed[HIGH_ADOPTION_HURDLE.name, "work", "efficiency"]["shape"] == "rising"
    assert indexed[HARD_EXECUTION.name, "attention", "efficiency"]["shape"] == "rising"
    assert indexed[HARD_EXECUTION.name, "work", "efficiency"]["shape"] == "rising"
    assert indexed[HIGH_CAPABILITY_REQUIREMENT.name, "work", "capability"]["shape"] == "hump"
    assert indexed[HIGH_CAPABILITY_REQUIREMENT.name, "work", "efficiency"]["shape"] == "hump"
    assert indexed[LOW_INFERENCE_RETURNS.name, "attention", "efficiency"]["shape"] == "approximately flat"
    assert indexed[SLOW_REVIEW_GROWTH.name, "attention", "efficiency"]["shape"] == "falling"
    assert indexed[PROPORTIONAL_REVIEW.name, "attention", "capability"]["shape"] == "hump"

    # The price quantity itself rises as price falls. Spending can rise, fall,
    # or peak depending on whether demand crosses the constant-revenue line.
    spending_shapes = {}
    for industry in work_paradigms():
        curve = indexed[industry.name, "work", "price"]
        spending = np.asarray(curve["demand"])[::-1] * np.asarray(curve["values"])[::-1]
        spending_shapes[industry.name] = shape(spending)
    assert spending_shapes == {
        LOW_ADOPTION_HURDLE.name: "falling",
        REFERENCE_INDUSTRY.name: "hump",
        HIGH_ADOPTION_HURDLE.name: "rising",
        HARD_EXECUTION.name: "rising",
        HIGH_CAPABILITY_REQUIREMENT.name: "hump",
    }


def test_published_effort_floor_solutions_satisfy_the_kkt_inequality():
    root = Path(__file__).resolve().parents[1]
    curves = json.loads(
        (root / "figures/paradigms.json").read_text()
    )["main"]["curves"]
    bindings = 0
    for curve in curves:
        industry = curve["industry"]
        for value, scope, effort, success in zip(
            curve["values"], curve["s"], curve["x"], curve["success"]
        ):
            assert effort >= 1
            if effort > 1.0001:
                continue
            bindings += 1
            capability = value if curve["axis"] == "capability" else 1
            efficiency = value if curve["axis"] == "efficiency" else 1
            price = value if curve["axis"] == "price" else 1
            execution_exponent = scope / (
                industry["execution_scale"]
                * capability
                * (efficiency * effort) ** industry["inference_returns"]
            )
            marginal_value = (
                industry["value_per_work_hour"]
                * success
                * industry["inference_returns"]
                * execution_exponent
                / effort
            )
            # At a lower-bound optimum, increasing effort cannot add more
            # value than its token price.
            assert marginal_value <= price + 1e-6
    assert bindings > 0
