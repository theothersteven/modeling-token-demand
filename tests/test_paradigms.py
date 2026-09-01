"""Economic behavior checks for the configuration-only scenario gallery."""

from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pytest

from modeling_token_demand import AttentionConstrainedOptimizer, IndustryModel, PolicyOptimizer, Scenario
from modeling_token_demand.calibrations import (
    CAPABILITY_VALLEY, CONCENTRATED_ADOPTION, EARLY_SATURATION,
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
    assert len(illustrative_industries()) == 14
    assert len(illustrative_industries(include_singletons=False)) == 11
    assert len(singleton_industries()) == 3
    assert CONCENTRATED_ADOPTION == HIGH_ADOPTION_CONCENTRATION
    assert CONCENTRATED_ADOPTION not in singleton_industries()
    assert set(work_paradigms() + attention_paradigms()) <= set(illustrative_industries())
    assert len(work_paradigms()) == len(attention_paradigms()) == 3
    assert EARLY_SATURATION.adoption_location == CONCENTRATED_ADOPTION.adoption_location
    assert EARLY_SATURATION.adoption_scale == CONCENTRATED_ADOPTION.adoption_scale


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
    assert all(model.adoption_share(0) == 0 for model in models)
    # Conditioning on positive hurdles slightly shifts the median; the spread
    # comparison still reverses around the adoption transition.
    below = [model.adoption_share(location - 1) for model in models]
    above = [model.adoption_share(location + 1) for model in models]
    assert below[0] < below[1] < below[2]
    assert above[0] > above[1] > above[2]


def test_value_and_concentration_use_the_requested_geometric_steps():
    assert [industry.value_per_work_hour for industry in (
        LOW_ECONOMIC_VALUE, REFERENCE_INDUSTRY, HIGH_ECONOMIC_VALUE)] == [50, 100, 200]
    assert [industry.adoption_scale for industry in (
        HIGH_ADOPTION_CONCENTRATION, REFERENCE_INDUSTRY, LOW_ADOPTION_CONCENTRATION)] == [1, 4, 16]
    assert all(industry.adoption_location == 76 for industry in illustrative_industries())


def test_parameter_row_tables_match_the_calibrations():
    block = calibration_tables_markdown()
    headers = [line for line in block.splitlines() if line.startswith("| Parameter |")]
    assert len(headers) == 3
    assert all("Reference" in line for line in headers)
    assert "Value low | Value high | Concentration low | Concentration high" in headers[1]
    assert "| Work value $b$ (dollars) | 100 | **50** | **200** | 100 | 100 |" in block


def test_zero_overhead_removes_the_capability_valley():
    industry = replace(CAPABILITY_VALLEY, verification_fixed_hours=0,
                       human_attention_hours=100_000)
    optimizer = AttentionConstrainedOptimizer(gallery_settings())
    outcomes = [optimizer.solve(IndustryModel(industry), Scenario(model_capability=m))
                for m in (1, 2, 4)]
    demand = [outcome.attention_limited_tokens for outcome in outcomes]
    assert not boundary_hits(outcomes, gallery_settings())
    assert demand[1] / demand[0] == pytest.approx(2 ** (1 - industry.verification_elasticity), rel=1e-4)
    assert demand[2] / demand[0] == pytest.approx(4 ** (1 - industry.verification_elasticity), rel=1e-4)


@pytest.fixture(scope="module")
def work_outcomes():
    model = IndustryModel(CONCENTRATED_ADOPTION)
    optimizer = PolicyOptimizer(gallery_settings())
    prices = {price: optimizer.solve(model, Scenario(token_price_per_million=price))
              for price in (1, 2, 5, 10, 20)}
    efficient = {eta: optimizer.solve(model, Scenario(token_efficiency=eta)) for eta in (2, 5, 10)}
    return prices, efficient


def test_concentrated_adoption_unlocks_large_market_and_revenue(work_outcomes):
    prices, _ = work_outcomes
    assert .2 < prices[10].adoption_share < .5
    assert prices[5].adoption_share > .75
    assert prices[1].adoption_share > .95
    ratio = prices[5].work_limited_tokens / prices[10].work_limited_tokens
    assert ratio > 2  # Adoption expansion more than offsets the price cut.
    assert ratio / 2 > 1.5
    assert prices[1].work_limited_tokens / prices[10].work_limited_tokens > 10
    assert not boundary_hits(list(prices.values()), gallery_settings())
    demand = [prices[p].work_limited_tokens for p in sorted(prices)]
    assert np.all(np.diff(demand) <= 0)


def test_efficiency_rebound_is_real_not_just_a_price_plot(work_outcomes):
    prices, efficient = work_outcomes
    for eta, outcome in efficient.items():
        # z = eta*x maps a technology improvement to an effective price cut.
        # Interior solutions should reproduce this identity independently.
        assert outcome.work_limited_tokens == pytest.approx(
            prices[10 / eta].work_limited_tokens / eta, rel=1e-5)
    assert efficient[5].work_limited_tokens > prices[10].work_limited_tokens * 1.3


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
    assert main["audit"]["independent_checks"] >= 250
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
    assert indexed[HIGH_ADOPTION_CONCENTRATION.name, "work", "efficiency"]["shape"] == "hump"
    valley = indexed[CAPABILITY_VALLEY.name, "attention", "capability"]
    assert valley["shape"] == "U-shape"
    assert all(np.diff(valley["attention_value"]) > 0)
    offset = indexed["Offsetting efficiency", "attention", "efficiency"]
    assert max(offset["demand"]) / min(offset["demand"]) < 1.10
