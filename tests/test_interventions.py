"""Check the question-led figures against policies and economic accounting."""

import json
from pathlib import Path

import numpy as np
import pytest

from modeling_token_demand.interventions import configuration
from modeling_token_demand.model import IndustryModel, Policy


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def report():
    return json.loads((ROOT / "figures/interventions.json").read_text())


def test_intervention_outcomes_reproduce_model_and_distinguish_adoption(report):
    assert report["model"] == "single_attempt"
    assert len(report["curves"]) == 16
    assert report["audit"]["boundary_hits"] == []
    assert report["audit"]["independent_checks"] >= 48
    assert report["audit"]["max_relative_objective_error"] < 1e-8
    for curve in report["curves"]:
        assert curve["demand"] == pytest.approx(
            np.array(curve["assigned_work"]) * curve["tokens_per_assigned_work"])
        assert curve["completed_work"] == pytest.approx(
            np.array(curve["assigned_work"]) * curve["success"])
        for index in {0, curve["baseline_index"], len(curve["values"]) - 1}:
            industry, scenario = configuration(curve, curve["values"][index])
            policy = Policy(curve["s"][index], curve["x"][index])
            outcome = IndustryModel(industry).evaluate(policy, scenario)
            if curve["regime"] == "work":
                expected = outcome.work_limited_tokens
                assert curve["adoption"][index] == pytest.approx(outcome.adoption_share)
                assert curve["completed_work"][index] <= curve["assigned_work"][index] \
                    <= industry.potential_work_hours
            else:
                expected = outcome.attention_limited_tokens
                assert curve["adoption"] is None  # no market adoption rate in this regime
                assert outcome.surplus_per_attention_hour > 0
                assert curve["assigned_work"][index] / policy.delegation_hours \
                    * outcome.verification_hours_per_chunk == pytest.approx(100_000)
            assert curve["demand"][index] == pytest.approx(expected)


def test_uniform_verification_scales_throughput_without_changing_policy(report):
    curve = next(c for c in report["curves"]
                 if c["experiment"] == "verification-speed" and c["regime"] == "attention")
    for metric in ("demand", "assigned_work", "completed_work"):
        values = np.array(curve[metric])
        assert values / values[curve["baseline_index"]] == pytest.approx(1 / np.array(curve["values"]))
    for metric in ("s", "x", "success"):
        assert curve[metric] == pytest.approx([curve[metric][0]] * len(curve["values"]))


def test_harness_only_changes_feasibility_not_execution(report):
    curves = [c for c in report["curves"]
              if c["experiment"] == "harness-feasibility" and c["regime"] == "work"]
    # Same proportional frontier shift; only m also improves execution.
    base_industry, base_scenario = configuration(curves[0], 1)
    baseline = IndustryModel(base_industry)
    policy = Policy(4, 100_000)
    models = [IndustryModel(configuration(c, 5)[0]) for c in curves]
    scenarios = [configuration(c, 5)[1] for c in curves]
    assert models[0].capability_share(policy, scenarios[0]) == pytest.approx(
        models[1].capability_share(policy, scenarios[1]))
    assert models[0].conditional_success(policy, scenarios[0]) == baseline.conditional_success(policy, base_scenario)
    assert models[1].conditional_success(policy, scenarios[1]) > baseline.conditional_success(policy, base_scenario)


def test_exported_question_panels_use_correct_regimes_units_and_baselines(report):
    plots = json.loads((ROOT / "figures/interactive.json").read_text())["plots"]
    for key in {c["experiment"] for c in report["curves"]}:
        spec = plots[f"intervention-{key}.png"]
        assert spec["columns"] == 3 and len(spec["panels"]) == 6
        assert not spec["shared_y"]
        for index, panel in enumerate(spec["panels"]):
            regime = "work" if index < 3 else "attention"
            metric = ("demand", "assigned_work", "completed_work")[index % 3]
            for line in panel["lines"]:
                record = next(c for c in report["curves"] if c["experiment"] == key
                              and c["regime"] == regime and c["label"] == line["name"])
                assert line["x"] == record["values"]
                raw = np.array(record[metric])
                expected = 100 * raw / record["industry"]["potential_work_hours"] \
                    if index in (1, 2) else raw / raw[record["baseline_index"]]
                assert line["y"] == pytest.approx(expected)
            assert panel["yscale"] == "linear"
            assert (panel["xlim"][0] > panel["xlim"][1]) == (key in ("verification-speed", "review-growth"))


def test_manuscript_efficiency_table_uses_regenerated_endpoints(report):
    source = (ROOT / "README.md").read_text()
    for alpha in (.25, .5, .75):
        curves = [next(c for c in report["curves"]
                       if c["experiment"] == "efficiency-returns"
                       and c["industry"]["inference_returns"] == alpha
                       and c["regime"] == regime) for regime in ("work", "attention")]
        ratios = [c["demand"][c["values"].index(100)] / c["demand"][c["baseline_index"]]
                  for c in curves]
        assert f"| {alpha:.2f} | {ratios[0]:.2f} | {ratios[1]:.2f} |" in source
        assert all(c["completed_work"][-1] > c["completed_work"][c["baseline_index"]]
                   for c in curves)
        if alpha == .25:
            assert ratios[0] > 1  # Adoption can still outweigh savings at 100x efficiency.
        else:
            assert ratios[0] < 1
        assert ratios[1] < 1
