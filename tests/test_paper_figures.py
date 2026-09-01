"""The six paper figures must preserve audited quantities and dollar units."""

import json
from pathlib import Path

import numpy as np
import pytest

from modeling_token_demand.paper_figures import ATTENTION_CASES, AXES, WORK_CASES


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def data():
    plots = json.loads((ROOT / "figures/interactive.json").read_text())["plots"]
    curves = json.loads((ROOT / "figures/paradigms.json").read_text())["main"]["curves"]
    return plots, curves


@pytest.mark.parametrize(
    "regime,cases",
    (("work", WORK_CASES), ("attention", ATTENTION_CASES)),
)
@pytest.mark.parametrize("axis", ("capability", "efficiency", "price"))
def test_pair_figures_show_absolute_token_demand_and_spending(data, regime, cases, axis):
    plots, curves = data
    panels = plots[f"{regime}-{axis}-demand-spending.png"]["panels"]
    assert len(panels) == 2
    assert panels[0]["ylabel"] == "Token demand (trillion tokens)"
    assert panels[1]["ylabel"] == "Token spending (USD millions)"
    assert panels[0]["yscale"] == panels[1]["yscale"] == "log"
    expected_labels = {case[1] for case in cases}
    assert {line["name"] for line in panels[0]["lines"]} == expected_labels
    assert {line["name"] for line in panels[1]["lines"]} == expected_labels

    lower, upper = AXES[axis]["limits"]
    for name, label, _, _ in cases:
        record = next(
            curve for curve in curves
            if curve["industry"]["name"] == name
            and curve["regime"] == regime
            and curve["axis"] == axis
        )
        x_values = np.asarray(record["values"])
        selected = (x_values >= lower) & (x_values <= upper)
        demand = np.asarray(record["demand"])[selected]
        prices = x_values[selected] if axis == "price" else np.full(selected.sum(), 10.0)
        demand_line = next(line for line in panels[0]["lines"] if line["name"] == label)
        spending_line = next(line for line in panels[1]["lines"] if line["name"] == label)
        assert demand_line["x"] == pytest.approx(x_values[selected])
        assert spending_line["x"] == pytest.approx(x_values[selected])
        assert demand_line["y"] == pytest.approx(demand / 1e12)
        assert spending_line["y"] == pytest.approx(demand * prices / 1e12)

    reverse = panels[0]["xlim"][0] > panels[0]["xlim"][1]
    assert reverse == AXES[axis]["reverse"]


def test_work_and_attention_figures_use_different_industry_mechanisms(data):
    plots, _ = data
    work = {line["name"] for line in plots["work-capability-demand-spending.png"]["panels"][0]["lines"]}
    attention = {line["name"] for line in plots["attention-capability-demand-spending.png"]["panels"][0]["lines"]}
    assert work == {"Gradual adoption", "Clustered adoption", "Early saturation"}
    assert attention == {"Reusable review", "Balanced review", "Proportional review"}
