"""Publication views must preserve audited outcomes and economic denominators."""

import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def data():
    return tuple(json.loads((ROOT / f"figures/{name}.json").read_text())
                 for name in ("interactive", "paradigms", "interventions"))


WORK_NAMES = {"Reference": "Reference industry", "Concentrated adoption": "Adoption concentration: high",
              "Early saturation": "Early saturation"}
ATTENTION_NAMES = {"Review grows slowly": "Verification burden: low",
                   "Review nearly proportional": "Verification burden: high"}


def test_price_view_separates_adoption_purchases_and_spending(data):
    plots, comparisons, _ = data
    panels = plots["plots"]["price-adoption-and-spending.png"]["panels"]
    for line in panels[0]["lines"]:
        record = next(r for r in comparisons["curves"] if r["axis"] == "price"
                      and r["industry"]["name"] == WORK_NAMES[line["name"]])
        points = [record["values"].index(value) for value in line["x"]]
        assert line["x"] == [value for value in record["values"] if value <= 20]
        assert line["y"] == pytest.approx(100 * np.asarray(record["adoption"])[points])
        purchase = next(l for l in panels[1]["lines"] if l["name"] == line["name"])
        spending = next(l for l in panels[2]["lines"] if l["name"] == line["name"])
        assert purchase["x"] == spending["x"] == line["x"]
        expected = np.asarray(record["demand"])[points] / record["demand"][record["baseline_index"]]
        assert purchase["y"] == pytest.approx(expected)
        assert spending["y"] == pytest.approx(expected * np.asarray(line["x"]) / 10)
    assert all(panel["xlim"] == [20, 1] for panel in panels)


@pytest.mark.parametrize("regime, names", [("work", WORK_NAMES), ("attention", ATTENTION_NAMES)])
def test_capability_panels_reconstruct_demand_from_its_two_margins(data, regime, names):
    plots, comparisons, _ = data
    panels = plots["plots"][f"capability-{regime}-decomposition.png"]["panels"]
    assert {line["name"] for line in panels[0]["lines"]} == set(names)
    for label, name in names.items():
        record = next(r for r in comparisons["curves"] if r["axis"] == "capability"
                      and r["regime"] == regime and r["industry"]["name"] == name)
        lines = [next(line for line in panel["lines"] if line["name"] == label) for panel in panels]
        assert all(line["x"] == record["values"] for line in lines)
        baseline = record["baseline_index"]
        activity = np.asarray(record["adoption"] if regime == "work" else record["leverage"])
        expected = activity * 100 if regime == "work" else activity / activity[baseline]
        assert lines[0]["y"] == pytest.approx(expected)
        assert lines[1]["y"] == pytest.approx(np.asarray(record["x"]) / record["x"][baseline])
        assert lines[2]["y"] == pytest.approx(np.asarray(record["demand"]) / record["demand"][baseline])
        indexed_activity = np.asarray(lines[0]["y"]) / lines[0]["y"][baseline]
        assert lines[2]["y"] == pytest.approx(indexed_activity * np.asarray(lines[1]["y"]))


def test_review_view_preserves_speed_direction_and_distinct_outcomes(data):
    plots, _, interventions = data
    panels = plots["plots"]["verification-expansion.png"]["panels"]
    for panel, regime in zip(panels, ("work", "attention")):
        record = next(r for r in interventions["curves"]
                      if r["experiment"] == "verification-speed" and r["regime"] == regime)
        assert panel["xlim"] == [.5, 10]
        for line in panel["lines"]:
            metric = "demand" if line["name"] == "Tokens" else "completed_work"
            raw = np.asarray(record[metric])
            assert line["x"] == pytest.approx(1 / np.asarray(record["values"]))
            assert line["y"] == pytest.approx(raw / raw[record["baseline_index"]])
            if regime == "attention":
                assert line["x"] == pytest.approx(line["y"])
