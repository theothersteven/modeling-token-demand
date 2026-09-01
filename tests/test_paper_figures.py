"""The six paper figures must preserve audited indexed quantities."""

import json
from pathlib import Path

import numpy as np
import pytest

from modeling_token_demand.paper_figures import (
    ATTENTION_CASES, AXES, LEVER_AXES, LEVER_CASES, WORK_CASES,
)


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
def test_pair_figures_show_indexed_outcomes_and_price_spending(data, regime, cases, axis):
    plots, curves = data
    panels = plots[f"{regime}-{axis}-demand-spending.png"]["panels"]
    mechanism_view = regime == "work" and axis == "efficiency"
    price_effort_view = regime == "work" and axis == "price"
    assert len(panels) == (4 if mechanism_view or price_effort_view else 2)
    expected_labels = {case[1] for case in cases}
    first_labels = {line["name"] for line in panels[0]["lines"]}
    if axis == "price":
        if price_effort_view:
            assert panels[0]["ylabel"] == "Potential work assigned to AI (%)"
            assert panels[1]["ylabel"] == "Token demand index (baseline = 1)"
            assert panels[2]["ylabel"] == "Model effort level, x"
            assert panels[3]["ylabel"] == "Token spending index (baseline = 1)"
            assert panels[0]["yscale"] == "linear"
            assert all(panel["yscale"] == "log" for panel in panels[1:])
            assert first_labels == expected_labels
            assert {line["name"] for line in panels[1]["lines"]} == (
                expected_labels | {"Constant revenue"}
            )
            assert {line["name"] for line in panels[2]["lines"]} == expected_labels
            assert {line["name"] for line in panels[3]["lines"]} == expected_labels
        else:
            assert panels[0]["ylabel"] == "Token demand index (baseline = 1)"
            assert panels[1]["ylabel"] == "Token spending index (baseline = 1)"
            assert panels[0]["yscale"] == panels[1]["yscale"] == "log"
            assert first_labels == expected_labels | {"Constant revenue"}
    elif regime == "work":
        assert panels[0]["ylabel"] == "Potential work assigned to AI (%)"
        assert panels[1]["ylabel"] == "Token demand index (baseline = 1)"
        assert panels[0]["yscale"] == "linear"
        assert panels[1]["yscale"] == "log"
        assert first_labels == expected_labels
    else:
        assert panels[0]["ylabel"] == "Supervisory leverage index (baseline = 1)"
        assert panels[1]["ylabel"] == "Token demand index (baseline = 1)"
        assert panels[0]["yscale"] == panels[1]["yscale"] == "log"
        assert first_labels == expected_labels
    second_labels = {line["name"] for line in panels[1]["lines"]}
    if mechanism_view:
        assert second_labels == expected_labels | {"Revenue neutral (price fixed)"}
        assert panels[2]["ylabel"] == "Model effort level, x"
        assert panels[3]["ylabel"] == r"Effective inference per work unit, $\eta x$"
        assert {line["name"] for line in panels[2]["lines"]} == expected_labels
        assert {line["name"] for line in panels[3]["lines"]} == expected_labels
    elif not price_effort_view:
        assert second_labels == expected_labels

    lower, upper = AXES[axis]["limits"]
    for name, label, _, _, _ in cases:
        record = next(
            curve for curve in curves
            if curve["industry"]["name"] == name
            and curve["regime"] == regime
            and curve["axis"] == axis
        )
        x_values = np.asarray(record["values"])
        selected = (x_values >= lower) & (x_values <= upper)
        demand = np.asarray(record["demand"])
        normalized_demand = demand / demand[record["baseline_index"]]
        first_line = next(line for line in panels[0]["lines"] if line["name"] == label)
        second_line = next(line for line in panels[1]["lines"] if line["name"] == label)
        assert first_line["x"] == pytest.approx(x_values[selected])
        assert second_line["x"] == pytest.approx(x_values[selected])
        if axis == "price":
            spending = demand * x_values
            normalized_spending = spending / spending[record["baseline_index"]]
            if price_effort_view:
                assert first_line["y"] == pytest.approx(
                    100 * np.asarray(record["adoption"])[selected]
                )
                assert second_line["y"] == pytest.approx(normalized_demand[selected])
                effort_line = next(
                    line for line in panels[2]["lines"] if line["name"] == label
                )
                spending_line = next(
                    line for line in panels[3]["lines"] if line["name"] == label
                )
                effort_level = np.asarray(record["tokens_per_assigned_work"])
                assert effort_line["x"] == pytest.approx(x_values[selected])
                assert effort_line["y"] == pytest.approx(effort_level[selected])
                assert spending_line["x"] == pytest.approx(x_values[selected])
                assert spending_line["y"] == pytest.approx(normalized_spending[selected])
            else:
                assert first_line["y"] == pytest.approx(normalized_demand[selected])
                assert second_line["y"] == pytest.approx(normalized_spending[selected])
        elif regime == "work":
            assert second_line["y"] == pytest.approx(normalized_demand[selected])
            assert first_line["y"] == pytest.approx(
                100 * np.asarray(record["adoption"])[selected]
            )
            if mechanism_view:
                effort_level = np.asarray(record["tokens_per_assigned_work"])
                effort_line = next(
                    line for line in panels[2]["lines"] if line["name"] == label
                )
                effective_line = next(
                    line for line in panels[3]["lines"] if line["name"] == label
                )
                assert effort_line["x"] == pytest.approx(x_values[selected])
                assert effort_line["y"] == pytest.approx(effort_level[selected])
                assert effective_line["x"] == pytest.approx(x_values[selected])
                assert effective_line["y"] == pytest.approx(
                    (x_values * effort_level)[selected]
                )
        else:
            assert second_line["y"] == pytest.approx(normalized_demand[selected])
            leverage = np.asarray(record["leverage"])
            assert first_line["y"] == pytest.approx(
                (leverage / leverage[record["baseline_index"]])[selected]
            )

    reverse = panels[0]["xlim"][0] > panels[0]["xlim"][1]
    assert reverse == AXES[axis]["reverse"]


def test_work_and_attention_figures_use_different_industry_mechanisms(data):
    plots, _ = data
    work = {line["name"] for line in plots["work-capability-demand-spending.png"]["panels"][0]["lines"]}
    attention = {line["name"] for line in plots["attention-capability-demand-spending.png"]["panels"][0]["lines"]}
    assert work == {
        "Low adoption hurdle", "Reference industry", "High adoption hurdle",
        "Hard execution", "High capability requirement",
    }
    assert attention == {
        "Reference industry", "Hard execution", "Low inference returns",
        "Slow-growing review", "Nearly proportional review",
    }


def test_attention_value_figure_shows_hourly_and_token_reservation_prices(data):
    plots, curves = data
    panels = plots["attention-capability-value.png"]["panels"]
    assert len(panels) == 2
    assert panels[0]["ylabel"] == (
        r"Hourly attention price, $\rho^*(m)$ (\$/review hour)"
    )
    assert panels[0]["yscale"] == "log"
    assert panels[1]["ylabel"] == (
        r"Reservation token price, $c_{\rm res}(m)/c_0$"
    )
    assert panels[1]["yscale"] == "log"

    lower, upper = 1, 5
    for name, label, _, _, _ in ATTENTION_CASES:
        record = next(
            curve for curve in curves
            if curve["industry"]["name"] == name
            and curve["regime"] == "attention"
            and curve["axis"] == "capability"
        )
        x_values = np.asarray(record["values"])
        selected = (x_values >= lower) & (x_values <= upper)
        level = next(line for line in panels[0]["lines"] if line["name"] == label)
        reservation = next(
            line for line in panels[1]["lines"] if line["name"] == label
        )
        gross_value = np.asarray(record["attention_value"])
        assert level["x"] == pytest.approx(x_values[selected])
        assert level["y"] == pytest.approx(gross_value[selected])
        reservation_capability = np.asarray(record["reservation_capability"])
        reservation_price = np.asarray(record["reservation_price"])
        reservation_selected = (
            (reservation_capability >= lower)
            & (reservation_capability <= upper)
        )
        assert reservation["x"] == pytest.approx(
            reservation_capability[reservation_selected]
        )
        assert reservation["y"] == pytest.approx(
            reservation_price[reservation_selected]
            / record["reservation_price_baseline"]
        )
        assert reservation["y"][0] == pytest.approx(1)
        assert np.all(np.diff(reservation["y"]) >= 0)
        assert record["reservation_x"][-1] == pytest.approx(1)
        assert record["reservation_effort_floor_bindings"] > 0
        assert record["reservation_max_relative_gap"] < 2e-6


def test_figure_one_marks_demand_maxima_on_adoption_curves(data):
    plots, curves = data
    panels = plots["work-capability-demand-spending.png"]["panels"]
    points = panels[0]["points"]
    assert panels[1]["points"] == []
    assert {point["name"] for point in points} == {case[1] for case in WORK_CASES}

    lower, upper = AXES["capability"]["limits"]
    for name, label, color, _, _ in WORK_CASES:
        record = next(
            curve for curve in curves
            if curve["industry"]["name"] == name
            and curve["regime"] == "work"
            and curve["axis"] == "capability"
        )
        values = np.asarray(record["values"])
        demand = np.asarray(record["demand"])
        adoption = 100 * np.asarray(record["adoption"])
        selected = (values >= lower) & (values <= upper)
        indices = np.flatnonzero(selected)
        maximum = indices[np.argmax(demand[selected])]
        point = next(point for point in points if point["name"] == label)
        assert point["kind"] == "demand-peak"
        assert point["marker"] == "star"
        assert point["color"] == color
        assert point["x"] == pytest.approx(values[maximum])
        assert point["y"] == pytest.approx(adoption[maximum])


@pytest.mark.parametrize("regime", ("work", "attention"))
def test_price_panel_restores_the_constant_revenue_reference(data, regime):
    plots, _ = data
    demand_panel_index = 1 if regime == "work" else 0
    demand_panel = plots[f"{regime}-price-demand-spending.png"]["panels"][
        demand_panel_index
    ]
    benchmark = next(line for line in demand_panel["lines"] if line["name"] == "Constant revenue")
    assert benchmark["y"] == pytest.approx([1 / price for price in benchmark["x"]])
    assert benchmark["color"] == "#000000"
    assert benchmark["dash"] == "dot"
    assert benchmark["marker"] is None
    assert benchmark["width"] < min(
        line["width"] for line in demand_panel["lines"]
        if line["name"] != "Constant revenue"
    )


def test_efficiency_panel_marks_fixed_price_revenue_neutrality(data):
    plots, _ = data
    panel = plots["work-efficiency-demand-spending.png"]["panels"][1]
    benchmark = next(
        line for line in panel["lines"]
        if line["name"] == "Revenue neutral (price fixed)"
    )
    assert benchmark["y"] == pytest.approx(np.ones(len(benchmark["x"])))
    assert benchmark["color"] == "#000000"
    assert benchmark["dash"] == "dot"


@pytest.mark.parametrize(
    "regime,cases",
    (("work", WORK_CASES), ("attention", ATTENTION_CASES)),
)
def test_industry_lines_use_color_and_marker_identity(data, regime, cases):
    plots, _ = data
    lines = plots[f"{regime}-capability-demand-spending.png"]["panels"][0]["lines"]
    by_name = {line["name"]: line for line in lines}

    reference = by_name["Reference industry"]
    assert reference["color"] == "#595959"
    assert reference["dash"] == "solid"
    assert reference["marker"] is None

    assert [case[2] for case in cases] == [
        "#595959", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    ]
    assert len({case[4] for case in cases if case[4]}) == len(cases) - 1

    for _, label, color, style, marker in cases:
        line = by_name[label]
        if label == "Reference industry":
            continue
        assert line["color"] == color
        assert line["dash"] == "dash"
        assert style == "--"
        assert line["marker"] is not None
        assert marker is not None


@pytest.mark.parametrize("experiment", tuple(LEVER_AXES))
def test_capability_lever_figures_show_adoption_and_indexed_revenue(experiment):
    plots = json.loads((ROOT / "figures/interactive.json").read_text())["plots"]
    records = json.loads((ROOT / "figures/interventions.json").read_text())["curves"]
    spec = LEVER_AXES[experiment]
    panels = plots[spec["filename"]]["panels"]
    assert len(panels) == 2
    assert panels[0]["ylabel"] == "Potential work assigned to AI (%)"
    assert panels[0]["yscale"] == "linear"
    assert panels[1]["ylabel"] == "Token revenue index (baseline = 1)"
    assert panels[1]["yscale"] == "log"
    assert (panels[0]["xlim"][0] > panels[0]["xlim"][1]) == spec["reverse"]

    for source_label, label, color, _, marker in LEVER_CASES:
        record = next(r for r in records if r["experiment"] == experiment
                      and r["regime"] == "work" and r["label"] == source_label)
        adoption = next(line for line in panels[0]["lines"] if line["name"] == label)
        revenue = next(line for line in panels[1]["lines"] if line["name"] == label)
        assert adoption["x"] == record["values"]
        assert adoption["y"] == pytest.approx(100 * np.asarray(record["adoption"]))
        demand = np.asarray(record["demand"])
        assert revenue["y"] == pytest.approx(demand / demand[record["baseline_index"]])
        assert adoption["color"] == revenue["color"] == color
        assert (adoption["marker"] is not None) == (marker is not None)
