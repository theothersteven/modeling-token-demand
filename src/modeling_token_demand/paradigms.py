"""Reproducible existence examples using the unchanged economic model.

The notebook calls this module to generate the gallery and its audit data.
The exploratory scanner uses the same shape and boundary diagnostics, but its
coarse classifications are candidate labels, not claims of smooth reversals.
"""

from dataclasses import asdict, replace
import json
import math
from pathlib import Path

import numpy as np

from .calibrations import (
    CAPABILITY_VALLEY, attention_paradigms, work_paradigms,
)
from .model import IndustryModel, Scenario
from .optimizer import AttentionConstrainedOptimizer, OptimizationSettings, PolicyOptimizer


COLORS = {
    "Reference industry": "#595959",
    "Adoption concentration: high": "#D55E00",
    "Adoption concentration: low": "#D55E00",
    "Early saturation": "#008B8B",
    "Supervisory leverage": "#009E73",
    "Review bottleneck": "#CC79A7",
    "Capability valley": "#7A5195",
    "Verification burden: low": "#009E73",
    "Verification burden: high": "#009E73",
    "Offsetting efficiency": "#6B6B00",
}


def gallery_settings():
    return OptimizationSettings(
        min_delegation_hours=.002, max_delegation_hours=800,
        min_tokens_per_work_hour=200, max_tokens_per_work_hour=200_000_000,
        grid_points_per_dimension=17, local_starts_per_attempt=4,
    )


def shape(values, excursion=0.08):
    """Detect material excursions, including discrete jumps; not smoothness.

    Reversals must move at least 8% from a running extremum. A flat sampled
    tail alone is never reported as an asymptotic saturation result.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)) \
            or np.any(values <= 0):
        raise ValueError("shape requires at least two finite positive observations")
    logs = np.log(values)
    threshold = math.log1p(excursion)
    direction = 0
    anchor = extreme = float(logs[0])
    turns = []
    for value in logs[1:]:
        value = float(value)
        if direction == 0:
            if abs(value - anchor) > threshold:
                direction = 1 if value > anchor else -1
                turns.append(direction)
                extreme = value
        elif direction > 0:
            extreme = max(extreme, value)
            if extreme - value > threshold:
                direction = -1
                turns.append(direction)
                extreme = value
        else:
            extreme = min(extreme, value)
            if value - extreme > threshold:
                direction = 1
                turns.append(direction)
                extreme = value
    return {
        (): "approximately flat", (1,): "rising", (-1,): "falling",
        (1, -1): "hump", (-1, 1): "U-shape",
        (-1, 1, -1): "fall-rise-fall", (1, -1, 1): "rise-fall-rise",
    }.get(tuple(turns), "multiple reversals")


def boundary_hits(outcomes, settings):
    hits = set()
    for outcome in outcomes:
        policy = outcome.policy
        for name, value, lower, upper in (
            ("s", policy.delegation_hours, settings.min_delegation_hours,
             settings.max_delegation_hours),
            ("x", policy.tokens_per_work_hour, settings.min_tokens_per_work_hour,
             settings.max_tokens_per_work_hour),
        ):
            if value <= lower * 1.0001 or value >= upper / 1.0001:
                hits.add(name)
        if policy.max_attempts == settings.max_attempts:
            hits.add("k")
    return sorted(hits)


def axis_values(lower, upper, baseline, points=81, extra=()):
    return np.unique(np.concatenate((np.geomspace(lower, upper, points),
                                     [baseline], extra)))


def curve_record(industry, axis, values, outcomes, regime):
    demand = [o.work_limited_tokens if regime == "work" else o.attention_limited_tokens
              for o in outcomes]
    baseline = 10.0 if axis == "price" else 1.0
    baseline_index = list(values).index(baseline)
    assigned = [industry.potential_work_hours * o.adoption_share if regime == "work"
                else industry.human_attention_hours * o.policy.delegation_hours
                / (o.expected_attempts * o.verification_hours_per_attempt) for o in outcomes]
    return {
        "industry": asdict(industry), "axis": axis, "regime": regime,
        "values": list(map(float, values)), "baseline_index": baseline_index,
        "demand": demand, "shape": shape(demand[::-1] if axis == "price" else demand),
        "adoption": [o.adoption_share for o in outcomes],
        "assigned_work": assigned,
        "completed_work": [amount * o.eventual_success for amount, o in zip(assigned, outcomes)],
        "tokens_per_assigned_work": [o.policy.tokens_per_work_hour * o.expected_attempts
                                     for o in outcomes],
        "surplus": [o.surplus_per_work_hour for o in outcomes],
        "attention_value": [o.surplus_per_attention_hour + industry.human_cost_per_hour
                            for o in outcomes],
        "s": [o.policy.delegation_hours for o in outcomes],
        "x": [o.policy.tokens_per_work_hour for o in outcomes],
        "k": [o.policy.max_attempts for o in outcomes],
        "leverage": [o.policy.delegation_hours / o.verification_hours_per_attempt
                     for o in outcomes],
        "expected_attempts": [o.expected_attempts for o in outcomes],
    }


def solve_gallery(points=81):
    """Solve all panels; reject bound-driven shapes and audit selected optima."""
    settings = gallery_settings()
    work = PolicyOptimizer(settings)
    attention = AttentionConstrainedOptimizer(settings)
    strict_settings = replace(
        settings, grid_points_per_dimension=25, local_starts_per_attempt=8,
        min_delegation_hours=settings.min_delegation_hours / 10,
        max_delegation_hours=settings.max_delegation_hours * 10,
        min_tokens_per_work_hour=settings.min_tokens_per_work_hour / 10,
        max_tokens_per_work_hour=settings.max_tokens_per_work_hour * 10,
        max_attempts=16,
    )
    strict_work = PolicyOptimizer(strict_settings)
    general_attention = AttentionConstrainedOptimizer(strict_settings)
    grids = {
        "capability": ("model_capability", axis_values(.25, 10, 1, points, [2, 5])),
        "efficiency": ("token_efficiency", axis_values(.25, 10, 1, points, [2, 5])),
        "price": ("token_price_per_million", axis_values(1, 80, 10, points, [2, 5, 20, 40])),
    }
    curves, audit = [], []
    # Only adoption differs between the first two work cases, so reuse the
    # identical optimized policies (the objective is surplus, not adoption).
    policy_cache = {}
    for industry in work_paradigms():
        model = IndustryModel(industry)
        technical = replace(industry, name="", adoption_midpoint=0, adoption_scale=1)
        for axis, (field, values) in grids.items():
            outcomes = []
            for value in values:
                scenario = Scenario(**{field: float(value)})
                key = (technical, scenario)
                if key not in policy_cache:
                    policy_cache[key] = work.solve(model, scenario).policy
                outcomes.append(model.evaluate(policy_cache[key], scenario))
            hits = boundary_hits(outcomes, settings)
            if hits:
                raise AssertionError(f"{industry.name}/{axis}: bound hits {hits}")
            record = curve_record(industry, axis, values, outcomes, "work")
            if axis == "price":
                demand = np.asarray(record["demand"])
                assert np.max(np.diff(demand)) <= 1e-5 * np.max(demand)
            adoption = np.asarray(record["adoption"])
            ordered_adoption = adoption[::-1] if axis == "price" else adoption
            assert np.min(np.diff(ordered_adoption)) >= -1e-8
            # Audit endpoints, baseline, and demand maximum against a denser
            # search with expanded numerical bounds and additional retry caps.
            indices = {0, len(values) - 1, record["baseline_index"],
                       int(np.argmax(record["demand"]))}
            for index in sorted(indices):
                scenario = Scenario(**{field: float(values[index])})
                checked = strict_work.solve(model, scenario)
                relative = abs(checked.surplus_per_work_hour - outcomes[index].surplus_per_work_hour) \
                    / max(1, abs(checked.surplus_per_work_hour))
                assert relative < 1e-8, (industry.name, axis, values[index], relative)
                assert math.isclose(checked.work_limited_tokens,
                                    outcomes[index].work_limited_tokens, rel_tol=5e-4)
                audit.append(relative)
            curves.append(record)
            print(f"Gallery: {industry.name} / {axis}: {record['shape']}", flush=True)
    for base in attention_paradigms():
        industry = replace(base, human_attention_hours=100_000)
        model = IndustryModel(industry)
        values = axis_values(.1, 30, 1, points, [2, 5, 10]) if base == CAPABILITY_VALLEY \
            else grids["capability"][1]
        outcomes = [attention.solve_interior(model, Scenario(model_capability=float(value)))
                    for value in values]
        assert not boundary_hits(outcomes, settings), industry.name
        assert all(o.surplus_per_attention_hour > 0 for o in outcomes)
        record = curve_record(industry, "capability", values, outcomes, "attention")
        indices = {0, len(values) - 1, record["baseline_index"],
                   int(np.argmin(record["demand"])), int(np.argmax(record["demand"]))}
        for index in sorted(indices):
            checked = general_attention.solve(model, Scenario(model_capability=float(values[index])))
            relative = abs(checked.surplus_per_attention_hour - outcomes[index].surplus_per_attention_hour) \
                / max(1, abs(checked.surplus_per_attention_hour))
            assert relative < 1e-8, (industry.name, values[index], relative)
            assert math.isclose(checked.attention_limited_tokens,
                                outcomes[index].attention_limited_tokens, rel_tol=5e-4)
            audit.append(relative)
        curves.append(record)
        print(f"Gallery: {industry.name} / capability: {record['shape']}", flush=True)
    return {"settings": asdict(settings), "points_per_axis_before_anchors": points,
            "audit": {"boundary_hits": [], "independent_checks": len(audit),
                      "max_relative_objective_error": max(audit)}, "curves": curves}


def _index(record, key="demand"):
    values = np.asarray(record[key])
    return values / values[record["baseline_index"]]


def audit_main_sweeps(settings, sweep_sets, scenario_axes, industries):
    """Check the main Section 3 samples, with independent extrema re-solves."""
    strict = replace(
        settings, grid_points_per_dimension=25, local_starts_per_attempt=8,
        min_delegation_hours=settings.min_delegation_hours / 10,
        max_delegation_hours=settings.max_delegation_hours * 10,
        min_tokens_per_work_hour=settings.min_tokens_per_work_hour / 10,
        max_tokens_per_work_hour=settings.max_tokens_per_work_hour * 10,
        max_attempts=16,
    )
    work = PolicyOptimizer(strict)
    attention = AttentionConstrainedOptimizer(strict)
    configurations = {industry.name: industry for industry in industries}
    curves, errors = [], []
    for axis, field, values in scenario_axes:
        for regime in ("work", "attention"):
            for name, outcomes in sweep_sets[f"{regime} {axis}"].items():
                industry = configurations[name]
                assert not boundary_hits(outcomes, settings), (name, regime, axis)
                record = curve_record(industry, axis, values, outcomes, regime)
                indices = {0, len(values) - 1, record["baseline_index"],
                           int(np.argmin(record["demand"])), int(np.argmax(record["demand"]))}
                model = IndustryModel(industry)
                for index in sorted(indices):
                    scenario = Scenario(**{field: float(values[index])})
                    checked = work.solve(model, scenario) if regime == "work" \
                        else attention.solve_interior(model, scenario)
                    metric = "surplus_per_work_hour" if regime == "work" else "surplus_per_attention_hour"
                    expected = getattr(checked, metric)
                    error = abs(expected - getattr(outcomes[index], metric)) / max(1, abs(expected))
                    assert error < 1e-8, (name, regime, axis, values[index], error)
                    quantity = "work_limited_tokens" if regime == "work" else "attention_limited_tokens"
                    assert math.isclose(getattr(checked, quantity), record["demand"][index], rel_tol=5e-4)
                    errors.append(error)
                curves.append(record)
        print(f"Main-figure independent audit: {axis} passed.", flush=True)
    return {"curves": curves, "audit": {"boundary_hits": [],
            "independent_checks": len(errors), "max_relative_objective_error": max(errors)}}


def _format_axis(ax, xlabel, ticks, baseline, ylabel, log_y=False, reverse=False):
    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xticks(ticks, [f"{tick:g}" for tick in ticks])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axvline(baseline, color=".65", linestyle=":", linewidth=.9)
    ax.grid(which="major", alpha=.2)
    if reverse:
        ax.invert_xaxis()


def build_paradigm_figures(directory: Path, points=81):
    """Called by the notebook, so the existing export captures these figures."""
    import matplotlib.pyplot as plt

    report = solve_gallery(points)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    work = [r for r in report["curves"] if r["regime"] == "work"]
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.1), constrained_layout=True)
    for ax, axis, title, xlabel, ticks, baseline in zip(
        axes, ("capability", "efficiency", "price"),
        ("(a) Adoption can create a capability hump", "(b) Efficiency: rebound, then savings",
         "(c) Cheaper tokens unlock new work"),
        ("Model capability, m", "Token efficiency, eta", "Token price, USD / million"),
        ((.25, .5, 1, 2, 5, 10), (.25, .5, 1, 2, 5, 10), (1, 2, 5, 10, 20, 40, 80)),
        (1, 1, 10),
    ):
        for record in (r for r in work if r["axis"] == axis):
            name = record["industry"]["name"]
            ax.plot(record["values"], _index(record), label=name, color=COLORS[name], linewidth=2,
                    linestyle="-" if name == "Reference industry" or name.endswith(": high") else "-.")
        if axis == "price":
            prices = next(r["values"] for r in work if r["axis"] == "price")
            ax.plot(prices, 10 / np.array(prices), color="black", linestyle="-.",
                    label="Constant revenue (10 / price)", linewidth=1.4)
        _format_axis(ax, xlabel, ticks, baseline, "Token demand index (baseline = 1)",
                     log_y=axis == "price", reverse=axis == "price")
        if axis != "price":
            ax.set_ylim(bottom=0)
        ax.axhline(1, color=".75", linestyle=":", linewidth=.9)
        ax.set_title(title, loc="left", fontsize=10)
    fig.legend(*axes[-1].get_legend_handles_labels(), loc="outside upper center",
               ncol=4, frameon=False, fontsize=9)
    fig.savefig(directory / "paradigm-work-demand.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for record in (r for r in work if r["axis"] == "price"):
        name = record["industry"]["name"]
        prices = np.asarray(record["values"])
        axes[0].plot(prices, 100 * np.array(record["adoption"]), label=name,
                     color=COLORS[name], linewidth=2,
                     linestyle="-" if name == "Reference industry" or name.endswith(": high") else "-.")
        axes[1].plot(prices, prices / 10 * _index(record), label=name,
                     color=COLORS[name], linewidth=2,
                     linestyle="-" if name == "Reference industry" or name.endswith(": high") else "-.")
    for ax, title, ylabel in zip(axes,
        ("(a) Work assigned to AI approaches a ceiling", "(b) Revenue can rise and then fall"),
        ("Adopted share of potential work (%)", "Token revenue index (price = 10 baseline)")):
        _format_axis(ax, "Token price, USD / million (cheaper to the right)",
                     (1, 2, 5, 10, 20, 40, 80), 10, ylabel, reverse=True)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylim(bottom=0)
    axes[0].set_ylim(0, 102)
    axes[1].axhline(1, color=".55", linestyle=":", linewidth=1)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center",
               ncol=3, frameon=False, fontsize=9)
    fig.savefig(directory / "paradigm-adoption-and-revenue.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.8), constrained_layout=True)
    titles = ("(a) Low verification burden: rising", "(b) High verification burden: hump",
              "(c) Capability valley: fall, then rise")
    for ax, record, title in zip(axes,
            (r for r in report["curves"] if r["regime"] == "attention"), titles):
        name = record["industry"]["name"]
        ax.plot(record["values"], _index(record), label=name, color=COLORS[name], linewidth=2.3,
                linestyle="--" if name.endswith(": low") else "-" if name.endswith(": high") else "-.")
        ticks = (.1, .25, 1, 2, 5, 10, 30) if name == "Capability valley" \
            else (.25, .5, 1, 2, 5, 10)
        _format_axis(ax, "Model capability, m", ticks, 1,
                     "Token demand index (m = 1 baseline)")
        ax.axhline(1, color=".7", linestyle=":", linewidth=.9)
        ax.set_title(title, loc="left", fontsize=10)
    fig.legend([ax.lines[0] for ax in axes], [r.name for r in attention_paradigms()],
               loc="outside upper center", ncol=3, frameon=False, fontsize=9)
    fig.savefig(directory / "paradigm-attention-capability.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    (directory / "paradigms.json").write_text(json.dumps(report, separators=(",", ":"), allow_nan=False))
    print(f"Gallery audit: {report['audit']}", flush=True)
    return report
