"""Question-led Section 5 experiments using the paper's existing model.

Each curve changes one parameter, reoptimizes the interaction policy, and
records demand separately from delegated and successfully completed work.
The notebook calls this module so static and interactive figures share data.
"""

from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

from .calibrations import REFERENCE_INDUSTRY
from .model import Industry, IndustryModel, Scenario
from .optimizer import AttentionConstrainedOptimizer, PolicyOptimizer
from .paradigms import axis_values, boundary_hits, gallery_settings


def experiments(points=81):
    """Axis values and isolated changes; all unmentioned inputs stay fixed."""
    return (
        dict(
            key="verification-speed", baseline=1.0,
            values=axis_values(.1, 2, 1, points, [.25, .5]),
            xlabel="Verification time multiplier, v (faster to the right)",
            ticks=[.1, .25, .5, 1, 2], scale="log", reverse=True,
            cases=[dict(label="Reference industry", target="scenario",
                        parameter="verification_time_multiplier")],
        ),
        dict(
            key="review-growth", baseline=.5,
            values=np.unique(np.r_[np.linspace(.1, .95, points), .25, .5]),
            xlabel="Review growth, β (slower growth to the right)",
            ticks=[.1, .25, .5, .75, .95], scale="linear", reverse=True,
            cases=[dict(label=f"Capability m = {m:g}", target="industry",
                        parameter="verification_elasticity",
                        scenario_overrides=dict(model_capability=m)) for m in (1, 5)],
        ),
        dict(
            key="harness-feasibility", baseline=1.0,
            values=axis_values(.25, 10, 1, points, [2, 5]),
            xlabel="Improvement factor (relative to baseline)",
            ticks=[.25, .5, 1, 2, 5, 10], scale="log", reverse=False,
            cases=[
                dict(label="Feasibility only: λ", target="industry",
                     parameter="capability_horizon_hours", parameter_scale=12.0),
                dict(label="Model capability: m", target="scenario",
                     parameter="model_capability"),
            ],
        ),
        dict(
            key="efficiency-returns", baseline=1.0,
            values=axis_values(1, 100, 1, points, [2, 5, 10, 50]),
            xlabel="Token efficiency, η (effective inference per token)",
            ticks=[1, 2, 5, 10, 20, 50, 100], scale="log", reverse=False,
            cases=[dict(label=f"Inference returns α = {alpha:g}", target="scenario",
                        parameter="token_efficiency",
                        industry_overrides=dict(inference_returns=alpha))
                   for alpha in (.25, .5, .75)],
        ),
    )


def configuration(record, value):
    """Reconstruct exactly the industry and scenario at a recorded point."""
    industry, scenario = Industry(**record["industry"]), Scenario(**record["scenario"])
    change = {record["parameter"]: float(value) * record["parameter_scale"]}
    if record["target"] == "industry":
        industry = replace(industry, **change)
    else:
        scenario = replace(scenario, **change)
    return industry, scenario


def solve_interventions(points=81):
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
    strict = {"work": PolicyOptimizer(strict_settings),
              "attention": AttentionConstrainedOptimizer(strict_settings)}
    records, errors = [], []
    for experiment in experiments(points):
        for case in experiment["cases"]:
            for regime in ("work", "attention"):
                industry = replace(
                    REFERENCE_INDUSTRY, **case.get("industry_overrides", {}),
                    human_attention_hours=100_000 if regime == "attention" else None,
                )
                scenario = Scenario(**case.get("scenario_overrides", {}))
                record = dict(
                    experiment=experiment["key"], label=case["label"], regime=regime,
                    industry=asdict(industry), scenario=asdict(scenario),
                    target=case["target"], parameter=case["parameter"],
                    parameter_scale=case.get("parameter_scale", 1.0),
                    values=list(map(float, experiment["values"])),
                    baseline_index=list(experiment["values"]).index(experiment["baseline"]),
                )
                outcomes = []
                for value in record["values"]:
                    current, scenario = configuration(record, value)
                    model = IndustryModel(current)
                    outcome = (work.solve(model, scenario) if regime == "work"
                               else attention.solve_interior(model, scenario))
                    outcomes.append(outcome)
                hits = boundary_hits(outcomes, settings)
                if hits:
                    raise AssertionError(f"{case['label']}/{experiment['key']}/{regime}: {hits}")
                if regime == "attention":
                    assert all(o.surplus_per_attention_hour > 0 for o in outcomes)
                    assert all(o.policy.max_attempts == 1 for o in outcomes)
                assigned = np.array([
                    industry.potential_work_hours * o.adoption_share if regime == "work"
                    else industry.human_attention_hours * o.policy.delegation_hours
                    / (o.expected_attempts * o.verification_hours_per_attempt)
                    for o in outcomes
                ])
                demand = np.array([o.work_limited_tokens if regime == "work"
                                   else o.attention_limited_tokens for o in outcomes])
                intensity = np.array([o.policy.tokens_per_work_hour * o.expected_attempts
                                      for o in outcomes])
                np.testing.assert_allclose(demand, assigned * intensity, rtol=1e-12)
                record.update(
                    demand=demand.tolist(), assigned_work=assigned.tolist(),
                    completed_work=(assigned * [o.eventual_success for o in outcomes]).tolist(),
                    tokens_per_assigned_work=intensity.tolist(),
                    success=[o.eventual_success for o in outcomes],
                    adoption=[o.adoption_share for o in outcomes] if regime == "work" else None,
                    s=[o.policy.delegation_hours for o in outcomes],
                    x=[o.policy.tokens_per_work_hour for o in outcomes],
                    k=[o.policy.max_attempts for o in outcomes],
                    expected_attempts=[o.expected_attempts for o in outcomes],
                    verification_hours=[o.verification_hours_per_attempt for o in outcomes],
                    objective=[work.objective_value(o) if regime == "work"
                               else attention.objective_value(o) for o in outcomes],
                )
                # Check endpoints, baseline, and sampled extrema against a
                # denser, wider search that enumerates all retry caps. For
                # attention this also independently checks the scalar solver.
                checkpoints = {0, len(outcomes) - 1, record["baseline_index"]}
                for metric in ("demand", "assigned_work", "completed_work"):
                    checkpoints.update((int(np.argmin(record[metric])),
                                        int(np.argmax(record[metric]))))
                for index in sorted(checkpoints):
                    current, scenario = configuration(record, record["values"][index])
                    checked = strict[regime].solve(IndustryModel(current), scenario)
                    expected = strict[regime].objective_value(checked)
                    error = abs(expected - record["objective"][index]) / max(1, abs(expected))
                    assert error < 1e-8, (experiment["key"], case["label"], regime, error)
                    assert not boundary_hits([checked], strict_settings)
                    errors.append(error)
                records.append(record)
        print(f"Section 5: {experiment['key']} solved and audited.", flush=True)

    return dict(settings=asdict(settings), curves=records, audit=dict(
        boundary_hits=[], independent_checks=len(errors),
        max_relative_objective_error=max(errors),
        policy_count=sum(len(record["values"]) for record in records),
    ))


def build_intervention_figures(directory: Path, points=81):
    """Same rows and outcomes for every question, with independent y-axes."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    directory.mkdir(parents=True, exist_ok=True)
    report = solve_interventions(points)
    for experiment in experiments(points):
        fig, axes = plt.subplots(2, 3, figsize=(15.8, 8.4), constrained_layout=True)
        records = [r for r in report["curves"] if r["experiment"] == experiment["key"]]
        colors = ["#595959", "#0072B2", "#D55E00"]
        styles = ["-", "--", "-."]
        for row, regime in enumerate(("work", "attention")):
            for line_index, record in enumerate(r for r in records if r["regime"] == regime):
                for column, metric in enumerate(("demand", "assigned_work", "completed_work")):
                    values = np.array(record[metric])
                    if regime == "work" and column > 0:
                        values = 100 * values / record["industry"]["potential_work_hours"]
                    else:
                        values = values / values[record["baseline_index"]]
                    axes[row, column].plot(
                        record["values"], values, label=record["label"],
                        color=colors[line_index], linestyle=styles[line_index], linewidth=2.2,
                    )
            for column, outcome in enumerate(("Token demand", "Work delegated", "Work completed")):
                ax = axes[row, column]
                prefix = "Fixed work" if regime == "work" else "Scarce attention"
                ax.set_title(f"({chr(97 + row * 3 + column)}) {prefix}: {outcome.lower()}",
                             loc="left", fontsize=10)
                ax.set_xscale(experiment["scale"])
                ax.set_xticks(experiment["ticks"])
                ax.xaxis.set_major_formatter(ScalarFormatter())
                limits = (min(experiment["values"]), max(experiment["values"]))
                ax.set_xlim(limits[::-1] if experiment["reverse"] else limits)
                ax.set_xlabel(experiment["xlabel"], fontsize=9)
                if regime == "work" and column > 0:
                    ax.set_ylabel("Share of potential work (%)", fontsize=9)
                    ax.set_ylim(0, 100)
                else:
                    ax.set_ylabel("Index (own baseline = 1)", fontsize=9)
                    ax.axhline(1, color=".65", linestyle=":", linewidth=.9)
                    ax.set_ylim(bottom=0)
                ax.axvline(experiment["baseline"], color=".65", linestyle=":", linewidth=.9)
                ax.grid(alpha=.18)
                ax.tick_params(labelsize=8)
                ax.spines[["top", "right"]].set_visible(False)
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside upper center",
                   ncol=3, frameon=False, fontsize=10)
        fig.savefig(directory / f"intervention-{experiment['key']}.png", dpi=180,
                    bbox_inches="tight")
        plt.close(fig)
    (directory / "interventions.json").write_text(
        json.dumps(report, separators=(",", ":"), allow_nan=False)
    )
    print(f"Section 5 audit: {report['audit']}", flush=True)
    return report
