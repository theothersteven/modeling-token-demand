"""Find qualitative comparative statics by changing one industry parameter.

This is an exploratory calibration aid, not an estimator.  Every candidate
starts from ``REFERENCE_INDUSTRY`` and changes exactly one economic or
technical field.  The report records whether token demand, adoption, or token
spending is rising, falling, hump-shaped, or U-shaped over the plotted range.

Run from the project root with::

    .venv/bin/python scripts/scan_one_parameter.py
"""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modeling_token_demand import (  # noqa: E402
    AttentionConstrainedOptimizer,
    IndustryModel,
    OptimizationSettings,
    PolicyOptimizer,
    Scenario,
)
from modeling_token_demand.calibrations import REFERENCE_INDUSTRY  # noqa: E402
from modeling_token_demand.paradigms import boundary_hits, shape  # noqa: E402


CANDIDATES = {
    "capability_horizon_hours": (3, 6, 9, 18, 24, 36, 48),
    "capability_shape": (.35, .5, .75, 1, 1.5, 2, 3),
    "execution_scale": (.5, 1, 2, 8, 16, 32),
    "inference_returns": (.1, .2, .3, .4, .6, .7, .8, .9),
    "verification_fixed_hours": (.00001, .0001, .001, .003, .01, .1, .3, 1),
    "verification_scale": (.001, .005, .01, .025, .1, .2, .5, 1),
    "verification_elasticity": (.05, .15, .25, .35, .65, .75, .85, .95, 1),
    "value_per_work_hour": (25, 50, 75, 150, 200, 400),
    "human_cost_per_hour": (25, 50, 75, 150, 200, 400),
    "adoption_location": (
        20, 25, 30, 35, 40, 60, 68, 72, 80, 84, 86, 88, 90, 92, 108, 140,
    ),
    "adoption_scale": (.25, .5, 1, 2, 8, 16, 32, 64),
}

AXES = {
    "capability": (
        "model_capability", .1, 30, 1, (2, 5, 10),
    ),
    "efficiency": (
        "token_efficiency", .25, 10, 1, (2, 5),
    ),
    "price": (
        "token_price", .1, 8, 1, (.2, .5, 2, 4),
    ),
}


def axis_values(lower, upper, baseline, points, anchors):
    return np.unique(np.concatenate((
        np.geomspace(lower, upper, points), [baseline], anchors,
    )))


def ordered(values, axis):
    """Order observations in the direction of technological improvement."""

    values = np.asarray(values, dtype=float)
    return values[::-1] if axis == "price" else values


def series_record(outcomes, scenarios, regime, axis):
    demand = np.asarray([
        outcome.work_limited_tokens
        if regime == "work" else outcome.attention_limited_tokens
        for outcome in outcomes
    ], dtype=float)
    adoption = np.asarray([outcome.adoption_share for outcome in outcomes])
    spending = demand * np.asarray([scenario.token_price for scenario in scenarios])
    metrics = {"demand": demand, "spending": spending}
    if regime == "work":
        metrics["adoption"] = adoption
    records = {}
    for metric, values in metrics.items():
        trajectory = ordered(values, axis)
        records[metric] = {
            "shape": shape(trajectory),
            "start": float(trajectory[0]),
            "end": float(trajectory[-1]),
            "minimum": float(np.min(trajectory)),
            "maximum": float(np.max(trajectory)),
            "range_ratio": float(np.max(trajectory) / np.min(trajectory)),
        }
    return records


def solve_candidate(industry, grids, work, attention):
    model = IndustryModel(industry)
    result = {"curves": {}, "boundary_hits": [], "positive_attention_value": True}
    for axis, (field, values) in grids.items():
        scenarios = [Scenario(**{field: float(value)}) for value in values]
        work_outcomes = [work.solve(model, scenario) for scenario in scenarios]
        attention_outcomes = []
        for scenario in scenarios:
            try:
                outcome = attention.solve_interior(model, scenario)
            except ValueError:
                outcome = attention.solve(model, scenario)
            attention_outcomes.append(outcome)
        for regime, outcomes in (
            ("work", work_outcomes), ("attention", attention_outcomes),
        ):
            hits = boundary_hits(outcomes, work.settings)
            result["boundary_hits"].extend(
                f"{regime}/{axis}/{name}" for name in hits
            )
            if regime == "attention":
                result["positive_attention_value"] &= all(
                    outcome.surplus_per_attention_hour > 0 for outcome in outcomes
                )
            result["curves"][f"{regime}/{axis}"] = series_record(
                outcomes, scenarios, regime, axis,
            )
    result["boundary_hits"] = sorted(set(result["boundary_hits"]))
    return result


def signature(record):
    return {
        f"{curve}/{metric}": values["shape"]
        for curve, metrics in record["curves"].items()
        for metric, values in metrics.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("build/one-parameter-scan.json"),
    )
    parser.add_argument("--points", type=int, default=25)
    args = parser.parse_args()

    settings = OptimizationSettings(
        min_delegation_hours=.002,
        max_delegation_hours=800,
        min_tokens_per_work_hour=1,
        max_tokens_per_work_hour=2_000,
        grid_points_per_dimension=11,
        local_starts=2,
    )
    work = PolicyOptimizer(settings)
    attention = AttentionConstrainedOptimizer(settings)
    grids = {
        axis: (
            field,
            axis_values(lower, upper, baseline, args.points, anchors),
        )
        for axis, (field, lower, upper, baseline, anchors) in AXES.items()
    }

    reference = replace(
        REFERENCE_INDUSTRY,
        human_attention_hours=100_000,
    )
    reference_record = solve_candidate(reference, grids, work, attention)
    reference_signature = signature(reference_record)
    records = [{
        "field": None,
        "value": None,
        "industry": asdict(reference),
        **reference_record,
        "signature": reference_signature,
        "different_shapes": [],
    }]

    total = sum(len(values) for values in CANDIDATES.values())
    completed = 0
    for field, values in CANDIDATES.items():
        for value in values:
            industry = replace(
                reference,
                name=f"{field}={value:g}",
                **{field: value},
            )
            record = solve_candidate(industry, grids, work, attention)
            candidate_signature = signature(record)
            different = sorted(
                key for key, candidate_shape in candidate_signature.items()
                if candidate_shape != reference_signature[key]
            )
            records.append({
                "field": field,
                "value": value,
                "industry": asdict(industry),
                **record,
                "signature": candidate_signature,
                "different_shapes": different,
            })
            completed += 1
            print(
                f"Scanned {completed}/{total}: {field}={value:g}; "
                f"different shapes={len(different)}",
                flush=True,
            )

    report = {
        "model": "single_attempt",
        "rule": "Each candidate changes exactly one field from the reference.",
        "points_per_axis_before_anchors": args.points,
        "settings": asdict(settings),
        "axes": {
            axis: {"field": field, "values": values.tolist()}
            for axis, (field, values) in grids.items()
        },
        "reference_signature": reference_signature,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, separators=(",", ":"), allow_nan=False))
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
