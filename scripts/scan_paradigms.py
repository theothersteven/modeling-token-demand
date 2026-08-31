"""Reproducible, exploratory parameter scan; does not change the model.

Run from the project root with ``python scripts/scan_paradigms.py``.
Results are diagnostics, not an estimate of how common each behavior is.
"""

import argparse
from collections import Counter
from dataclasses import asdict, replace
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modeling_token_demand import (
    AttentionConstrainedOptimizer, IndustryModel, OptimizationSettings,
    PolicyOptimizer, Scenario,
)
from modeling_token_demand.calibrations import REFERENCE_INDUSTRY
from modeling_token_demand.paradigms import boundary_hits, curve_record, shape


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/paradigm-scan.json"))
    parser.add_argument("--points", type=int, default=25)
    args = parser.parse_args()
    settings = OptimizationSettings(
        max_tokens_per_work_hour=20_000_000,
        grid_points_per_dimension=11, local_starts=2,
    )
    work = PolicyOptimizer(settings)
    attention = AttentionConstrainedOptimizer(settings)
    values = np.unique(np.append(np.geomspace(0.25, 10, args.points), [1, 2, 5]))
    records = []
    candidates = list(product((0.5, 1, 3), (0.5, 1, 2), (0.25, 0.5, 0.8, 0.95)))
    for index, (capability, execution, beta) in enumerate(candidates):
        industry = replace(
            REFERENCE_INDUSTRY,
            name=f"frontier={capability:g}; execution={execution:g}; beta={beta:g}",
            capability_horizon_hours=12 * capability,
            execution_scale=4 * execution,
            verification_elasticity=beta,
            human_attention_hours=100_000,
        )
        model = IndustryModel(industry)
        for axis, field in (("capability", "model_capability"),
                            ("efficiency", "token_efficiency")):
            scenarios = [Scenario(**{field: float(value)}) for value in values]
            work_outcomes = [work.solve(model, scenario) for scenario in scenarios]
            attention_outcomes = []
            for scenario in scenarios:
                try:
                    outcome = attention.solve_interior(model, scenario)
                except ValueError:
                    outcome = attention.solve(model, scenario)
                attention_outcomes.append(outcome)
            for regime, outcomes in (("work", work_outcomes),
                                     ("attention", attention_outcomes)):
                hits = boundary_hits(outcomes, settings)
                # Adoption does not change the optimal work policy. Reuse that
                # policy to scan its independent threshold distribution exactly.
                midpoints = tuple(REFERENCE_INDUSTRY.adoption_midpoint + offset
                                  for offset in (-9, -1, 3, 7, 11))
                adoption_grid = product(midpoints, (1, 4, 16)) if regime == "work" \
                    else [(REFERENCE_INDUSTRY.adoption_midpoint, REFERENCE_INDUSTRY.adoption_scale)]
                for midpoint, spread in adoption_grid:
                    variant = replace(industry, adoption_midpoint=midpoint,
                                      adoption_scale=spread)
                    variant_model = IndustryModel(variant)
                    selected = [variant_model.evaluate(o.policy, scenario)
                                for o, scenario in zip(outcomes, scenarios)]
                    demand = [o.work_limited_tokens if regime == "work"
                              else o.attention_limited_tokens for o in selected]
                    records.append({
                        "industry": asdict(variant), "axis": axis,
                        "regime": regime, "values": values.tolist(),
                        "demand": demand, "shape": shape(demand),
                        "adoption": [o.adoption_share for o in selected],
                        "surplus": [o.surplus_per_work_hour for o in selected],
                        "s": [o.policy.delegation_hours for o in selected],
                        "x": [o.policy.tokens_per_work_hour for o in selected],
                        "boundary_hits": hits,
                        "positive_attention_value": all(
                            o.surplus_per_attention_hour > 0 for o in selected),
                    })
        print(f"Scanned {index + 1}/{len(candidates)} technical configurations: "
              f"{industry.name}", flush=True)
    # A targeted second stage searches for smooth attention-side reversals.
    # Broader capability range; varying fixed overhead changes the transition
    # from fixed checkpoint overhead to scope-dependent review.
    extended = list(product(
        (.00001, .0001, .001, .003, .01, .03, .1, .3),
        (.1, .3, 1, 3), (.8, .9, .95, .98), (.15, .25, .5, .75),
    ))
    extended_values = np.unique(np.append(np.geomspace(.1, 30, 61), [1, 2, 5]))
    rejected = 0
    for h0, frontier, beta, alpha in extended:
        industry = replace(
            REFERENCE_INDUSTRY, name="Attention transition scan",
            verification_fixed_hours=h0, capability_horizon_hours=12 * frontier,
            verification_elasticity=beta, inference_returns=alpha,
            human_attention_hours=100_000,
        )
        model = IndustryModel(industry)
        try:
            outcomes = [attention.solve_interior(model, Scenario(model_capability=float(value)))
                        for value in extended_values]
        except ValueError:
            rejected += 1
            continue
        record = curve_record(industry, "capability", extended_values, outcomes, "attention")
        record["boundary_hits"] = boundary_hits(outcomes, settings)
        record["positive_attention_value"] = all(o.surplus_per_attention_hour > 0 for o in outcomes)
        records.append(record)
    counts = Counter((r["regime"], r["axis"], r["shape"]) for r in records
                     if not r["boundary_hits"] and
                     (r["regime"] == "work" or r["positive_attention_value"]))
    report = {
        "model": "single_attempt",
        "settings": asdict(settings), "shape_excursion": 0.08,
        "technical_configurations": len(candidates),
        "threshold_configurations_per_work_case": 15,
        "additional_attention_configurations": len(extended),
        "additional_attention_outside_bounds": rejected,
        "counts_without_boundary_hits": {" / ".join(k): v for k, v in sorted(counts.items())},
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, separators=(",", ":"), allow_nan=False))
    print(json.dumps(report["counts_without_boundary_hits"], indent=2), flush=True)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
