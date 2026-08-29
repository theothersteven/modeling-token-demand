"""Plot optimized token demand against token price and token efficiency.

Run from any directory after installing the package with the ``plot`` extra:

    python scripts/plot_comparative_statics.py

The calibrations are illustrative rather than empirical.  Every plotted point
resolves the user's choice of delegation horizon, inference intensity, and
retry cap before computing industry-level demand.
"""

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from modeling_token_demand import (
    IndustryModel,
    OptimizationSettings,
    PolicyOptimizer,
    Scenario,
    illustrative_industries,
)


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures"


def optimized_demand(
    optimizer: PolicyOptimizer,
    scenario_name: str,
    values: np.ndarray,
) -> dict[str, np.ndarray]:
    """Solve all industry policies for one comparative-static variable."""

    curves = {}
    baseline = Scenario()
    for industry in illustrative_industries():
        model = IndustryModel(industry)
        demand = []
        for value in values:
            scenario = replace(baseline, **{scenario_name: float(value)})
            demand.append(optimizer.solve(model, scenario).realized_tokens / 1e9)
        curves[industry.name] = np.asarray(demand)
    return curves


def draw_curves(
    values: np.ndarray,
    curves: dict[str, np.ndarray],
    xlabel: str,
    output_name: str,
    reference_value: float,
    reverse_x: bool = False,
) -> None:
    """Draw one set of industry demand curves and save it as a PNG."""

    fig, ax = plt.subplots(figsize=(9.5, 6.0), constrained_layout=True)
    for label, demand in curves.items():
        style = {"linewidth": 2.8} if label == "Near an adoption threshold" else {}
        ax.plot(values, demand, label=label, **style)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(reference_value, color="0.35", linestyle="--", linewidth=1.2)
    if reverse_x:
        ax.invert_xaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Optimized physical token demand (billions per period)")
    ax.grid(which="major", alpha=0.25)
    ax.grid(which="minor", alpha=0.10)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(FIGURE_DIR / output_name, dpi=180)
    plt.close(fig)


def main() -> None:
    FIGURE_DIR.mkdir(exist_ok=True)

    # A lighter search is sufficient for smooth plots; the default optimizer
    # uses a denser grid when inspecting an individual optimum.
    optimizer = PolicyOptimizer(
        OptimizationSettings(
            grid_points_per_dimension=11,
            local_starts_per_attempt=2,
        )
    )

    token_prices = np.geomspace(1.0, 80.0, 33)
    price_curves = optimized_demand(
        optimizer, "token_price_per_million", token_prices
    )
    draw_curves(
        token_prices,
        price_curves,
        xlabel="Token price, $ per million physical tokens (lower cost to the right)",
        output_name="token-demand-vs-price.png",
        reference_value=10.0,
        reverse_x=True,
    )

    token_efficiencies = np.geomspace(0.35, 5.0, 33)
    efficiency_curves = optimized_demand(
        optimizer, "token_efficiency", token_efficiencies
    )
    draw_curves(
        token_efficiencies,
        efficiency_curves,
        xlabel="Token efficiency, effective inference per physical token",
        output_name="token-demand-vs-efficiency.png",
        reference_value=1.0,
    )

    print(f"Wrote figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
