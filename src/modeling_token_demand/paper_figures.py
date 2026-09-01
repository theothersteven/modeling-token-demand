"""Small, absolute-unit figures for the paper's six main comparisons.

The notebook remains the source of the optimized outcomes. This module only
selects audited curves and presents token demand and token spending together.
"""

from pathlib import Path

import numpy as np


WORK_CASES = (
    ("Reference industry", "Gradual adoption", "#355D8A", "-"),
    ("Adoption concentration: high", "Clustered adoption", "#C45123", "--"),
    ("Early saturation", "Early saturation", "#007F78", "-."),
)
ATTENTION_CASES = (
    ("Verification burden: low", "Reusable review", "#007F78", "-"),
    ("Reference industry", "Balanced review", "#355D8A", "--"),
    ("Verification burden: high", "Proportional review", "#785BA0", "-."),
)

AXES = {
    "capability": {
        "xlabel": "Model capability, m",
        "ticks": (.5, 1, 2, 5, 10),
        "limits": (.5, 10),
        "baseline": 1,
        "reverse": False,
    },
    "efficiency": {
        "xlabel": "Token efficiency, eta",
        "ticks": (.5, 1, 2, 5, 10),
        "limits": (.5, 10),
        "baseline": 1,
        "reverse": False,
    },
    "price": {
        "xlabel": "Token price, USD per million tokens (cheaper to the right)",
        "ticks": (1, 2, 5, 10, 20, 40),
        "limits": (1, 40),
        "baseline": 10,
        "reverse": True,
    },
}


def _curve(report, name, axis, regime):
    return next(
        record for record in report["main"]["curves"]
        if record["industry"]["name"] == name
        and record["axis"] == axis
        and record["regime"] == regime
    )


def _format(ax, title, axis, ylabel):
    spec = AXES[axis]
    ax.set_title(title, loc="left", fontsize=12, pad=12)
    ax.set_xlabel(spec["xlabel"], fontsize=11, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=11, labelpad=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(spec["ticks"], [f"{tick:g}" for tick in spec["ticks"]])
    left, right = spec["limits"]
    ax.set_xlim((right, left) if spec["reverse"] else (left, right))
    ax.axvline(spec["baseline"], color=".68", linestyle=":", linewidth=1)
    ax.grid(which="major", axis="y", alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10)


def _save(fig, axes, directory, filename):
    import matplotlib.pyplot as plt

    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="outside upper center",
        ncol=3,
        frameon=False,
        fontsize=11,
    )
    fig.savefig(directory / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _build_pair(directory, report, regime, axis, cases):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    for name, label, color, style in cases:
        record = _curve(report, name, axis, regime)
        x_values = np.asarray(record["values"])
        lower, upper = AXES[axis]["limits"]
        selected = (x_values >= lower) & (x_values <= upper)
        demand = np.asarray(record["demand"])
        price = x_values if axis == "price" else np.full_like(x_values, 10.0)
        # Demand is shown in trillions of tokens. Since price is dollars per
        # million tokens, demand * price / 1e12 is spending in USD millions.
        series = (demand / 1e12, demand * price / 1e12)
        for ax, values in zip(axes, series):
            ax.plot(
                x_values[selected],
                values[selected],
                label=label,
                color=color,
                linestyle=style,
                linewidth=2.5,
            )

    _format(axes[0], "(a) Token demand", axis, "Token demand (trillion tokens)")
    _format(axes[1], "(b) Token spending", axis, "Token spending (USD millions)")
    filename = f"{regime}-{axis}-demand-spending.png"
    _save(fig, axes, directory, filename)
    return filename


def build_exposition_figures(directory: Path, paradigms: dict, interventions: dict):
    """Build the six figures used in the short manuscript.

    ``interventions`` remains in the signature because the notebook calls this
    function alongside the intervention report. The short paper does not use
    those additional experiments.
    """

    del interventions
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    for regime, cases in (("work", WORK_CASES), ("attention", ATTENTION_CASES)):
        for axis in ("capability", "efficiency", "price"):
            files.append(_build_pair(directory, paradigms, regime, axis, cases))
    return tuple(files)
