"""Main-text figures composed from the notebook's already audited outcomes.

No model, optimizer, calibration, or fitted curve lives here. The larger
comparison gallery remains available; these views expose its economic margins.
"""

from pathlib import Path

import numpy as np


WORK_CASES = (
    ("Reference industry", "Reference", "#66717E", "-"),
    ("Adoption concentration: high", "Concentrated adoption", "#C45123", "--"),
    ("Early saturation", "Early saturation", "#007F78", "-."),
)
ATTENTION_CASES = (
    ("Verification burden: low", "Review grows slowly", "#007F78", "-"),
    ("Verification burden: high", "Review nearly proportional", "#785BA0", "--"),
)


def _curve(report, name, axis, regime):
    return next(r for r in report["curves"] if r["industry"]["name"] == name
                and r["axis"] == axis and r["regime"] == regime)


def _indexed(record, metric):
    values = np.asarray(record[metric])
    return values / values[record["baseline_index"]]


def _format(ax, title, xlabel, ylabel, ticks, baseline=1, *, log_y=False, reverse=False):
    ax.set_title(title, loc="left", fontsize=12, pad=12)
    ax.set_xlabel(xlabel, fontsize=11, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=11, labelpad=8)
    ax.set_xscale("log")
    ax.set_xticks(ticks, [f"{tick:g}" for tick in ticks])
    if log_y:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
    if reverse:
        ax.set_xlim(max(ticks), min(ticks))
    else:
        ax.set_xlim(min(ticks), max(ticks))
    ax.axvline(baseline, color=".72", linestyle=":", linewidth=1)
    if "baseline" in ylabel:
        ax.axhline(1, color=".72", linestyle=":", linewidth=1)
    ax.grid(which="major", axis="y", alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10)


def _save(fig, axes, directory, name):
    import matplotlib.pyplot as plt

    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center",
               ncol=3, frameon=False, fontsize=11)
    fig.savefig(directory / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_exposition_figures(directory: Path, paradigms: dict, interventions: dict):
    """Four questions, with explicit denominators and at most three case curves."""
    import matplotlib.pyplot as plt

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7), constrained_layout=True)
    for name, label, color, style in WORK_CASES:
        record = _curve(paradigms, name, "price", "work")
        prices = np.asarray(record["values"])
        selected = prices <= 20  # Focus on the adoption transition, not remote tails.
        demand = _indexed(record, "demand")
        for ax, values in zip(axes, (100 * np.asarray(record["adoption"]), demand,
                                     prices / 10 * demand)):
            ax.plot(prices[selected], values[selected], label=label,
                    color=color, linestyle=style, linewidth=2.5)
    for ax, title, ylabel in zip(axes,
            ("(a) Adoption", "(b) Token purchases", "(c) Token spending"),
            ("Potential work assigned to AI (%)", "Tokens (baseline = 1)",
             "Spending (baseline = 1)")):
        _format(ax, title, "Token price, USD / million (cheaper →)", ylabel,
                (1, 2, 5, 10, 20), baseline=10, log_y=ax is axes[1], reverse=True)
    axes[0].set_ylim(0, 102)
    _save(fig, axes, directory, "price-adoption-and-spending.png")

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7), constrained_layout=True)
    for name, label, color, style in WORK_CASES:
        record = _curve(paradigms, name, "capability", "work")
        series = (100 * np.asarray(record["adoption"]), _indexed(record, "x"),
                  _indexed(record, "demand"))
        for ax, values in zip(axes, series):
            ax.plot(record["values"], values, label=label, color=color,
                    linestyle=style, linewidth=2.5)
    for ax, title, ylabel in zip(axes,
            ("(a) More work uses AI", "(b) Fewer tokens per work unit", "(c) Total demand can peak"),
            ("Potential work assigned to AI (%)", "Tokens per work unit (baseline = 1)",
             "Token demand (baseline = 1)")):
        _format(ax, title, "Capability, m", ylabel, (.25, .5, 1, 2, 5, 10))
    axes[0].set_ylim(0, 102)
    _save(fig, axes, directory, "capability-work-decomposition.png")

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7), constrained_layout=True)
    for name, label, color, style in ATTENTION_CASES:
        record = _curve(paradigms, name, "capability", "attention")
        for ax, metric in zip(axes, ("leverage", "x", "demand")):
            ax.plot(record["values"], _indexed(record, metric), label=label,
                    color=color, linestyle=style, linewidth=2.5)
    for index, (ax, title, ylabel) in enumerate(zip(axes,
            ("(a) Work supervised per hour", "(b) Tokens per work unit", "(c) Total token demand"),
            ("Work per review hour (baseline = 1)", "Tokens per work unit (baseline = 1)",
             "Token demand (baseline = 1)"))):
        _format(ax, title, "Capability, m", ylabel, (.25, .5, 1, 2, 5, 10),
                log_y=index != 1)
        if index != 1:
            ax.set_yticks((.25, .5, 1, 2, 4, 8), ("0.25", "0.5", "1", "2", "4", "8"))
            ax.set_ylim(.25, 8)
    _save(fig, axes, directory, "capability-attention-decomposition.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    for ax, regime, title in zip(axes, ("work", "attention"),
            ("(a) Limited work: adoption and savings", "(b) Limited attention: more capacity")):
        record = next(r for r in interventions["curves"]
                      if r["experiment"] == "verification-speed" and r["regime"] == regime)
        speed = 1 / np.asarray(record["values"])
        for metric, label, color, style in (
                ("demand", "Tokens", "#355D8A", "-"),
                ("completed_work", "Completed work", "#007F78", "--")):
            ax.plot(speed, _indexed(record, metric), label=label,
                    color=color, linestyle=style, linewidth=2.5)
        _format(ax, title, "Review speed factor, 1 / v", "Index (baseline = 1)",
                (.5, 1, 2, 5, 10))
    _save(fig, axes, directory, "verification-expansion.png")

    return ("price-adoption-and-spending.png", "capability-work-decomposition.png",
            "capability-attention-decomposition.png", "verification-expansion.png")
