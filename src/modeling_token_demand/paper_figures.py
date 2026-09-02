"""Small indexed figures for the paper's main comparisons.

The notebook remains the source of the optimized outcomes.  This module only
selects audited one-parameter-at-a-time curves.  Capability and efficiency
figures pair the regime's expansion margin with token demand; the work-limited
price figure also shows effort and spending while restoring the
constant-revenue benchmark.
"""

from pathlib import Path

import numpy as np


WORK_CASES = (
    ("Reference industry", "Reference industry", "#595959", "-", None),
    ("Low adoption hurdle", "Low adoption hurdle", "#1f77b4", "--", "o"),
    ("High adoption hurdle", "High adoption hurdle", "#ff7f0e", "--", "s"),
    ("Hard execution", "Hard execution", "#2ca02c", "--", "^"),
    (
        "High capability requirement",
        "High capability requirement",
        "#d62728",
        "--",
        "x",
    ),
)
# Retain the adoption-hurdle cases even though their reservation-price curves
# coincide with the reference.  Their separate controls make that invariance
# directly inspectable in the interactive figure.
WORK_RESERVATION_CASES = WORK_CASES
ATTENTION_CASES = (
    ("Reference industry", "Reference industry", "#595959", "-", None),
    ("Hard execution", "Hard execution", "#1f77b4", "--", "^"),
    ("Low inference returns", "Low inference returns", "#ff7f0e", "--", "o"),
    ("Slow-growing review", "Slow-growing review", "#2ca02c", "--", "s"),
    ("Nearly proportional review", "Nearly proportional review", "#d62728", "--", "x"),
)

# Every displayed token-price axis stops before the low-inference-returns case
# reaches the economic effort floor.  The wider price sweep remains in the
# numerical report for robustness checks.
TOKEN_PRICE_LIMITS = (.1, 4.0)
TOKEN_PRICE_TICKS = (.1, .2, .5, 1, 2, 4)

# A modest capability regression keeps every reservation price positive and
# away from the zero-price/infinite-inference limit, while m=30 shows how the
# minimum viable effort regularizes large capability improvements.
RESERVATION_CAPABILITY_LIMITS = (.8, 30.0)
RESERVATION_CAPABILITY_TICKS = (.8, 1, 2, 5, 10, 30)

LEVER_CASES = (
    ("Reference model", "Reference model", "#595959", "-", None),
    ("Higher capability (m = 5)", "Higher capability ($m=5$)",
     "#1f77b4", "--", "o"),
    ("Hard execution (a = 1)", "Hard execution ($a=1$)",
     "#ff7f0e", "--", "^"),
)

LEVER_AXES = {
    "review-growth": {
        "xlabel": r"Review elasticity, $\beta$ (more scalable to the right)",
        "ticks": (.05, .25, .5, .75, .95), "baseline": .5,
        "reverse": True, "xscale": "linear",
        "filename": "lever-review-elasticity-adoption-revenue.png",
    },
    "verification-speed": {
        "xlabel": r"Review-cost multiplier, $\kappa_h$ (lower is better)",
        "ticks": (.1, .25, .5, 1, 2), "baseline": 1,
        "reverse": True, "xscale": "log",
        "filename": "lever-review-cost-adoption-revenue.png",
    },
    "inference-returns": {
        "xlabel": r"Marginal inference returns, $\alpha$",
        "ticks": (.1, .3, .5, .7, .9), "baseline": .5,
        "reverse": False, "xscale": "linear",
        "filename": "lever-inference-returns-adoption-revenue.png",
    },
}

AXES = {
    "capability": {
        "xlabel": "Model capability, m",
        "ticks": (.1, .25, .5, 1, 2, 5, 10, 30),
        "limits": (.1, 30),
        "baseline": 1,
        "reverse": False,
    },
    "efficiency": {
        "xlabel": r"Token efficiency, $\eta$",
        "ticks": (.25, .5, 1, 2, 5, 10),
        "limits": (.25, 10),
        "baseline": 1,
        "reverse": False,
    },
    "price": {
        "xlabel": "Normalized token price, c (cheaper to the right)",
        "ticks": TOKEN_PRICE_TICKS,
        "limits": TOKEN_PRICE_LIMITS,
        "baseline": 1,
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


def _format(ax, title, axis, ylabel, *, normalized=True, linear=False):
    spec = AXES[axis]
    ax.set_title(title, loc="left", fontsize=12, pad=12)
    ax.set_xlabel(spec["xlabel"], fontsize=11, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=11, labelpad=8)
    ax.set_xscale("log")
    ax.set_yscale("linear" if linear else "log")
    ax.set_xticks(spec["ticks"], [f"{tick:g}" for tick in spec["ticks"]])
    left, right = spec["limits"]
    ax.set_xlim((right, left) if spec["reverse"] else (left, right))
    ax.axvline(spec["baseline"], color=".68", linestyle=":", linewidth=1)
    if normalized:
        ax.axhline(1, color=".72", linestyle=":", linewidth=1)
    ax.grid(which="major", axis="y", alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10)


def _save(fig, axes, directory, filename):
    import matplotlib.pyplot as plt

    handles_by_label = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            handles_by_label.setdefault(label, handle)
    labels = list(handles_by_label)
    handles = [handles_by_label[label] for label in labels]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(.5, .98),
        ncol=3 if len(labels) > 4 else len(labels),
        frameon=False,
        fontsize=11,
    )
    fig.savefig(directory / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _indexed(record, key):
    values = np.asarray(record[key], dtype=float)
    return values / values[record["baseline_index"]]


def _constant_revenue(ax, prices):
    baseline = AXES["price"]["baseline"]
    ax.plot(
        prices,
        baseline / prices,
        color="black",
        linestyle=":",
        linewidth=1.4,
        marker=None,
        label="Constant revenue",
        zorder=12,
    )
    ax.text(
        .04,
        .94,
        "Above black line: spending is\nhigher than at the baseline price",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .86, "pad": 2},
        zorder=13,
    )


def _revenue_neutral_efficiency(ax, efficiencies):
    """At a fixed token price, unchanged demand means unchanged revenue."""

    ax.plot(
        efficiencies,
        np.ones_like(efficiencies),
        color="black",
        linestyle=":",
        linewidth=1.4,
        marker=None,
        label="Revenue neutral (price fixed)",
        zorder=12,
    )
    ax.text(
        .04,
        .94,
        "Above black line: token revenue is\nhigher than at $\\eta=1$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .86, "pad": 2},
        zorder=13,
    )


def _build_pair(directory, report, regime, axis, cases):
    import matplotlib.pyplot as plt

    mechanism_view = regime == "work" and axis == "efficiency"
    price_effort_view = regime == "work" and axis == "price"
    if mechanism_view:
        fig, axes_grid = plt.subplots(2, 2, figsize=(12.5, 9.2))
        axes = axes_grid.ravel()
        fig.subplots_adjust(
            left=.08, right=.98, bottom=.09, top=.82, wspace=.24, hspace=.42,
        )
    elif price_effort_view:
        fig, axes_grid = plt.subplots(2, 2, figsize=(12.5, 9.2))
        axes = axes_grid.ravel()
        fig.subplots_adjust(
            left=.08, right=.98, bottom=.09, top=.82, wspace=.24, hspace=.42,
        )
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
        fig.subplots_adjust(left=.08, right=.98, bottom=.16, top=.72, wspace=.24)
    for name, label, color, style, marker in cases:
        record = _curve(report, name, axis, regime)
        x_values = np.asarray(record["values"])
        lower, upper = AXES[axis]["limits"]
        selected = (x_values >= lower) & (x_values <= upper)
        demand = _indexed(record, "demand")
        if axis == "price":
            price = x_values
            baseline = record["baseline_index"]
            spending = (
                np.asarray(record["demand"]) * price
                / (record["demand"][baseline] * price[baseline])
            )
            if price_effort_view:
                adoption = 100 * np.asarray(record["adoption"])
                effort_level = np.asarray(record["tokens_per_assigned_work"])
                series = (adoption, demand, effort_level, spending)
            else:
                series = (demand, spending)
        elif regime == "work":
            adoption = 100 * np.asarray(record["adoption"])
            if mechanism_view:
                effort_level = np.asarray(record["tokens_per_assigned_work"])
                effective_inference = x_values * effort_level
                series = (adoption, demand, effort_level, effective_inference)
            else:
                series = (adoption, demand)
        else:
            series = (_indexed(record, "leverage"), demand)
        for ax, values in zip(axes, series):
            plotted_x = x_values[selected]
            ax.plot(
                plotted_x,
                values[selected],
                label=label,
                color=color,
                linestyle=style,
                linewidth=2.5,
                marker=marker,
                markevery=max(1, len(plotted_x) // 9) if marker else None,
                markersize=5.5,
                markerfacecolor="white" if marker else color,
                markeredgewidth=1.1,
            )

        if regime == "work" and axis == "capability":
            plotted_indices = np.flatnonzero(selected)
            peak_index = plotted_indices[np.argmax(demand[selected])]
            peak = axes[0].scatter(
                [x_values[peak_index]],
                [adoption[peak_index]],
                marker="*",
                s=256,
                color=color,
                edgecolor="white",
                linewidth=.9,
                zorder=20,
                clip_on=False,
            )
            # The interactive exporter attaches this marker to the same
            # visibility control as the corresponding line.
            peak.set_gid(f"demand-peak:{label}")

    if mechanism_view:
        _format(
            axes[0], "(a) Adoption", axis,
            "Potential work assigned to AI (%)", normalized=False, linear=True,
        )
        axes[0].set_ylim(0, 102)
        _format(
            axes[1], "(b) Token demand", axis,
            "Token demand index (baseline = 1)",
        )
        efficiency_values = np.asarray(
            _curve(report, cases[0][0], axis, regime)["values"]
        )
        selected_efficiencies = efficiency_values[
            (efficiency_values >= AXES[axis]["limits"][0])
            & (efficiency_values <= AXES[axis]["limits"][1])
        ]
        _revenue_neutral_efficiency(axes[1], selected_efficiencies)
        _format(
            axes[2], "(c) Optimal model effort level", axis,
            "Model effort level, x", normalized=False,
        )
        _format(
            axes[3], "(d) Effective inference", axis,
            r"Effective inference per work unit, $\eta x$", normalized=False,
        )
    elif axis == "price":
        price_values = np.asarray(_curve(report, cases[0][0], axis, regime)["values"])
        price_values = price_values[
            (price_values >= TOKEN_PRICE_LIMITS[0])
            & (price_values <= TOKEN_PRICE_LIMITS[1])
        ]
        if price_effort_view:
            _format(
                axes[0], "(a) Adoption", axis,
                "Potential work assigned to AI (%)", normalized=False, linear=True,
            )
            axes[0].set_ylim(0, 102)
            _constant_revenue(axes[1], price_values)
            _format(
                axes[1], "(b) Token demand", axis,
                "Token demand index (baseline = 1)",
            )
            _format(
                axes[2], "(c) Optimal model effort level", axis,
                "Model effort level, x", normalized=False,
            )
            _format(
                axes[3], "(d) Token spending", axis,
                "Token spending index (baseline = 1)",
            )
        else:
            _constant_revenue(axes[0], price_values)
            _format(
                axes[0], "(a) Token demand", axis,
                "Token demand index (baseline = 1)",
            )
            _format(
                axes[1], "(b) Token spending", axis,
                "Token spending index (baseline = 1)",
            )
    elif regime == "work":
        _format(
            axes[0], "(a) Adoption", axis,
            "Potential work assigned to AI (%)", normalized=False, linear=True,
        )
        axes[0].set_ylim(0, 102)
        _format(
            axes[1], "(b) Token demand", axis,
            "Token demand index (baseline = 1)",
        )
    else:
        _format(
            axes[0], "(a) Work per review hour", axis,
            "Supervisory leverage index (baseline = 1)",
        )
        _format(
            axes[1], "(b) Token demand", axis,
            "Token demand index (baseline = 1)",
        )
    filename = f"{regime}-{axis}-demand-spending.png"
    _save(fig, axes, directory, filename)
    return filename


def _build_work_reservation_price(directory, report):
    """Plot the exact work-limited reservation token price."""

    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, NullLocator

    axis = "capability"
    fig, ax = plt.subplots(1, 1, figsize=(7.4, 5.2))
    fig.subplots_adjust(left=.14, right=.98, bottom=.16, top=.72)
    lower, upper = RESERVATION_CAPABILITY_LIMITS
    for case_index, (name, label, color, style, marker) in enumerate(
        WORK_RESERVATION_CASES
    ):
        record = _curve(report, name, axis, "work")
        reservation_capability = np.asarray(
            record["reservation_capability"], dtype=float
        )
        selected = (
            (reservation_capability >= lower)
            & (reservation_capability <= upper)
        )
        marker_step = max(1, np.count_nonzero(selected) // 9)
        ax.plot(
            reservation_capability[selected],
            np.asarray(record["reservation_price"])[selected]
            / record["reservation_price_baseline"],
            label=label,
            color=color,
            linestyle=style,
            linewidth=2.5,
            marker=marker,
            # Offset markers across cases so coincident hurdle-only curves are
            # visible in the static figure as well as separately toggleable.
            markevery=(case_index % marker_step, marker_step) if marker else None,
            markersize=5.5,
            markerfacecolor="white" if marker else color,
            markeredgewidth=1.1,
        )

    _format(
        ax, "Token price preserving baseline work-limited surplus", axis,
        r"Reservation token price, $c_{\rm res}^{W}(m)/c_0$",
    )
    ax.set_xlim(lower, upper)
    ax.set_xticks(
        RESERVATION_CAPABILITY_TICKS,
        [f"{tick:g}" for tick in RESERVATION_CAPABILITY_TICKS],
    )
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    filename = "work-capability-reservation-price.png"
    _save(fig, [ax], directory, filename)
    return filename


def _build_attention_value(directory, report):
    """Plot hourly attention value and the exact reservation token price."""

    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, NullLocator

    axis = "capability"
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    fig.subplots_adjust(left=.08, right=.98, bottom=.16, top=.72, wspace=.24)
    for name, label, color, style, marker in ATTENTION_CASES:
        record = _curve(report, name, axis, "attention")
        x_values = np.asarray(record["values"], dtype=float)
        lower, upper = RESERVATION_CAPABILITY_LIMITS
        selected = (x_values >= lower) & (x_values <= upper)
        reservation_capability = np.asarray(
            record["reservation_capability"], dtype=float
        )
        reservation_selected = (
            (reservation_capability >= lower)
            & (reservation_capability <= upper)
        )
        series = (
            (x_values[selected], _indexed(record, "attention_value")[selected]),
            (
                reservation_capability[reservation_selected],
                np.asarray(record["reservation_price"])[reservation_selected]
                / record["reservation_price_baseline"],
            ),
        )
        for ax, (plotted_x, plotted_y) in zip(axes, series):
            ax.plot(
                plotted_x,
                plotted_y,
                label=label,
                color=color,
                linestyle=style,
                linewidth=2.5,
                marker=marker,
                markevery=max(1, len(plotted_x) // 9) if marker else None,
                markersize=5.5,
                markerfacecolor="white" if marker else color,
                markeredgewidth=1.1,
            )

    _format(
        axes[0], "(a) Value of reviewer attention", axis,
        "Reviewer-attention value index (baseline = 1)",
    )
    _format(
        axes[1], "(b) Token price preserving baseline user surplus", axis,
        r"Reservation token price, $c_{\rm res}^{H}(m)/c_0$",
    )
    for ax in axes:
        ax.set_xlim(lower, upper)
        ax.set_xticks(
            RESERVATION_CAPABILITY_TICKS,
            [f"{tick:g}" for tick in RESERVATION_CAPABILITY_TICKS],
        )
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
    filename = "attention-capability-value.png"
    _save(fig, axes, directory, filename)
    return filename


def _intervention_curve(report, experiment, label):
    return next(
        record for record in report["curves"]
        if record["experiment"] == experiment
        and record["regime"] == "work"
        and record["label"] == label
    )


def _build_lever_pair(directory, report, experiment):
    """Plot work-limited adoption and revenue for one capability lever."""
    import matplotlib.pyplot as plt

    spec = LEVER_AXES[experiment]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    fig.subplots_adjust(left=.08, right=.98, bottom=.16, top=.72, wspace=.24)
    for source_label, label, color, style, marker in LEVER_CASES:
        record = _intervention_curve(report, experiment, source_label)
        x_values = np.asarray(record["values"])
        adoption = 100 * np.asarray(record["adoption"])
        # Price is fixed at c=1, so supplier revenue has the same path as
        # demand. Indexing within each case makes the mechanisms comparable.
        revenue = _indexed(record, "demand")
        for ax, values in zip(axes, (adoption, revenue)):
            ax.plot(
                x_values, values, label=label, color=color, linestyle=style,
                linewidth=2.5, marker=marker,
                markevery=max(1, len(x_values) // 9) if marker else None,
                markersize=5.5, markerfacecolor="white" if marker else color,
                markeredgewidth=1.1,
            )

    for ax in axes:
        ax.set_xscale(spec["xscale"])
        ax.set_xticks(spec["ticks"], [f"{tick:g}" for tick in spec["ticks"]])
        ax.set_xlim(min(spec["ticks"]), max(spec["ticks"]))
        if spec["reverse"]:
            ax.invert_xaxis()
        ax.set_xlabel(spec["xlabel"], fontsize=11, labelpad=8)
        ax.axvline(spec["baseline"], color=".68", linestyle=":", linewidth=1)
        ax.grid(which="major", axis="y", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=10)
    axes[0].set_title("(a) Adoption", loc="left", fontsize=12, pad=12)
    axes[0].set_ylabel("Potential work assigned to AI (%)", fontsize=11, labelpad=8)
    axes[0].set_ylim(0, 102)
    axes[1].set_title("(b) Token revenue", loc="left", fontsize=12, pad=12)
    axes[1].set_ylabel("Token revenue index (baseline = 1)", fontsize=11, labelpad=8)
    axes[1].set_yscale("log")
    axes[1].axhline(1, color=".72", linestyle=":", linewidth=1)
    _save(fig, axes, directory, spec["filename"])
    return spec["filename"]


def build_exposition_figures(directory: Path, paradigms: dict, interventions: dict):
    """Build eight regime figures and three capability-lever figures."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    for axis in ("capability", "efficiency", "price"):
        files.append(_build_pair(directory, paradigms, "work", axis, WORK_CASES))
    files.append(_build_work_reservation_price(directory, paradigms))
    for axis in ("capability", "efficiency", "price"):
        files.append(
            _build_pair(directory, paradigms, "attention", axis, ATTENTION_CASES)
        )
    files.append(_build_attention_value(directory, paradigms))
    for experiment in ("review-growth", "verification-speed", "inference-returns"):
        files.append(_build_lever_pair(directory, interventions, experiment))
    return tuple(files)
