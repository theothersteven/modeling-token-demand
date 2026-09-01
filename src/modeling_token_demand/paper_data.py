"""Export the notebook's actual plotted coordinates, without a second model.

The notebook remains the numerical source of truth. This adapter records its
Matplotlib figures and only publishes the cache after all notebook audits pass.
"""

from contextlib import redirect_stdout
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys


SCHEMA_VERSION = 1


def source_fingerprint(root: Path) -> str:
    notebook = json.loads((root / "notebooks/comparative_statics.ipynb").read_text())
    sources = ["".join(cell["source"]) for cell in notebook["cells"]
               if cell["cell_type"] == "code"]
    for name in ("model.py", "optimizer.py", "calibrations.py", "paradigms.py", "interventions.py", "paper_figures.py", "__init__.py", "paper_data.py"):
        sources.append((root / "src/modeling_token_demand" / name).read_text())
    return hashlib.sha256("\n".join(sources).encode()).hexdigest()


def figure_payload(figure) -> dict:
    """Translate the existing line plots, including guides and axis direction."""
    from matplotlib.colors import to_hex

    panels = []
    first_labels = [line.get_label() for line in figure.axes[0].lines
                    if len(line.get_xdata()) > 2]
    shared_label = figure._supylabel.get_text() if figure._supylabel else ""
    for axis in figure.axes:
        lines, guides, points = [], [], []
        series_index = 0
        for line in axis.lines:
            xs = [float(value) for value in line.get_xdata()]
            ys = [float(value) for value in line.get_ydata()]
            if len(xs) == 2 and line.get_transform() == axis.get_xaxis_transform():
                guides.append({"axis": "x", "value": xs[0]})
                continue
            if len(ys) == 2 and line.get_transform() == axis.get_yaxis_transform():
                guides.append({"axis": "y", "value": ys[0]})
                continue
            label = line.get_label()
            # The shadow-elasticity panel omits duplicate Matplotlib labels.
            if label.startswith("_") and series_index < len(first_labels):
                label = first_labels[series_index]
            series_index += 1
            lines.append({
                "name": label, "x": xs, "y": ys,
                "color": to_hex(line.get_color()),
                "dash": {"--": "dash", "-.": "dashdot", ":": "dot"}.get(
                    line.get_linestyle(), "solid"),
                "marker": {
                    "o": "circle", "s": "square", "^": "triangle-up",
                    "v": "triangle-down", "<": "triangle-left",
                    ">": "triangle-right", "D": "diamond", "x": "x",
                }.get(line.get_marker()),
                "width": float(line.get_linewidth()),
                "order": float(line.get_zorder()),
            })
        for collection in axis.collections:
            identifier = collection.get_gid() or ""
            if not identifier.startswith("demand-peak:"):
                continue
            offsets = collection.get_offsets()
            colors = collection.get_facecolors()
            sizes = collection.get_sizes()
            if len(offsets) != 1 or not len(colors):
                raise ValueError("A demand-peak marker must contain one colored point")
            points.append({
                "name": identifier.removeprefix("demand-peak:"),
                "kind": "demand-peak",
                "x": float(offsets[0][0]),
                "y": float(offsets[0][1]),
                "color": to_hex(colors[0]),
                "marker": "star",
                "size": math.sqrt(float(sizes[0])) if len(sizes) else 11.0,
            })
        panels.append({
            "title": axis.get_title(loc="left") or axis.get_title(),
            "xlabel": axis.get_xlabel(), "ylabel": axis.get_ylabel() or shared_label,
            "xscale": axis.get_xscale(), "yscale": axis.get_yscale(),
            "xlim": list(axis.get_xlim()), "ylim": list(axis.get_ylim()),
            "xticks": [float(value) for value in axis.get_xticks()
                       if min(axis.get_xlim()) <= value <= max(axis.get_xlim())],
            "lines": lines, "guides": guides, "points": points,
        })
    return {
        "panels": panels,
        "columns": getattr(
            figure,
            "_paper_columns",
            figure.axes[0].get_subplotspec().get_gridspec().ncols,
        ),
        "center_last": bool(getattr(figure, "_paper_center_last", False)),
        "shared_y": len(panels) > 1 and all(
            figure.axes[0].get_shared_y_axes().joined(figure.axes[0], axis)
            for axis in figure.axes[1:]
        ),
    }


def export_notebook(root: Path, destination: Path) -> None:
    """Execute trusted project code in a separate process, with non-GUI plots."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    from unittest.mock import patch

    notebook_path = root / "notebooks/comparative_statics.ipynb"
    notebook = json.loads(notebook_path.read_text())
    fingerprint = source_fingerprint(root)
    plots = {}
    original_savefig = Figure.savefig

    def capture(figure, filename, *args, **kwargs):
        plots[Path(filename).name] = figure_payload(figure)
        return original_savefig(figure, filename, *args, **kwargs)

    namespace = {"__name__": "__main__"}
    with patch.object(Figure, "savefig", capture):
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            print(f"Running notebook cell {index + 1}/{len(notebook['cells'])}...", flush=True)
            exec(compile("".join(cell["source"]), f"{notebook_path}:cell-{index}", "exec"), namespace)
    plt.close("all")
    if source_fingerprint(root) != fingerprint:
        raise RuntimeError("Model or notebook changed during export; run refresh again.")
    payload = {"schema": SCHEMA_VERSION, "fingerprint": fingerprint, "plots": plots}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False))
    temporary.replace(destination)
    print(f"Exported {len(plots)} figures; all notebook audits passed.", flush=True)


def load_plot_data(root: Path, refresh: bool = False) -> dict:
    destination = root / "figures/interactive.json"
    fingerprint = source_fingerprint(root)
    if not refresh and destination.exists():
        payload = json.loads(destination.read_text())
        if payload.get("schema") == SCHEMA_VERSION and payload.get("fingerprint") == fingerprint:
            return payload["plots"]
    print("Refreshing chart data from the notebook (this can take several minutes).", flush=True)
    subprocess.run(
        [sys.executable, "-m", "modeling_token_demand.paper_data", str(root), str(destination)],
        cwd=root, check=True,
    )
    return json.loads(destination.read_text())["plots"]


if __name__ == "__main__":
    export_notebook(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
