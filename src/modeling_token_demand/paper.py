"""Build and locally preview an interactive edition of README.md.

Prose is always read from Markdown; numerical data is cached separately. The
preview only serves generated files and binds to the loopback interface.
"""

import argparse
from functools import partial
import hashlib
from html import escape
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import shutil
import tempfile
import threading
from urllib.parse import urlsplit, unquote

import mistune
from mistune.plugins.math import INLINE_MATH_PATTERN, math, parse_inline_math
from mistune.toc import add_toc_hook, render_toc_ul

from .paper_data import load_plot_data


ASSETS = Path(__file__).with_name("paper_assets")


def manuscript_math(md):
    math(md)
    # A closing delimiter cannot follow whitespace. Otherwise "$10; cost is
    # $x$" could swallow ordinary prose as a formula. Code spans stay literal.
    pattern = INLINE_MATH_PATTERN.replace(r'\$(?!\d)', r'(?<!\s)\$(?!\d)')
    md.inline.register("inline_math", pattern, parse_inline_math, before="codespan")


class PaperRenderer(mistune.HTMLRenderer):
    def __init__(self, plots):
        super().__init__(escape=True)
        self.plots = plots
        self.used_plots = {}

    def block_code(self, code, info=None):
        if info and info.strip() == "math":
            return '<div class="math display-math">\\[\n' + escape(code) + '\\]</div>\n'
        return super().block_code(code, info)

    def image(self, text, url, title=None):
        filename = Path(urlsplit(url).path).name
        if url.startswith("figures/") and filename in self.plots:
            identifier = Path(filename).stem
            self.used_plots[identifier] = self.plots[filename]
            return (
                f'<figure class="interactive-figure" id="{escape(identifier)}" '
                f'data-chart="{escape(identifier)}">'
                f'<figcaption>{escape(text)}</figcaption>'
                '<div class="chart-controls"></div><div class="chart-legend"></div>'
                '<div class="chart-panels"></div>'
                f'<img class="chart-fallback" src="{escape(url)}" alt="{escape(text)}">'
                '<p class="chart-help">Interactive controls need JavaScript. '
                'The original figure is shown until the chart is ready.</p></figure>'
            )
        return super().image(text, url, title)

    def paragraph(self, text):
        if text.startswith('<figure ') and text.endswith('</figure>'):
            return text + "\n"
        return super().paragraph(text)


def render_paper(markdown: str, plots: dict) -> tuple[str, dict]:
    renderer = PaperRenderer(plots)
    md = mistune.create_markdown(renderer=renderer, plugins=["table", manuscript_math, "strikethrough"])
    used_ids = set()

    def heading_id(token, index):
        base = re.sub(r"[^\w\s-]", "", token["text"].lower())
        base = re.sub(r"\s+", "-", base).strip("-") or f"section-{index}"
        value, suffix = base, 1
        while value in used_ids:
            suffix += 1
            value = f"{base}-{suffix}"
        used_ids.add(value)
        return value

    add_toc_hook(md, min_level=2, max_level=3, heading_id=heading_id)
    body, state = md.parse(markdown)
    headings = state.env.get("toc_items", [])
    toc = render_toc_ul(headings)
    # The exact visible title is taken from the manuscript, not a second copy.
    title_match = re.search(r"^#\s+(.+)$", markdown, re.MULTILINE)
    title = title_match.group(1) if title_match else "Research paper"
    data = json.dumps(renderer.used_plots, separators=(",", ":"), allow_nan=False)
    # Script text is raw HTML: escaping < prevents a label closing its element.
    data = data.replace("<", "\\u003c").replace("&", "\\u0026")
    template = (ASSETS / "page.html").read_text()
    replacements = {"TITLE": escape(title), "TOC": toc, "BODY": body, "CHART_DATA": data}
    html = re.sub(r"\{\{(TITLE|TOC|BODY|CHART_DATA)\}\}",
                  lambda match: replacements[match.group(1)], template)
    return html, renderer.used_plots


def copy_local_images(root: Path, output: Path, html: str) -> None:
    """Copy referenced local images, never expose the source checkout as HTTP."""
    for url in re.findall(r'<img[^>]+src="([^"]+)"', html):
        parsed = urlsplit(url)
        if parsed.scheme or parsed.netloc or parsed.path.startswith("/"):
            continue
        relative = Path(unquote(parsed.path))
        source = (root / relative).resolve()
        target = (output / relative).resolve()
        if root not in source.parents or output not in target.parents:
            raise ValueError(f"Image must stay inside the paper directory: {url}")
        if not source.is_file():
            raise FileNotFoundError(f"Missing manuscript image: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def build(root: Path, output: Path, refresh=False) -> Path:
    from plotly.offline import get_plotlyjs

    root, output = root.resolve(), output.resolve()
    if output == root or output in root.parents:
        raise ValueError("Use a dedicated output directory, not the source directory.")
    plots = load_plot_data(root, refresh=refresh)
    manuscript = (root / "README.md").read_text()
    html, used = render_paper(manuscript, plots)
    output.mkdir(parents=True, exist_ok=True)
    assets = output / "assets"
    assets.mkdir(exist_ok=True)
    for name in ("paper.css", "paper.js"):
        shutil.copyfile(ASSETS / name, assets / name)
    # Package the plotting runtime locally: opening index.html needs no Python.
    plotly_path = assets / "plotly.min.js"
    import plotly
    version_file = assets / "plotly-version.txt"
    if not plotly_path.exists() or not version_file.exists() or version_file.read_text() != plotly.__version__:
        plotly_path.write_text(get_plotlyjs())
        version_file.write_text(plotly.__version__)
    copy_local_images(root, output, html)
    # The gallery links its generated, non-executable numerical diagnostics.
    # Copy this specific artifact, not arbitrary workspace links.
    for name in ("paradigms.json", "interventions.json"):
        diagnostics = Path("figures") / name
        if str(diagnostics) in html and (root / diagnostics).is_file():
            (output / diagnostics).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / diagnostics, output / diagnostics)
    (output / "README.md").write_text(manuscript)
    # Manual builds and a watch rebuild can overlap. Each stages its own file.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output,
                                     prefix=".paper-", suffix=".tmp", delete=False) as staged:
        temporary = Path(staged.name)
        staged.write(html)
    try:
        temporary.replace(output / "index.html")
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Built {output / 'index.html'} ({len(used)} interactive figures).", flush=True)
    return output / "index.html"


def watch_signature(root: Path) -> str:
    """Ignore notebook outputs, generated HTML, and cache writes."""
    from .paper_data import source_fingerprint

    digest = hashlib.sha256(source_fingerprint(root).encode())
    paths = [root / "README.md", Path(__file__), *sorted(ASSETS.glob("*")),
             *sorted((root / "figures").glob("*.png"))]
    for path in paths:
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


class PreviewState:
    def __init__(self):
        self.version = 1
        self.building = False
        self.error = ""

    def payload(self):
        return {"version": self.version, "building": self.building, "error": self.error}


def watch(root: Path, output: Path, state: PreviewState, stop: threading.Event, interval=0.8):
    signature = watch_signature(root)
    while not stop.wait(interval):
        try:
            updated = watch_signature(root)
            if updated == signature:
                continue
            signature = updated
            state.building, state.error = True, ""
            build(root, output)
            # Keep the pre-build signature: an edit made during a long refresh
            # must be noticed on the next pass, not silently marked as rendered.
            state.version += 1
        except Exception as error:
            state.error = str(error)
            print(f"Build failed; keeping the last good page: {error}", flush=True)
        finally:
            state.building = False


class PreviewHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, state, **kwargs):
        self.state = state
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if urlsplit(self.path).path == "/__paper_status":
            data = json.dumps(self.state.payload()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            super().do_GET()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def list_directory(self, path):
        self.send_error(404)
        return None

    def log_message(self, format, *args):
        pass


def serve(root: Path, output: Path, port=8000):
    build(root, output)
    state, stop = PreviewState(), threading.Event()
    handler = partial(PreviewHandler, directory=str(output), state=state)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=watch, args=(root, output, state, stop), daemon=True)
    thread.start()
    print(f"Local paper: http://127.0.0.1:{server.server_port}\n"
          "Watching README.md, figure images, and model sources. Ctrl-C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "serve", "refresh"))
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project containing README.md")
    parser.add_argument("--output", type=Path, default=Path("build/paper"))
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (root / args.output).resolve()
    if args.command == "serve":
        serve(root, output, args.port)
    else:
        build(root, output, refresh=args.command == "refresh")


if __name__ == "__main__":
    main()
