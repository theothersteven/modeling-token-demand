"""The paper is a view of the manuscript and the notebook, not a second source."""

from functools import partial
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import re
from html import escape
import threading
import time
from urllib.request import urlopen

import pytest

pytest.importorskip("mistune")
pytest.importorskip("plotly")

from modeling_token_demand import paper
from modeling_token_demand.paper_data import figure_payload, source_fingerprint

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def plot_data():
    return json.loads((ROOT / "figures/interactive.json").read_text())["plots"]


def test_entire_manuscript_renders_with_all_figures(plot_data):
    source = (ROOT / "README.md").read_text()
    html, figures = paper.render_paper(source, plot_data)
    image_names = re.findall(r'!\[[^\]]*\]\(figures/([^)]+)\)', source)
    assert set(figures) == {Path(name).stem for name in image_names}
    assert html.count('class="interactive-figure"') == len(image_names)
    assert '<p><figure' not in html
    assert 'id="introduction"' in html
    tldr = html.split('<h3 id="tldr">', 1)[1].split(
        '<h2 id="1-modeling-assumptions">', 1
    )[0]
    for target in (
        '21-token-demand-peaks-as-adoption-nearly-saturates',
        '22-token-efficiency-can-raise-demand-before-it-saves-tokens',
        '23-cheaper-tokens-raise-demand-but-spending-eventually-falls',
        '33-jevons-paradox-is-harder-to-reproduce-in-the-attention-limited-regime',
        '31-capability-raises-demand-when-verification-cost-grows-slowly',
        '34-capability-raises-the-value-of-scarce-attention',
        '24-capability-raises-the-work-limited-reservation-token-price',
    ):
        assert f'href="#{target}"' in tldr
    assert 'id="1-modeling-assumptions"' in html
    assert 'href="#1-modeling-assumptions"' in html
    assert 'id="3-human-attention-is-limited"' in html
    assert 'language-math' not in html
    assert '<div class="math display-math">\\[' in html
    assert '<span class="math">\\(' in html
    assert '<table>' in html
    assert '{{BODY}}' not in html
    equations = re.findall(r'```math\n(.*?)\n```', source, re.DOTALL)
    assert len(equations) == html.count('class="math display-math"')
    assert all(escape(equation) in html for equation in equations)
    payload = json.loads((ROOT / 'figures/interactive.json').read_text())
    assert payload['fingerprint'] == source_fingerprint(ROOT)


def test_indexed_figure_notes_move_into_interactive_footer(plot_data):
    source = (ROOT / "README.md").read_text()
    html, _ = paper.render_paper(source, plot_data)
    assert html.count('class="chart-context"') == 8
    assert html.count('class="chart-instructions"') == 8
    for identifier in paper.INDEXED_FIGURE_NOTES:
        figure = re.search(
            rf'<figure class="interactive-figure" id="{identifier}".*?</figure>',
            html,
            re.DOTALL,
        ).group(0)
        if identifier not in {
            'attention-capability-value', 'work-capability-reservation-price'
        }:
            assert 'rather than its absolute level' in figure \
                or 'rather than their absolute levels' in figure
        assert 'class="chart-context"' in figure
        assert 'class="chart-instructions"' in figure


def test_manual_text_and_equation_edits_are_rendered():
    original, _ = paper.render_paper('# Paper\n\nOld text.\n\n```math\nx^2\n```', {})
    revised, _ = paper.render_paper('# Paper\n\nNew text and $\\eta$.\n\n```math\nx^3\n```', {})
    assert 'Old text.' in original and 'Old text.' not in revised
    assert 'New text' in revised and 'x^3' in revised
    assert r'\eta' in revised


def test_manuscript_uses_eight_ordered_figures(plot_data):
    source = (ROOT / 'README.md').read_text()
    figures = re.findall(r'!\[[^\]]*\]\(figures/([^)]+)\)', source)
    assert figures == [
        'work-capability-demand-spending.png',
        'work-efficiency-demand-spending.png',
        'work-price-demand-spending.png',
        'work-capability-reservation-price.png',
        'attention-capability-demand-spending.png',
        'attention-efficiency-demand-spending.png',
        'attention-price-demand-spending.png',
        'attention-capability-value.png',
    ]
    assert '## Abstract' not in source
    assert source.index('## Introduction') < source.index('## 1. Modeling assumptions')
    assert source.index('## 1. Modeling assumptions') < source.index('## 2. Work is limited')
    assert source.index('## 2. Work is limited') < source.index('## 3. Human attention is limited')
    assert source.index('## 3. Human attention is limited') < source.index(
        '## 4. Conclusion')
    assert len(source.split()) < 5000
    for filename in figures:
        panels = plot_data[filename]['panels']
        if filename == 'work-efficiency-demand-spending.png':
            expected_panels = 4
        elif filename == 'work-price-demand-spending.png':
            expected_panels = 4
        elif filename == 'work-capability-reservation-price.png':
            expected_panels = 1
        else:
            expected_panels = 2
        assert len(panels) == expected_panels
        if filename == 'attention-capability-value.png':
            assert panels[1]['ylabel'] == (
                r'Reservation token price, $c_{\rm res}^{H}(m)/c_0$'
            )
        elif filename == 'work-capability-reservation-price.png':
            assert panels[0]['ylabel'] == (
                r'Reservation token price, $c_{\rm res}^{W}(m)/c_0$'
            )
        else:
            assert panels[1]['ylabel'].endswith('index (baseline = 1)')
    work_price = plot_data['work-price-demand-spending.png']['panels']
    assert work_price[1]['ylabel'] == 'Token demand index (baseline = 1)'
    assert work_price[3]['ylabel'] == 'Token spending index (baseline = 1)'
    attention_price = plot_data['attention-price-demand-spending.png']['panels']
    assert attention_price[0]['ylabel'] == 'Token demand index (baseline = 1)'
    assert attention_price[1]['ylabel'] == 'Token spending index (baseline = 1)'


def test_math_currency_code_and_duplicate_headings():
    html, _ = paper.render_paper(
        '# Paper\n\n## Same\n\n## Same\n\nPrice is $10; effort is $x_i$. '
        'Code is `$not_math$`.\n\n```math\na < b\n```', {})
    assert 'Price is $10;' in html
    assert '<span class="math">\\(x_i\\)</span>' in html
    assert '<code>$not_math$</code>' in html
    assert 'a &lt; b' in html
    assert 'id="same"' in html and 'id="same-2"' in html


def test_chart_data_cannot_close_script_tag():
    plot = {"panels": [{"title": '</script><script>alert(1)</script>'}]}
    html, _ = paper.render_paper('![test](figures/test.png)', {"test.png": plot})
    assert '</script><script>alert' not in html
    assert r'\u003c/script>' in html


def test_notebook_cache_preserves_baselines_overlap_and_price_direction(plot_data):
    for regime in ('work', 'attention'):
        spec = plot_data[f'{regime}-limited-token-demand-indexed.png']
        assert spec['shared_y'] == (regime == 'attention')
        assert [panel['yscale'] for panel in spec['panels']] == (
            ['log', 'log', 'log'] if regime == 'attention' else ['linear', 'linear', 'log'])
        for index, panel in enumerate(spec['panels']):
            baseline = 1
            for line in panel['lines']:
                point = line['x'].index(baseline)
                assert line['y'][point] == pytest.approx(1)
        assert spec['panels'][2]['xlim'][0] > spec['panels'][2]['xlim'][1]
        benchmark = spec['panels'][2]['lines'][-1]
        assert benchmark['name'].startswith('Constant revenue')
        assert benchmark['y'] == pytest.approx([1 / x for x in benchmark['x']])
    for suffix in ('levels', 'indexed'):
        for panel in plot_data[f'attention-limited-token-demand-{suffix}.png']['panels']:
            lines = {line['name']: line for line in panel['lines']}
            for name in ('High adoption hurdle', 'Low adoption hurdle'):
                assert lines[name]['y'] == pytest.approx(lines['Reference industry']['y'])


def test_every_displayed_token_price_axis_uses_the_shared_window(plot_data):
    price_panels = {
        filename: [
            panel for panel in spec['panels']
            if 'token price' in panel['xlabel'].lower()
        ]
        for filename, spec in plot_data.items()
    }
    price_panels = {
        filename: panels for filename, panels in price_panels.items() if panels
    }
    assert set(price_panels) == {
        'work-limited-token-demand-levels.png',
        'work-limited-token-demand-indexed.png',
        'attention-limited-token-demand-levels.png',
        'attention-limited-token-demand-indexed.png',
        'token-demand-vs-price.png',
        'token-spend-vs-price.png',
        'paradigm-work-demand.png',
        'paradigm-adoption-and-revenue.png',
        'work-price-demand-spending.png',
        'attention-price-demand-spending.png',
    }
    for panels in price_panels.values():
        for panel in panels:
            assert panel['xlim'] == pytest.approx([4, .1])
            assert panel['xticks'] == pytest.approx([.1, .2, .5, 1, 2, 4])


def test_export_retains_unlabeled_second_panel_and_guides():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2)
    axes[0].plot([1, 2, 3], [4, 5, 6], label='Reference industry')
    axes[1].plot([1, 2, 3], [.1, .2, .3])
    axes[0].axvline(1)
    axes[0].axhline(1)
    data = figure_payload(fig)
    assert not data['shared_y']
    assert data['panels'][1]['lines'][0]['name'] == 'Reference industry'
    assert data['panels'][0]['guides'] == [{'axis': 'x', 'value': 1}, {'axis': 'y', 'value': 1}]
    plt.close(fig)


def test_fingerprint_ignores_manuscript_and_notebook_outputs(tmp_path):
    source_dir = tmp_path / 'src/modeling_token_demand'
    source_dir.mkdir(parents=True)
    for name in ('model.py', 'optimizer.py', 'calibrations.py', 'paradigms.py', 'interventions.py', 'paper_figures.py', '__init__.py', 'paper_data.py'):
        (source_dir / name).write_text('# source')
    notebook_path = tmp_path / 'notebooks/comparative_statics.ipynb'
    notebook_path.parent.mkdir()
    notebook = {'cells': [{'cell_type': 'code', 'source': ['x = 1'], 'outputs': []},
                          {'cell_type': 'markdown', 'source': ['Notes']}]}
    notebook_path.write_text(json.dumps(notebook))
    initial = source_fingerprint(tmp_path)
    (tmp_path / 'README.md').write_text('Edited manuscript')
    notebook['cells'][0]['outputs'] = ['new output']
    notebook['cells'][1]['source'] = ['Edited notebook notes']
    notebook_path.write_text(json.dumps(notebook))
    assert source_fingerprint(tmp_path) == initial
    notebook['cells'][0]['source'] = ['x = 2']
    notebook_path.write_text(json.dumps(notebook))
    assert source_fingerprint(tmp_path) != initial


def test_gallery_uses_independent_axes_and_separate_adoption_and_revenue(plot_data):
    work = plot_data['paradigm-work-demand.png']
    assert not work['shared_y']
    assert [panel['yscale'] for panel in work['panels']] == ['log', 'log', 'log']
    assert work['panels'][2]['xlim'][0] > work['panels'][2]['xlim'][1]
    for index, panel in enumerate(work['panels']):
        baseline = 1
        for line in panel['lines']:
            assert line['y'][line['x'].index(baseline)] == pytest.approx(1)
    adoption = plot_data['paradigm-adoption-and-revenue.png']['panels'][0]
    for line in adoption['lines']:
        assert all(0 <= value <= 100 for value in line['y'])
    attention = plot_data['paradigm-attention-capability.png']['panels']
    assert len(attention) == 5
    assert [panel['lines'][0]['name'] for panel in attention] == [
        'Reference industry', 'Hard execution', 'Low inference returns',
        'Slow-growing review', 'Nearly proportional review',
    ]


def test_build_reuses_data_and_updates_text(tmp_path, monkeypatch):
    (tmp_path / 'README.md').write_text('# Paper\n\nInitial text.')
    monkeypatch.setattr(paper, 'load_plot_data', lambda root, refresh=False: {})
    output = tmp_path / 'build/paper'
    paper.build(tmp_path, output)
    assert 'Initial text.' in (output / 'index.html').read_text()
    (tmp_path / 'README.md').write_text('# Paper\n\nEdited text.')
    paper.build(tmp_path, output)
    assert 'Edited text.' in (output / 'index.html').read_text()
    assert (output / 'assets/plotly.min.js').is_file()
    assert (output / 'README.md').read_text().endswith('Edited text.')


@pytest.mark.parametrize('filename', ['paradigms.json', 'interventions.json'])
def test_build_copies_linked_gallery_diagnostics(tmp_path, monkeypatch, filename):
    (tmp_path / 'README.md').write_text(f'# Paper\n\n[Diagnostics](figures/{filename})')
    (tmp_path / 'figures').mkdir()
    payload = '{"audit":{"boundary_hits":[]}}'
    (tmp_path / 'figures' / filename).write_text(payload)
    monkeypatch.setattr(paper, 'load_plot_data', lambda root, refresh=False: {})
    output = tmp_path / 'build/paper'
    paper.build(tmp_path, output)
    assert (output / 'figures' / filename).read_text() == payload


def test_build_packages_linked_reading_files_without_exposing_other_files(tmp_path, monkeypatch):
    notebook = tmp_path / 'notebooks/comparative_statics.ipynb'
    notebook.parent.mkdir()
    notebook.write_text('{"cells":[]}')
    (tmp_path / 'REVIEW.md').write_text('# Review')
    (tmp_path / 'private.txt').write_text('not a paper artifact')
    (tmp_path / 'README.md').write_text(
        '# Paper\n\n[Notebook](notebooks/comparative_statics.ipynb)\n\n'
        '[Review](REVIEW.md)\n\n[Other](private.txt)')
    monkeypatch.setattr(paper, 'load_plot_data', lambda root, refresh=False: {})
    output = tmp_path / 'build/paper'
    paper.build(tmp_path, output)
    assert (output / 'notebooks/comparative_statics.ipynb').read_text() == notebook.read_text()
    assert (output / 'REVIEW.md').read_text() == '# Review'
    assert not (output / 'private.txt').exists()


def test_local_images_cannot_escape_output(tmp_path):
    root = tmp_path / 'project'
    output = root / 'build/paper'
    output.mkdir(parents=True)
    (root / 'image.png').write_bytes(b'original image')
    paper.copy_local_images(root, output, '<img src="image.png">')
    assert (output / 'image.png').read_bytes() == b'original image'
    with pytest.raises(ValueError, match='inside the paper directory'):
        paper.copy_local_images(root, output, '<img src="../private.png">')


def test_watch_rebuilds_and_keeps_last_good_page_on_error(tmp_path, monkeypatch):
    source = tmp_path / 'README.md'
    source.write_text('initial')
    monkeypatch.setattr(paper, 'watch_signature', lambda root: source.read_text())
    calls = []

    def fake_build(root, output):
        calls.append(source.read_text())
        if source.read_text() == 'bad':
            raise ValueError('test build failure')

    monkeypatch.setattr(paper, 'build', fake_build)
    state, stop = paper.PreviewState(), threading.Event()
    worker = threading.Thread(target=paper.watch, args=(tmp_path, tmp_path / 'out', state, stop, .02))
    worker.start()
    try:
        time.sleep(.05)
        source.write_text('updated')
        deadline = time.monotonic() + 2
        while state.version == 1 and time.monotonic() < deadline:
            time.sleep(.02)
        assert state.version == 2 and calls == ['updated']
        source.write_text('bad')
        deadline = time.monotonic() + 2
        while not state.error and time.monotonic() < deadline:
            time.sleep(.02)
        assert state.version == 2
        assert state.error == 'test build failure'
        source.write_text('fixed')
        deadline = time.monotonic() + 2
        while state.version == 2 and time.monotonic() < deadline:
            time.sleep(.02)
        assert state.version == 3 and not state.error
    finally:
        stop.set()
        worker.join(timeout=2)


def test_preview_status_endpoint_and_no_directory_listing(tmp_path):
    state = paper.PreviewState()
    handler = partial(paper.PreviewHandler, state=state, directory=str(tmp_path))
    try:
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    except PermissionError:
        pytest.skip('Sandbox disallows binding a local socket')
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(f'http://127.0.0.1:{server.server_port}/__paper_status') as response:
            assert json.load(response) == state.payload()
            assert response.headers['Cache-Control'] == 'no-store'
        from urllib.error import HTTPError
        with pytest.raises(HTTPError, match='404'):
            urlopen(f'http://127.0.0.1:{server.server_port}/')
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
