/* The page uses exported coordinates only. No economic model lives here. */
(() => {
  'use strict';
  const reference = 'Reference industry';
  const adoptionNames = ['High adoption hurdle', 'Low adoption hurdle'];

  function plainLabel(text) {
    return text.replace(/\\\$/g, '\uE000').replace(/\$ per million/g, 'USD / million')
      .replace(/\$|\\left|\\right/g, '')
      .replace(/\\kappa_h/g, 'κₕ').replace(/\\alpha/g, 'α')
      .replace(/\\beta/g, 'β').replace(/\\eta/g, 'η').replace(/\\rho/g, 'ρ')
      .replace(/\\star/g, '*').replace(/\\log/g, ' log ')
      .replace(/c_\{\\rm res\}/g, 'cᵣₑₛ').replace(/\^\{W\}/g, 'ᵂ')
      .replace(/\^\{H\}/g, 'ᴴ').replace(/c_0/g, 'c₀')
      .replace(/[{}^]/g, '').replace(/\uE000/g, '$')
      .replace(/\s+/g, ' ').trim();
  }

  function coincide(left, right) {
    return left && right && left.x.length === right.x.length &&
      left.x.every((x, i) => x === right.x[i]) &&
      left.y.every((y, i) => Math.abs(y - right.y[i]) <= 1e-10 * Math.max(1, Math.abs(y)));
  }

  function overlappingAdoption(spec) {
    return adoptionNames.filter(name => spec.panels.every(panel =>
      coincide(panel.lines.find(line => line.name === reference),
        panel.lines.find(line => line.name === name))));
  }

  function axisRange(values, scale) {
    const finite = values.filter(value => Number.isFinite(value) && (scale !== 'log' || value > 0));
    if (!finite.length) return scale === 'log' ? [-1, 1] : [0, 1];
    const transformed = finite.map(value => scale === 'log' ? Math.log10(value) : value);
    const low = Math.min(...transformed), high = Math.max(...transformed);
    const padding = Math.max((high - low) * 0.07, scale === 'log' ? 0.025 : 0.01);
    return [low - padding, high + padding];
  }

  function fitRanges(spec, visible, shared) {
    const samples = spec.panels.map(panel => [
      ...panel.lines.filter(line => visible.has(line.name)).flatMap(line => line.y),
      ...panel.guides.filter(guide => guide.axis === 'y').map(guide => guide.value)
    ]);
    if (shared) {
      const range = axisRange(samples.flat(), spec.panels[0].yscale);
      return spec.panels.map(() => [...range]);
    }
    return spec.panels.map((panel, i) => axisRange(samples[i], panel.yscale));
  }

  function originalRange(limits, scale) {
    return limits.map(value => scale === 'log' ? Math.log10(value) : value);
  }

  // Also export the small data-only surface for Node's built-in test runner.
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {plainLabel, coincide, overlappingAdoption, axisRange, fitRanges, originalRange};
  }
  if (typeof document === 'undefined') return;

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function button(label, onClick, className = '') {
    const node = element('button', className, label);
    node.type = 'button';
    node.addEventListener('click', onClick);
    return node;
  }

  async function mountChart(figure, spec) {
    const overlapping = overlappingAdoption(spec);
    const series = new Map();
    spec.panels.forEach(panel => panel.lines.forEach(line => {
      if (!series.has(line.name)) series.set(line.name, line);
    }));
    const visible = new Set(series.keys());
    const referenceName = series.has(reference) ? reference : series.has('Reference') ? 'Reference' : null;
    let shared = spec.shared_y;
    let scaleSelect = null;
    let ranges = spec.panels.map(panel => originalRange(panel.ylim, panel.yscale));
    let synchronizing = false;
    const charts = [], legendButtons = new Map();
    const controls = figure.querySelector('.chart-controls');
    const legend = figure.querySelector('.chart-legend');
    const panels = figure.querySelector('.chart-panels');
    const help = figure.querySelector('.chart-help');
    const caption = figure.querySelector('figcaption').textContent;
    panels.style.setProperty('--panels', spec.columns || spec.panels.length);
    panels.classList.toggle('center-last-panel', Boolean(spec.center_last));
    figure.setAttribute('aria-label', caption);

    function updateLegend() {
      legendButtons.forEach((node, name) => node.setAttribute('aria-pressed', String(visible.has(name))));
    }

    async function updateVisibility() {
      updateLegend();
      await Promise.all(charts.map(({plot, lines, points}) => Plotly.restyle(plot, {
        visible: [
          ...lines.map(line => visible.has(line.name)),
          ...points.map(point => visible.has(point.name))
        ]
      })));
    }

    async function applyRanges() {
      synchronizing = true;
      try {
        await Promise.all(charts.map(({plot}, i) => Plotly.relayout(plot, {
          'yaxis.autorange': false, 'yaxis.range': ranges[i]
        })));
      } finally { synchronizing = false; }
    }

    if (spec.shared_y) {
      const label = element('label', 'scale-control', 'Y-axis ');
      const select = scaleSelect = element('select');
      select.setAttribute('aria-label', `${caption}: y-axis scaling`);
      [['shared', 'Shared scale'], ['independent', 'Independent scales']].forEach(([value, text]) => {
        const option = element('option', '', text); option.value = value; select.append(option);
      });
      select.addEventListener('change', async () => {
        shared = select.value === 'shared';
        ranges = fitRanges(spec, visible, shared);
        await applyRanges();
      });
      label.append(select); controls.append(label);
    } else if (spec.panels.length > 1) {
      controls.append(element('span', 'scale-control', 'Independent y-axes'));
    }
    controls.append(
      button('Show all', () => { series.forEach((_, name) => visible.add(name)); updateVisibility(); }),
      button(referenceName ? 'Reference only' : 'First series only', () => {
        visible.clear();
        if (referenceName) visible.add(referenceName);
        else visible.add(series.keys().next().value);
        updateVisibility();
      }),
      button(spec.panels.length > 1 ? 'Fit all visible' : 'Fit visible', async () => {
        ranges = fitRanges(spec, visible, shared); await applyRanges();
      }),
      button('Reset view', async () => {
        ranges = shared ? spec.panels.map(panel => originalRange(panel.ylim, panel.yscale))
          : fitRanges(spec, visible, false);
        await applyRanges();
        await Promise.all(charts.map(({plot}, i) => Plotly.relayout(plot, {
          'xaxis.autorange': false,
          'xaxis.range': originalRange(spec.panels[i].xlim, spec.panels[i].xscale)
        })));
      })
    );

    for (const [name, line] of series) {
      const label = plainLabel(name);
      const node = button('', () => {
        if (visible.has(name)) visible.delete(name); else visible.add(name);
        updateVisibility();
      }, 'legend-item');
      node.setAttribute('aria-label', label);
      node.setAttribute('aria-pressed', 'true');
      node.title = `${label}. Click to toggle; double-click to isolate.`;
      if (overlapping.includes(name)) node.title += ' This curve overlaps the reference.';
      const swatch = element('span', `line-swatch ${line.dash}`);
      swatch.style.setProperty('--swatch', line.color);
      swatch.setAttribute('aria-hidden', 'true');
      const markerGlyph = {
        circle: '○', square: '□', 'triangle-up': '△',
        'triangle-down': '▽', 'triangle-left': '◁',
        'triangle-right': '▷', diamond: '◇', x: '×'
      }[line.marker];
      if (markerGlyph) {
        const marker = element('span', 'line-marker', markerGlyph);
        marker.style.color = line.color;
        swatch.append(marker);
      }
      node.append(swatch, element('span', '', label));
      node.addEventListener('dblclick', () => {
        visible.clear(); visible.add(name); updateVisibility();
      });
      legendButtons.set(name, node); legend.append(node);
    }
    if (overlapping.length) legend.append(element('p', 'overlap-note',
      'The adoption-concentration curves overlap the reference here. They have separate toggles; double-click high or low to see that curve on its own.'));

    for (const [index, panel] of spec.panels.entries()) {
      const section = element('section', 'chart-panel');
      const heading = element('div', 'panel-heading');
      const title = panel.title || (spec.panels.length > 1 ? panel.ylabel : panel.xlabel);
      const titleNode = element('span', '', plainLabel(title));
      const plot = element('div', 'plot');
      plot.setAttribute('role', 'img');
      plot.setAttribute('aria-label', `${caption}: ${plainLabel(title)}. Use the line controls above to compare series.`);
      let expanded = false;
      const expand = button('Expand', () => setExpanded(!expanded), 'expand-panel');
      expand.setAttribute('aria-label', `Expand ${plainLabel(title)}`);
      const actions = element('div', 'panel-actions');
      let fit = null;
      if (spec.panels.length > 1) {
        fit = button('Fit visible', async () => {
          // Local fitting unlocks shared axes, but does not move other panels.
          shared = false;
          if (scaleSelect) scaleSelect.value = 'independent';
          ranges[index] = fitRanges(spec, visible, false)[index];
          synchronizing = true;
          try {
            await Plotly.relayout(plot, {'yaxis.autorange': false, 'yaxis.range': ranges[index]});
          } finally { synchronizing = false; }
        }, 'fit-panel');
        fit.disabled = true;
        fit.setAttribute('aria-label', `Fit visible lines in ${plainLabel(title)}`);
        fit.title = 'Fit this panel’s y-axis to its visible lines. Other panels stay unchanged; y-scales become independent.';
        actions.append(fit);
      }
      actions.append(expand);
      function setExpanded(value) {
        expanded = value;
        section.classList.toggle('expanded', value);
        document.body.classList.toggle('panel-open', value);
        expand.textContent = value ? 'Close ×' : 'Expand';
        expand.setAttribute('aria-label', `${value ? 'Close' : 'Expand'} ${plainLabel(title)}`);
        if (value) {
          section.setAttribute('role', 'dialog'); section.setAttribute('aria-modal', 'true');
          section.setAttribute('aria-label', plainLabel(title)); expand.focus();
        } else {
          section.removeAttribute('role'); section.removeAttribute('aria-modal'); expand.focus();
        }
        requestAnimationFrame(() => Plotly.Plots.resize(plot));
      }
      section.addEventListener('keydown', event => {
        if (!expanded) return;
        if (event.key === 'Escape') { event.preventDefault(); setExpanded(false); }
        if (event.key === 'Tab') {
          const focusable = [...section.querySelectorAll('button,a[href],[tabindex="0"]')];
          const first = focusable[0], last = focusable[focusable.length - 1];
          if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
          else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
        }
      });
      heading.append(titleNode, actions); section.append(heading, plot); panels.append(section);
      const lines = [...panel.lines].sort((a, b) => a.order - b.order);
      const traces = lines.map(line => ({
        type: 'scatter', mode: line.marker ? 'lines+markers' : 'lines',
        name: plainLabel(line.name), x: line.x, y: line.y,
        line: {color: line.color, dash: line.dash, width: line.width},
        ...(line.marker ? {marker: {
          symbol: `${line.marker}-open`, size: 7, color: line.color,
          line: {color: line.color, width: 1}, maxdisplayed: 10
        }} : {}),
        hovertemplate: '%{x:.4g}<br>%{y:.5g}<extra>%{fullData.name}</extra>'
      }));
      const points = panel.points || [];
      traces.push(...points.map(point => ({
        type:'scatter', mode:'markers', name:plainLabel(point.name),
        x:[point.x], y:[point.y], showlegend:false, cliponaxis:false,
        marker:{
          symbol:point.marker, size:point.size, color:point.color,
          line:{color:'#fff', width:1}
        },
        hovertemplate:'Maximum token demand in plotted range<br>%{x:.4g}<br>%{y:.5g}<extra>%{fullData.name}</extra>'
      })));
      const shapes = panel.guides.map(guide => guide.axis === 'x' ? {
        type:'line', xref:'x', yref:'paper', x0:guide.value, x1:guide.value, y0:0, y1:1,
        line:{color:'#9ba59a', width:1, dash:'dot'}
      } : {
        type:'line', xref:'paper', yref:'y', x0:0, x1:1, y0:guide.value, y1:guide.value,
        line:{color:'#9ba59a', width:1, dash:'dot'}
      });
      const layout = {
        autosize:true, showlegend:false, paper_bgcolor:'#fff', plot_bgcolor:'#fff',
        font:{family:'system-ui, sans-serif', size:10, color:'#596359'},
        margin:{l:65, r:12, t:16, b:70}, hovermode:'closest', dragmode:'zoom', shapes,
        xaxis:{type:panel.xscale, title:{
          text:plainLabel(panel.xlabel),
          font:{size:10}, standoff:14},
          range:originalRange(panel.xlim, panel.xscale), autorange:false,
          tickmode:'array', tickvals:panel.xticks, ticktext:panel.xticks.map(String),
          gridcolor:'#e9ede6', zeroline:false, automargin:true},
        yaxis:{type:panel.yscale, nticks:6, minorloglabels:'complete', title:{
          text:plainLabel(panel.ylabel).replace(/^(Attention|Work)-limited token demand/, 'Token demand'),
          font:{size:10}, standoff:5},
          range:ranges[index], autorange:false, gridcolor:'#e9ede6', zeroline:false, automargin:true}
      };
      await Plotly.newPlot(plot, traces, layout, {
        responsive:true, displaylogo:false, scrollZoom:false,
        modeBarButtonsToRemove:['select2d','lasso2d','autoScale2d','resetScale2d'],
        toImageButtonOptions:{format:'png', filename:figure.id + '-' + (index + 1), scale:2},
        doubleClick:false
      });
      charts.push({plot, lines, points});
      if (fit) fit.disabled = false;
      plot.on('plotly_relayout', async update => {
        if (synchronizing) return;
        const range = update['yaxis.range'] || (
          update['yaxis.range[0]'] !== undefined
            ? [update['yaxis.range[0]'], update['yaxis.range[1]']] : null);
        if (!range) return;
        ranges[index] = [...range];
        if (shared) { ranges = spec.panels.map(() => [...range]); await applyRanges(); }
      });
    }
    const benchmark = [...series.keys()].some(name => name.startsWith('Constant revenue'));
    const instructions = help.querySelector('.chart-instructions') || help;
    instructions.textContent = 'Click a line label to toggle; double-click to isolate. Drag to zoom. '
      + (spec.panels.length > 1
        ? 'Each panel’s Fit visible changes only its y-axis and switches to independent scales. Fit all visible fits the whole figure.'
        : 'Hiding lines keeps the scale fixed; choose Fit visible to rescale.')
      + (benchmark ? ' Price panel: above the black line means higher revenue than at the baseline price.' : '');
    figure.classList.add('chart-ready');
  }

  async function start() {
    const data = JSON.parse(document.getElementById('chart-data').textContent);
    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (!entry.isIntersecting) return;
        observer.unobserve(entry.target);
        if (typeof Plotly === 'undefined') return;
        mountChart(entry.target, data[entry.target.dataset.chart]).catch(error => {
          const help = entry.target.querySelector('.chart-help');
          const instructions = help.querySelector('.chart-instructions') || help;
          instructions.textContent =
            'Interactive chart unavailable; the original figure is shown. ' + error.message;
          entry.target.querySelector('.chart-panels').replaceChildren();
          entry.target.querySelector('.chart-controls').replaceChildren();
          entry.target.querySelector('.chart-legend').replaceChildren();
        });
      });
    }, {rootMargin:'400px'});
    document.querySelectorAll('[data-chart]').forEach(figure => observer.observe(figure));

    if (location.protocol !== 'http:' || !['127.0.0.1', 'localhost'].includes(location.hostname)) return;
    let version = null;
    async function poll() {
      try {
        const response = await fetch('/__paper_status', {cache:'no-store'});
        if (!response.ok) return;
        const state = await response.json();
        if (version !== null && version !== state.version) location.reload();
        version = state.version;
      } catch (_) { /* Keep the last rendered page if the local preview stops. */ }
    }
    await poll();
    setInterval(poll, 1200);
  }
  start();
})();
