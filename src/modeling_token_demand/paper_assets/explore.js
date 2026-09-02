/* Interactive plot builder: solves the model in the browser (see model.js). */
(() => {
  'use strict';
  if (typeof document === 'undefined') return;
  const M = window.TokenDemandModel;
  const section = document.getElementById('explore');
  if (!section || !M) return;

  const PALETTE = ['#595959', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                   '#8c564b', '#e377c2', '#17becf', '#bcbd22', '#7f7f7f'];
  const WORK_SET = ['Reference industry', 'Low adoption hurdle', 'High adoption hurdle',
                    'Hard execution', 'High capability requirement'];
  const ATTENTION_SET = ['Reference industry', 'Hard execution', 'Low inference returns',
                         'Slow-growing review', 'Nearly proportional review'];
  const PRECISION = {fast: {grid: 11, starts: 2}, accurate: {grid: 17, starts: 4}};
  const MAX_Y = 4;

  const X_OPTIONS = [
    ...M.SCENARIO_PARAMETERS.map(p => ({key: p.key, label: `${p.label}, ${p.symbol}`, scenario: true})),
    ...M.INDUSTRY_PARAMETERS.map(p => ({key: p.key, label: `${p.label}, ${p.symbol}`, scenario: false})),
  ];
  const X_DEFAULTS = {
    m: [0.1, 30, 'log'], eta: [0.25, 10, 'log'], c: [0.1, 4, 'log'], v: [0.1, 2, 'log'],
    lambda: [1, 50, 'log'], nu: [0.5, 3, 'linear'], a: [0.5, 16, 'log'], alpha: [0.1, 0.9, 'linear'],
    h0: [0.001, 0.3, 'log'], h1: [0.005, 0.5, 'log'], beta: [0.05, 0.95, 'linear'],
    b: [25, 400, 'log'], w: [25, 400, 'log'], mu: [20, 120, 'linear'], sigma: [0.5, 32, 'log'],
    W: [1e5, 1e7, 'log'], H: [1e4, 1e6, 'log'],
  };

  function axis(key, min, max, reverse = false, points = 60, scale) {
    return {key, min, max, points, scale: scale || X_DEFAULTS[key][2], reverse};
  }
  function yv(key, index = false) {
    const output = M.OUTPUTS.find(o => o.key === key);
    return {key, index, scale: (index || output.log) && !output.percent ? 'log' : 'linear'};
  }

  const figurePresets = [
    ['Figure 1 · Work-limited: adoption and token demand vs capability',
     {regime: 'work', x: axis('m', 0.1, 30), y: [yv('A'), yv('D', true)], industries: WORK_SET}],
    ['Figure 2 · Work-limited: adoption, demand, effort vs token efficiency',
     {regime: 'work', x: axis('eta', 0.25, 10), y: [yv('A'), yv('D', true), yv('x'), yv('etax')], industries: WORK_SET}],
    ['Figure 3 · Work-limited: adoption, demand, effort, spending vs price',
     {regime: 'work', x: axis('c', 0.1, 4, true), y: [yv('A'), yv('D', true), yv('x'), yv('R', true)], industries: WORK_SET}],
    ['Figure 4 · Work-limited reservation token price vs capability',
     {regime: 'work', x: axis('m', 0.8, 30, false, 40), y: [yv('cres')], industries: WORK_SET}],
    ['Figure 5 · Attention-limited: leverage and demand vs capability',
     {regime: 'attention', x: axis('m', 0.1, 30), y: [yv('ell', true), yv('D', true)], industries: ATTENTION_SET}],
    ['Figure 6 · Attention-limited: leverage and demand vs token efficiency',
     {regime: 'attention', x: axis('eta', 0.25, 10), y: [yv('ell', true), yv('D', true)], industries: ATTENTION_SET}],
    ['Figure 7 · Attention-limited: demand and spending vs price',
     {regime: 'attention', x: axis('c', 0.1, 4, true), y: [yv('D', true), yv('R', true)], industries: ATTENTION_SET}],
    ['Figure 8 · Attention value and reservation price vs capability',
     {regime: 'attention', x: axis('m', 0.8, 30, false, 40), y: [yv('rho', true), yv('cres')], industries: ATTENTION_SET}],
    ['Beyond the paper · Attention-limited demand vs review elasticity β',
     {regime: 'attention', x: axis('beta', 0.05, 0.95, false, 40, 'linear'), y: [yv('D'), yv('rho'), yv('s')],
      industries: ['Reference industry', 'Hard execution']}],
    ['Beyond the paper · Work-limited demand vs fixed review time h₀',
     {regime: 'work', x: axis('h0', 0.001, 0.3, false, 40), y: [yv('A'), yv('D'), yv('s')],
      industries: ['Reference industry', 'High adoption hurdle']}],
  ];

  /* ---------- state ---------- */

  function makeIndustry(name, color) {
    return {name, color, params: M.presetIndustry(name)};
  }
  function defaultState() {
    return {
      regime: 'work', x: axis('m', 0.1, 30), y: [yv('A'), yv('D', true)],
      industries: WORK_SET.map((name, i) => makeIndustry(name, PALETTE[i % PALETTE.length])),
      precision: 'accurate',
    };
  }
  function presetState(preset) {
    return {
      regime: preset.regime, x: Object.assign({}, preset.x), y: preset.y.map(y => Object.assign({}, y)),
      industries: preset.industries.map((name, i) => makeIndustry(name, PALETTE[i % PALETTE.length])),
      precision: state ? state.precision : 'accurate',
    };
  }
  let state = null;

  function encodeState() {
    const compact = {
      r: state.regime, p: state.precision, x: state.x,
      y: state.y, i: state.industries.map(ind => ({n: ind.name, c: ind.color, p: ind.params})),
    };
    const bytes = new TextEncoder().encode(JSON.stringify(compact));
    let binary = '';
    bytes.forEach(byte => { binary += String.fromCharCode(byte); });
    return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
  }
  function decodeState(text) {
    try {
      const padded = text.replace(/-/g, '+').replace(/_/g, '/') + '='.repeat((4 - text.length % 4) % 4);
      const binary = atob(padded);
      const bytes = Uint8Array.from(binary, char => char.charCodeAt(0));
      const compact = JSON.parse(new TextDecoder().decode(bytes));
      const industries = compact.i.map(ind => ({
        name: String(ind.n).slice(0, 60), color: /^#[0-9a-f]{6}$/i.test(ind.c) ? ind.c : PALETTE[0],
        params: Object.assign({}, M.REFERENCE_INDUSTRY, Object.fromEntries(
          M.INDUSTRY_PARAMETERS.map(p => [p.key, Number(ind.p[p.key])]))),
      }));
      const validX = X_OPTIONS.some(o => o.key === compact.x.key);
      const y = compact.y.filter(y => M.OUTPUTS.some(o => o.key === y.key)).slice(0, MAX_Y);
      if (!validX || !y.length || !industries.length) return null;
      return {
        regime: compact.r === 'attention' ? 'attention' : 'work',
        precision: compact.p === 'fast' ? 'fast' : 'accurate',
        x: {key: compact.x.key, min: Number(compact.x.min), max: Number(compact.x.max),
            points: Math.min(200, Math.max(5, Math.round(Number(compact.x.points)) || 60)),
            scale: compact.x.scale === 'linear' ? 'linear' : 'log', reverse: Boolean(compact.x.reverse)},
        y: y.map(item => ({key: item.key, index: Boolean(item.index), scale: item.scale === 'log' ? 'log' : 'linear'})),
        industries,
      };
    } catch (_) { return null; }
  }

  /* ---------- DOM helpers ---------- */

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function button(label, onClick, className = '') {
    const node = el('button', className, label);
    node.type = 'button';
    node.addEventListener('click', onClick);
    return node;
  }
  function select(options, value, onChange, ariaLabel) {
    const node = el('select');
    if (ariaLabel) node.setAttribute('aria-label', ariaLabel);
    options.forEach(([optionValue, text]) => {
      const option = el('option', '', text); option.value = optionValue; node.append(option);
    });
    node.value = value;
    node.addEventListener('change', () => onChange(node.value));
    return node;
  }
  function numberInput(value, onChange, options = {}) {
    const node = el('input');
    node.type = 'number'; node.value = formatNumber(value);
    node.step = options.step || 'any';
    if (options.ariaLabel) node.setAttribute('aria-label', options.ariaLabel);
    node.addEventListener('change', () => onChange(Number(node.value)));
    return node;
  }
  function labelled(text, control) {
    const label = el('label'); label.append(text, control); return label;
  }
  function formatNumber(value) {
    return Number.isFinite(value) ? Number(value.toPrecision(6)).toString() : '';
  }

  /* ---------- build the controls ---------- */

  const toolbar = section.querySelector('.explore-toolbar');
  const configHost = section.querySelector('.explore-config');
  const industryHost = section.querySelector('.explore-industries');
  const legendHost = section.querySelector('.chart-legend');
  const panelHost = section.querySelector('.chart-panels');
  const statusNode = section.querySelector('.explore-status');
  const warningNode = section.querySelector('.explore-warnings');
  const hidden = new Set();

  function renderToolbar() {
    toolbar.replaceChildren();
    const presetSelect = select(
      [['', 'Load a figure from the paper…'], ...figurePresets.map(([label], i) => [String(i), label])],
      '', value => {
        if (value === '') return;
        state = presetState(figurePresets[Number(value)][1]);
        hidden.clear();
        renderAll(); schedule();
      }, 'Load a preset figure');
    toolbar.append(presetSelect);
    toolbar.append(button('Copy link to this plot', async () => {
      const hash = `#explore=${encodeState()}`;
      history.replaceState(null, '', hash);
      try {
        await navigator.clipboard.writeText(`${location.origin}${location.pathname}${hash}`);
        setStatus('Link copied. It encodes every setting on this page.');
      } catch (_) { setStatus('Link is in the address bar; copy it from there.'); }
    }));
    toolbar.append(button('Download CSV', downloadCsv));
    toolbar.append(button('Reset', () => { state = defaultState(); hidden.clear(); renderAll(); schedule(); }));
  }

  function renderConfig() {
    configHost.replaceChildren();
    const regime = el('fieldset', 'explore-fieldset');
    regime.append(el('legend', '', 'Regime'));
    [['work', 'Work is limited (Section 2)'], ['attention', 'Human attention is limited (Section 3)']].forEach(([value, text]) => {
      const label = el('label', 'explore-radio');
      const radio = el('input'); radio.type = 'radio'; radio.name = 'explore-regime'; radio.value = value;
      radio.checked = state.regime === value;
      radio.addEventListener('change', () => { state.regime = value; renderConfig(); schedule(); });
      label.append(radio, text); regime.append(label);
    });
    regime.append(el('p', 'explore-note', state.regime === 'work'
      ? 'Objective: surplus per work unit u(s,x). Demand D = W·A(u*)·x*.'
      : 'Objective: surplus per review hour J(s,x). Demand D = H·[s*/h(s*)]·x*.'));

    const xAxis = el('fieldset', 'explore-fieldset');
    xAxis.append(el('legend', '', 'X axis (swept variable)'));
    const row = el('div', 'explore-row');
    row.append(labelled('Variable ', select(X_OPTIONS.map(o => [o.key, o.label]), state.x.key, value => {
      const [min, max, scale] = X_DEFAULTS[value];
      state.x = {key: value, min, max, points: state.x.points, scale, reverse: value === 'c'};
      renderConfig(); renderIndustries(); schedule();
    }, 'Swept variable')));
    row.append(labelled('Min ', numberInput(state.x.min, value => { state.x.min = value; schedule(); }, {ariaLabel: 'Minimum'})));
    row.append(labelled('Max ', numberInput(state.x.max, value => { state.x.max = value; schedule(); }, {ariaLabel: 'Maximum'})));
    row.append(labelled('Points ', numberInput(state.x.points, value => {
      state.x.points = Math.min(200, Math.max(5, Math.round(value) || 60)); schedule();
    }, {step: '1', ariaLabel: 'Number of points'})));
    row.append(labelled('Scale ', select([['log', 'log'], ['linear', 'linear']], state.x.scale, value => { state.x.scale = value; schedule(); }, 'X scale')));
    const reverse = el('label', 'explore-check');
    const reverseBox = el('input'); reverseBox.type = 'checkbox'; reverseBox.checked = state.x.reverse;
    reverseBox.addEventListener('change', () => { state.x.reverse = reverseBox.checked; schedule(); });
    reverse.append(reverseBox, 'Reverse axis');
    row.append(reverse);
    xAxis.append(row);
    xAxis.append(el('p', 'explore-note', X_OPTIONS.find(o => o.key === state.x.key).scenario
      ? 'Baseline: 1 (dotted vertical line). Indexed outputs equal 1 there. Other scenario variables stay at 1.'
      : 'Baseline for indexing: each industry’s own configured value of this parameter. Scenario variables m, η, c stay at 1.'));

    const yAxes = el('fieldset', 'explore-fieldset');
    yAxes.append(el('legend', '', `Y axes (one panel each, up to ${MAX_Y})`));
    state.y.forEach((item, index) => {
      const yRow = el('div', 'explore-row');
      yRow.append(select(M.OUTPUTS.map(o => [o.key, o.label]), item.key, value => {
        const fresh = yv(value, item.index); item.key = fresh.key; item.scale = fresh.scale; renderConfig(); schedule();
      }, `Panel ${index + 1} variable`));
      const indexLabel = el('label', 'explore-check');
      const indexBox = el('input'); indexBox.type = 'checkbox'; indexBox.checked = item.index;
      indexBox.addEventListener('change', () => { item.index = indexBox.checked; schedule(); });
      indexLabel.append(indexBox, 'Index to baseline');
      yRow.append(indexLabel);
      yRow.append(labelled('Scale ', select([['log', 'log'], ['linear', 'linear']], item.scale, value => { item.scale = value; schedule(); }, `Panel ${index + 1} scale`)));
      if (state.y.length > 1) yRow.append(button('Remove', () => { state.y.splice(index, 1); renderConfig(); schedule(); }, 'explore-small'));
      yAxes.append(yRow);
    });
    if (state.y.length < MAX_Y) yAxes.append(button('+ Add panel', () => { state.y.push(yv('x')); renderConfig(); schedule(); }, 'explore-small'));
    if (state.y.some(item => M.OUTPUTS.find(o => o.key === item.key).reservation)) {
      yAxes.append(el('p', 'explore-note', 'Reservation price: the token price at which the optimized objective equals its value at the baseline scenario (m = η = c = 1) for that industry; (s, x) re-optimize at every candidate price. Undefined when price itself is swept. Slower to compute.'));
    } else {
      yAxes.append(el('p', 'explore-note', 'Percent outputs (adoption, probabilities) are always shown in levels; every other output can be indexed to 1 at the baseline.'));
    }

    const precision = el('fieldset', 'explore-fieldset');
    precision.append(el('legend', '', 'Solver'));
    precision.append(labelled('Precision ', select([['accurate', 'Accurate (paper settings: 17×17 grid, 4 refinements)'], ['fast', 'Fast (11×11 grid, 2 refinements)']],
      state.precision, value => { state.precision = value; schedule(); }, 'Solver precision')));
    precision.append(el('p', 'explore-note', 'Search bounds: s ∈ [0.002, 800], x ∈ [1, 2000]; x = 1 is the economic effort floor. A warning appears if an optimum touches a numerical bound.'));
    configHost.append(regime, xAxis, yAxes, precision);
  }

  function renderIndustries() {
    industryHost.replaceChildren();
    const heading = el('div', 'explore-industries-head');
    heading.append(el('span', 'explore-legend-label', `Industries (${state.industries.length} ${state.industries.length === 1 ? 'line' : 'lines'})`));
    heading.append(button('+ Add industry', () => {
      const industry = makeIndustry('Reference industry', PALETTE[state.industries.length % PALETTE.length]);
      industry.name = `Industry ${state.industries.length + 1}`;
      state.industries.push(industry);
      renderIndustries(); schedule();
    }, 'explore-small'));
    industryHost.append(heading);
    industryHost.append(el('p', 'explore-note', 'Values that differ from the reference industry are highlighted, as in the paper’s tables. Hover a symbol for its definition.'));
    const list = el('div', 'explore-industry-list');
    state.industries.forEach((industry, index) => list.append(industryCard(industry, index)));
    industryHost.append(list);
  }

  function industryCard(industry, index) {
    const card = el('div', 'explore-industry');
    const head = el('div', 'explore-industry-head');
    const color = el('input'); color.type = 'color'; color.value = industry.color;
    color.setAttribute('aria-label', `${industry.name} line color`);
    color.addEventListener('input', () => { industry.color = color.value; schedule(); });
    const name = el('input'); name.type = 'text'; name.value = industry.name; name.className = 'explore-name';
    name.setAttribute('aria-label', 'Industry name');
    name.addEventListener('change', () => { industry.name = name.value.trim() || `Industry ${index + 1}`; schedule(); });
    const preset = select([['', 'Set parameters from…'], ...M.INDUSTRY_PRESETS.map(p => [p.name, p.name])], '', value => {
      if (!value) return;
      industry.params = M.presetIndustry(value);
      if (/^Industry \d+$/.test(industry.name) || M.INDUSTRY_PRESETS.some(p => p.name === industry.name)) industry.name = value;
      renderIndustries(); schedule();
    }, 'Load preset parameters');
    head.append(color, name, preset);
    head.append(button('Duplicate', () => {
      const copy = {name: `${industry.name} (copy)`, color: PALETTE[state.industries.length % PALETTE.length], params: Object.assign({}, industry.params)};
      state.industries.splice(index + 1, 0, copy); renderIndustries(); schedule();
    }, 'explore-small'));
    if (state.industries.length > 1) head.append(button('Remove', () => { state.industries.splice(index, 1); renderIndustries(); schedule(); }, 'explore-small'));
    card.append(head);
    const grid = el('div', 'explore-params');
    for (const spec of M.INDUSTRY_PARAMETERS) {
      const wrapper = el('label', 'explore-param');
      wrapper.title = `${spec.label}: ${spec.help}`;
      const input = numberInput(industry.params[spec.key], value => {
        industry.params[spec.key] = value;
        wrapper.classList.toggle('changed', value !== M.REFERENCE_INDUSTRY[spec.key]);
        schedule();
      }, {ariaLabel: `${industry.name}: ${spec.label}`});
      wrapper.classList.toggle('changed', industry.params[spec.key] !== M.REFERENCE_INDUSTRY[spec.key]);
      wrapper.classList.toggle('swept', spec.key === state.x.key);
      wrapper.append(el('span', 'explore-symbol', spec.symbol), input);
      grid.append(wrapper);
    }
    card.append(grid);
    return card;
  }

  function renderAll() { renderToolbar(); renderConfig(); renderIndustries(); }

  /* ---------- computation ---------- */

  let timer = null, run = 0, latest = null;
  function schedule() {
    clearTimeout(timer);
    timer = setTimeout(compute, 200);
  }
  function setStatus(text, busy = false) {
    statusNode.textContent = text;
    statusNode.classList.toggle('busy', busy);
  }

  function validate() {
    const problems = [];
    const x = state.x;
    const xSpec = [...M.SCENARIO_PARAMETERS, ...M.INDUSTRY_PARAMETERS].find(p => p.key === x.key);
    if (!(Number.isFinite(x.min) && Number.isFinite(x.max)) || x.min >= x.max) problems.push('X range needs min < max.');
    else {
      if (x.min < xSpec.min) problems.push(`${xSpec.symbol} must be at least ${xSpec.min}.`);
      if (xSpec.max !== undefined && x.max > xSpec.max) problems.push(`${xSpec.symbol} must stay below one.`);
      if (x.scale === 'log' && x.min <= 0) problems.push('A log x-axis needs a positive minimum.');
    }
    state.industries.forEach(industry => {
      M.validateIndustry(industry.params).forEach(problem => problems.push(`${industry.name}: ${problem}`));
    });
    return problems;
  }

  async function compute() {
    const problems = validate();
    if (problems.length) { setStatus(problems.join(' ')); return; }
    const id = ++run;
    const started = performance.now();
    setStatus('Computing…', true);
    const settings = PRECISION[state.precision];
    const outputKeys = [...new Set(state.y.map(y => y.key))];
    const isScenario = X_OPTIONS.find(o => o.key === state.x.key).scenario;
    const results = [], warnings = new Set();
    for (const industry of state.industries) {
      const anchor = isScenario ? 1 : industry.params[state.x.key];
      const values = M.axisValues(state.x.min, state.x.max, state.x.points, state.x.scale, anchor);
      // Yield between industries so the page stays responsive on long sweeps.
      await new Promise(resolve => setTimeout(resolve, 0));
      if (id !== run) return;
      let result;
      try {
        result = M.sweep(industry.params, {key: state.x.key, values}, state.regime, outputKeys, settings);
      } catch (error) { warnings.add(`${industry.name}: ${error.message}`); continue; }
      result.warnings.forEach(w => warnings.add(`${industry.name}: ${w}`));
      results.push({industry: {name: industry.name, color: industry.color}, anchor, values, series: result.series});
    }
    if (id !== run) return;
    latest = {results, outputKeys, regime: state.regime, x: Object.assign({}, state.x), y: state.y.map(y => Object.assign({}, y))};
    warningNode.textContent = [...warnings].join(' ');
    await draw();
    const seconds = ((performance.now() - started) / 1000).toFixed(1);
    setStatus(`${results.reduce((n, r) => n + r.values.length, 0)} optimizations solved in ${seconds} s.`);
  }

  function anchorIndex(values, anchor) {
    let best = 0;
    values.forEach((v, i) => { if (Math.abs(v - anchor) < Math.abs(values[best] - anchor)) best = i; });
    return best;
  }

  /* ---------- drawing ---------- */

  function xLabel() {
    const option = X_OPTIONS.find(o => o.key === latest.x.key);
    return option.label + (latest.x.key === 'c' && latest.x.reverse ? ' (cheaper to the right)' : '');
  }
  function yLabel(item) {
    const output = M.OUTPUTS.find(o => o.key === item.key);
    if (output.percent) return `${output.label} (%)`;
    if (item.index) return `${output.label} index (baseline = 1)`;
    return output.unit ? `${output.label} (${output.unit})` : output.label;
  }
  function seriesFor(result, item) {
    const output = M.OUTPUTS.find(o => o.key === item.key);
    let y = result.series[item.key].slice();
    if (output.percent) y = y.map(v => 100 * v);
    else if (item.index) {
      const base = y[anchorIndex(result.values, result.anchor)];
      y = y.map(v => v / base);
    }
    return y;
  }
  function dashFor(industry) {
    return industry.name === 'Reference industry' ? 'solid' : 'dash';
  }
  /** Fully labelled ticks for a log axis: 1-2-5 per decade, denser on narrow ranges, thinned when crowded. */
  function logTicks(lo, hi) {
    const first = Math.floor(Math.log10(lo)), last = Math.ceil(Math.log10(hi));
    const collect = mantissas => {
      const values = [];
      for (let k = first; k <= last; k++) for (const mantissa of mantissas) {
        const value = Number((mantissa * Math.pow(10, k)).toPrecision(3));
        if (value >= lo * 0.999 && value <= hi * 1.001) values.push(value);
      }
      return values;
    };
    let ticks = collect([1, 2, 5]);
    if (ticks.length > 9) ticks = collect([1]);
    else if (ticks.length < 4) ticks = collect([1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9]);
    if (ticks.length < 4) ticks = collect([1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9]);
    return ticks;
  }
  function tickLabel(value) {
    const magnitude = Math.abs(value);
    if (magnitude >= 1e12) return `${Number((value / 1e12).toPrecision(3))}T`;
    if (magnitude >= 1e9) return `${Number((value / 1e9).toPrecision(3))}B`;
    if (magnitude >= 1e6) return `${Number((value / 1e6).toPrecision(3))}M`;
    if (magnitude >= 1e4) return `${Number((value / 1e3).toPrecision(3))}k`;
    if (magnitude < 1e-3 && value !== 0) return value.toExponential(1).replace('e-', 'e−');
    return String(Number(value.toPrecision(3)));
  }

  async function draw() {
    if (!latest) return;
    const {results, y: items} = latest;
    legendHost.replaceChildren();
    results.forEach(result => {
      const name = result.industry.name;
      const node = button('', () => { if (hidden.has(name)) hidden.delete(name); else hidden.add(name); draw(); }, 'legend-item');
      node.setAttribute('aria-pressed', String(!hidden.has(name)));
      node.title = `${name}. Click to toggle; double-click to isolate.`;
      const swatch = el('span', `line-swatch ${dashFor(result.industry) === 'solid' ? '' : 'dash'}`);
      swatch.style.setProperty('--swatch', result.industry.color);
      node.append(swatch, el('span', '', name));
      node.addEventListener('dblclick', () => {
        hidden.clear(); results.forEach(r => { if (r.industry.name !== name) hidden.add(r.industry.name); }); draw();
      });
      legendHost.append(node);
    });
    panelHost.style.setProperty('--panels', Math.min(2, items.length));
    while (panelHost.children.length > items.length) panelHost.lastChild.remove();
    while (panelHost.children.length < items.length) {
      const panel = el('section', 'chart-panel');
      panel.append(el('div', 'panel-heading'), el('div', 'plot'));
      panelHost.append(panel);
    }
    const anchors = new Set(results.map(r => r.anchor));
    const lowest = Math.min(latest.x.min, latest.x.max), highest = Math.max(latest.x.min, latest.x.max);
    const xRange = latest.x.scale === 'log' ? [Math.log10(lowest), Math.log10(highest)] : [lowest, highest];
    const drawings = items.map((item, index) => {
      const output = M.OUTPUTS.find(o => o.key === item.key);
      const panel = panelHost.children[index];
      panel.querySelector('.panel-heading').textContent = `(${String.fromCharCode(97 + index)}) ${output.label}`;
      const traces = results.map(result => ({
        type: 'scatter', mode: 'lines', name: result.industry.name,
        x: result.values, y: seriesFor(result, item),
        visible: hidden.has(result.industry.name) ? 'legendonly' : true,
        line: {color: result.industry.color, dash: dashFor(result.industry), width: dashFor(result.industry) === 'solid' ? 2.6 : 2.2},
        hovertemplate: '%{x:.4g}<br>%{y:.5g}<extra>%{fullData.name}</extra>',
      }));
      const shapes = [];
      if (anchors.size === 1) {
        const anchor = [...anchors][0];
        if (anchor > lowest && anchor < highest) {
          shapes.push({type: 'line', xref: 'x', yref: 'paper', x0: anchor, x1: anchor, y0: 0, y1: 1, line: {color: '#9ba59a', width: 1, dash: 'dot'}});
        }
      }
      if (item.index && !output.percent) shapes.push({type: 'line', xref: 'paper', yref: 'y', x0: 0, x1: 1, y0: 1, y1: 1, line: {color: '#9ba59a', width: 1, dash: 'dot'}});
      // Constant-revenue benchmarks for price sweeps, as in Figures 3 and 7.
      if (latest.x.key === 'c' && item.index && results[0] && (item.key === 'D' || item.key === 'R')) {
        const values = results[0].values;
        traces.push({type: 'scatter', mode: 'lines', name: 'Constant revenue', x: values,
          y: values.map(v => item.key === 'D' ? 1 / v : 1), line: {color: '#000', width: 1.4, dash: 'dot'},
          hovertemplate: 'Constant revenue<extra></extra>'});
      }
      const layout = {
        autosize: true, showlegend: false, paper_bgcolor: '#fff', plot_bgcolor: '#fff',
        font: {family: 'system-ui, sans-serif', size: 10, color: '#596359'},
        margin: {l: 65, r: 12, t: 16, b: 70}, hovermode: 'closest', dragmode: 'zoom', shapes,
        xaxis: Object.assign({
          type: latest.x.scale, title: {text: xLabel(), font: {size: 10}, standoff: 14},
          range: latest.x.reverse ? [xRange[1], xRange[0]] : xRange, autorange: false,
          gridcolor: '#e9ede6', zeroline: false, automargin: true,
        }, latest.x.scale === 'log' ? (() => {
          const ticks = logTicks(lowest, highest);
          return {tickmode: 'array', tickvals: ticks, ticktext: ticks.map(tickLabel)};
        })() : {}),
        yaxis: Object.assign({
          type: item.scale, title: {text: yLabel(item), font: {size: 10}, standoff: 5},
          autorange: true, gridcolor: '#e9ede6', zeroline: false, automargin: true, nticks: 6,
        }, item.scale === 'log' ? (() => {
          const positive = traces.flatMap(trace => trace.visible === true ? trace.y : []).filter(v => Number.isFinite(v) && v > 0);
          if (!positive.length) return {};
          const ticks = logTicks(Math.min(...positive), Math.max(...positive));
          return ticks.length >= 2 ? {tickmode: 'array', tickvals: ticks, ticktext: ticks.map(tickLabel)} : {};
        })() : {},
           output.percent ? {range: [0, 102], autorange: false} : {}),
      };
      return Plotly.react(panel.querySelector('.plot'), traces, layout, {
        responsive: true, displaylogo: false, scrollZoom: false,
        modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d'],
        toImageButtonOptions: {format: 'png', filename: `explore-${item.key}`, scale: 2},
      });
    });
    await Promise.all(drawings);
    section.classList.add('explore-ready');
  }

  function downloadCsv() {
    if (!latest) { setStatus('Nothing computed yet.'); return; }
    const header = ['industry', latest.x.key, ...latest.outputKeys];
    const rows = [header.join(',')];
    latest.results.forEach(result => {
      result.values.forEach((value, i) => {
        rows.push([JSON.stringify(result.industry.name), value, ...latest.outputKeys.map(key => result.series[key][i])].join(','));
      });
    });
    const link = document.createElement('a');
    link.href = 'data:text/csv;charset=utf-8,' + encodeURIComponent(rows.join('\n'));
    link.download = `token-demand-${latest.regime}-${latest.x.key}.csv`;
    document.body.append(link); link.click(); link.remove();
  }

  /* ---------- start ---------- */

  const fromHash = location.hash.startsWith('#explore=') ? decodeState(location.hash.slice('#explore='.length)) : null;
  state = fromHash || defaultState();
  renderAll();
  if (fromHash) requestAnimationFrame(() => section.scrollIntoView({block: 'start'}));
  function begin() {
    if (typeof Plotly === 'undefined') { setStatus('Plotting library unavailable; charts cannot be drawn.'); return; }
    schedule();
  }
  // Compute lazily once the section is near the viewport, with a fallback so a
  // background tab (where IntersectionObserver stays silent) still gets a plot.
  let started = false;
  const start = () => { if (started) return; started = true; begin(); };
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(entries => {
      if (!entries.some(entry => entry.isIntersecting)) return;
      observer.disconnect(); start();
    }, {rootMargin: '600px'});
    observer.observe(section);
    setTimeout(start, 4000);
  } else start();
})();
