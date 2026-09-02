const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {overlappingAdoption, axisRange, fitRanges, originalRange, plainLabel} =
  require('../src/modeling_token_demand/paper_assets/paper.js');
const plots = JSON.parse(fs.readFileSync(path.join(__dirname, '../figures/interactive.json'))).plots;

test('only identical adoption curves receive an overlap note', () => {
  assert.deepEqual(overlappingAdoption(plots['attention-limited-token-demand-indexed.png']),
    ['High adoption hurdle', 'Low adoption hurdle']);
  assert.deepEqual(overlappingAdoption(plots['work-limited-token-demand-indexed.png']), []);
});

test('shared scales agree; independent scales reveal the efficiency response', () => {
  const spec = plots['attention-limited-token-demand-indexed.png'];
  const visible = new Set(spec.panels[0].lines.map(line => line.name));
  const shared = fitRanges(spec, visible, true);
  assert.deepEqual(shared[0], shared[1]);
  assert.deepEqual(shared[1], shared[2]);
  const independent = fitRanges(spec, visible, false);
  assert.ok(independent[1][1] - independent[1][0] < independent[2][1] - independent[2][0]);
  assert.ok(independent[1][0] < 0 && independent[1][1] > 0); // y=1 retained
});

test('fit visible uses only selected lines, and empty visibility is valid', () => {
  const spec = plots['attention-limited-token-demand-levels.png'];
  const all = new Set(spec.panels[0].lines.map(line => line.name));
  const full = fitRanges(spec, all, true)[0];
  const reference = fitRanges(spec, new Set(['Reference industry']), true)[0];
  assert.ok(reference[1] - reference[0] < full[1] - full[0]);
  assert.deepEqual(fitRanges(spec, new Set(), false), [[-1, 1], [-1, 1], [-1, 1]]);
});

test('every displayed token-price range runs from 4 down to 0.1', () => {
  const pricePanels = Object.values(plots).flatMap(spec => spec.panels)
    .filter(panel => panel.xlabel.toLowerCase().includes('token price'));
  assert.equal(pricePanels.length, 15);
  for (const panel of pricePanels) {
    assert.deepEqual(panel.xlim, [4, .1]);
    assert.deepEqual(panel.xticks, [.1, .2, .5, 1, 2, 4]);
    const range = originalRange(panel.xlim, panel.xscale);
    assert.ok(range[0] > range[1]);
  }
});

test('linear elasticity axes are not converted into log scales', () => {
  assert.deepEqual(originalRange([0, 1.02], 'linear'), [0, 1.02]);
  const range = axisRange([0, .5, 1], 'linear');
  assert.ok(range[0] < 0 && range[1] > 1);
  assert.equal(plainLabel('Token efficiency, $\\eta$'), 'Token efficiency, η');
  assert.equal(plainLabel('Review elasticity, $\\beta$'), 'Review elasticity, β');
  assert.equal(plainLabel('Review-cost multiplier, $\\kappa_h$'), 'Review-cost multiplier, κₕ');
  assert.equal(plainLabel('Marginal inference returns, $\\alpha$'), 'Marginal inference returns, α');
  assert.equal(plainLabel('Normalized token price, $c$'), 'Normalized token price, c');
  assert.equal(
    plainLabel('Reservation token price, $c_{\\rm res}^{W}(m)/c_0$'),
    'Reservation token price, cᵣₑₛᵂ(m)/c₀'
  );
  assert.equal(
    plainLabel('Reservation token price, $c_{\\rm res}^{H}(m)/c_0$'),
    'Reservation token price, cᵣₑₛᴴ(m)/c₀'
  );
  assert.equal(
    plainLabel('Hourly attention price, $\\rho^*(m)$ (\\$/review hour)'),
    'Hourly attention price, ρ*(m) ($/review hour)'
  );
});

// A deliberately small DOM/Plotly test double exercises the actual event
// handlers without adding a browser or Node dependency to this Python project.
async function mountedFigure(spec = plots['attention-limited-token-demand-indexed.png']) {
  const vm = require('node:vm');
  let document;
  class Element {
    constructor(tag) {
      this.tagName = tag; this.children = []; this.attributes = {}; this.handlers = {};
      this.className = ''; this.dataset = {}; this._text = '';
      this.style = {setProperty(name, value){ this[name] = value; }};
      this.classList = {
        contains: value => this.className.split(' ').includes(value),
        add: value => { if (!this.classList.contains(value)) this.className += ' ' + value; },
        remove: value => { this.className = this.className.split(' ').filter(x => x !== value).join(' '); },
        toggle: (value, enabled) => enabled ? this.classList.add(value) : this.classList.remove(value)
      };
    }
    set textContent(value) { this._text = value; this.children = []; }
    get textContent() { return this._text + this.children.map(child => child.textContent).join(''); }
    append(...nodes) { this.children.push(...nodes); }
    replaceChildren(...nodes) { this.children = nodes; }
    setAttribute(name, value) { this.attributes[name] = value; }
    getAttribute(name) { return this.attributes[name]; }
    removeAttribute(name) { delete this.attributes[name]; }
    addEventListener(name, handler) { (this.handlers[name] ||= []).push(handler); }
    async dispatch(name, payload = {}) { for (const handler of this.handlers[name] || []) await handler(payload); }
    focus() { document.activeElement = this; }
    querySelectorAll(selector) {
      const descendants = this.children.flatMap(child => [child, ...child.querySelectorAll('*')]);
      if (selector === '*') return descendants;
      if (selector.startsWith('.')) return descendants.filter(child => child.classList.contains(selector.slice(1)));
      return descendants.filter(child => child.tagName === selector);
    }
    querySelector(selector) { return this.querySelectorAll(selector)[0]; }
  }
  const figure = new Element('figure'); figure.dataset.chart = 'test'; figure.id = 'test';
  const caption = new Element('figcaption'); caption.textContent = 'Attention demand'; figure.append(caption);
  for (const name of ['chart-controls', 'chart-legend', 'chart-panels', 'chart-help']) {
    const child = new Element('div'); child.className = name; figure.append(child);
  }
  const data = new Element('script');
  data.textContent = JSON.stringify({test: spec});
  document = {
    body: new Element('body'), activeElement:null,
    createElement: tag => new Element(tag),
    getElementById: id => id === 'chart-data' ? data : new Element('span'),
    querySelectorAll: () => [figure]
  };
  const rendered = [];
  const Plotly = {
    async newPlot(plot, traces, layout) {
      plot.data = traces; plot.layout = layout;
      plot.on = (name, handler) => plot.addEventListener(name, handler);
      rendered.push(plot);
    },
    async restyle(plot, update) { plot.data.forEach((line, i) => { line.visible = update.visible[i]; }); },
    async relayout(plot, update) {
      for (const [key, value] of Object.entries(update)) {
        const [axis, attribute] = key.split('.'); plot.layout[axis][attribute] = value;
      }
    },
    Plots:{resize(){}}
  };
  const context = {
    document, Plotly, location:{protocol:'file:'}, requestAnimationFrame: callback => callback(),
    IntersectionObserver: class {
      constructor(callback) { this.callback = callback; }
      observe(target) { this.callback([{target, isIntersecting:true}]); }
      unobserve() {}
    }
  };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../src/modeling_token_demand/paper_assets/paper.js'), 'utf8'), context);
  for (let i = 0; i < 20 && !figure.classList.contains('chart-ready'); i++) {
    await new Promise(resolve => setImmediate(resolve));
  }
  assert.ok(figure.classList.contains('chart-ready'), figure.querySelector('.chart-help').textContent);
  return {figure, rendered, document};
}

test('actual controls mount, link visibility, isolate and restore traces', async () => {
  const {figure, rendered} = await mountedFigure();
  assert.equal(rendered.length, 3);
  assert.equal(rendered[0].data.length, 8); // each one-parameter case is selectable
  assert.equal(rendered[2].data.length, 9); // includes revenue benchmark
  const legend = figure.querySelectorAll('.legend-item');
  const execution = legend.find(node => node.getAttribute('aria-label') === 'Hard execution');
  const initialRanges = JSON.stringify(rendered.map(plot => plot.layout.yaxis.range));
  await execution.dispatch('click');
  assert.equal(execution.getAttribute('aria-pressed'), 'false');
  assert.ok(rendered.every(plot => plot.data.find(line => line.name === 'Hard execution').visible === false));
  assert.equal(JSON.stringify(rendered.map(plot => plot.layout.yaxis.range)), initialRanges);
  await execution.dispatch('dblclick');
  assert.ok(rendered.every(plot => plot.data.filter(line => line.visible).length === 1));
  await figure.querySelectorAll('button').find(node => node.textContent === 'Show all').dispatch('click');
  assert.ok(rendered.every(plot => plot.data.every(line => line.visible)));
});

test('paper figures preserve solid reference, marked alternatives, and dotted benchmark', async () => {
  const {figure, rendered} = await mountedFigure(plots['work-price-demand-spending.png']);
  const demand = rendered[1].data;
  const reference = demand.find(line => line.name === 'Reference industry');
  assert.equal(reference.mode, 'lines');
  assert.equal(reference.line.color, '#595959');
  assert.equal(reference.line.dash, 'solid');

  const alternatives = demand.filter(line =>
    !['Reference industry', 'Constant revenue'].includes(line.name));
  assert.ok(alternatives.every(line => line.mode === 'lines+markers'));
  assert.ok(alternatives.every(line => line.line.dash === 'dash'));
  assert.equal(new Set(alternatives.map(line => line.marker.symbol)).size, alternatives.length);

  const benchmark = demand.find(line => line.name === 'Constant revenue');
  assert.equal(benchmark.mode, 'lines');
  assert.equal(benchmark.line.color, '#000000');
  assert.equal(benchmark.line.dash, 'dot');
  assert.ok(benchmark.line.width < reference.line.width);

  const markerLabels = figure.querySelectorAll('.legend-item')
    .filter(node => node.querySelector('.line-marker'));
  assert.equal(markerLabels.length, alternatives.length);
});

test('figure one demand-peak stars match and follow their adoption curves', async () => {
  const {figure, rendered} = await mountedFigure(plots['work-capability-demand-spending.png']);
  const stars = rendered[0].data.filter(trace => trace.marker?.symbol === 'star');
  assert.equal(stars.length, 5);
  assert.equal(rendered[1].data.filter(trace => trace.marker?.symbol === 'star').length, 0);
  for (const star of stars) {
    const line = rendered[0].data.find(trace => trace.name === star.name && trace !== star);
    assert.equal(star.marker.color, line.line.color);
    assert.equal(star.x.length, 1);
    assert.equal(star.y.length, 1);
  }

  const reference = figure.querySelectorAll('.legend-item')
    .find(node => node.getAttribute('aria-label') === 'Reference industry');
  await reference.dispatch('click');
  assert.ok(rendered[0].data.filter(trace => trace.name === 'Reference industry')
    .every(trace => trace.visible === false));
});

test('figure two square exposes physical and effective inference mechanisms', async () => {
  const {figure, rendered} = await mountedFigure(plots['work-efficiency-demand-spending.png']);
  assert.equal(rendered.length, 4);
  assert.equal(figure.querySelector('.chart-panels').style['--panels'], 2);

  const neutral = rendered[1].data.find(trace => trace.name === 'Revenue neutral (price fixed)');
  assert.ok(neutral.y.every(value => value === 1));
  const physical = rendered[2].data.find(trace => trace.name === 'Hard execution');
  const effective = rendered[3].data.find(trace => trace.name === 'Hard execution');
  assert.ok(physical.y[physical.y.length - 1] < physical.y[0]);
  assert.ok(effective.y[effective.y.length - 1] > effective.y[0]);

  const hard = figure.querySelectorAll('.legend-item')
    .find(node => node.getAttribute('aria-label') === 'Hard execution');
  await hard.dispatch('click');
  assert.ok(rendered.every(plot => plot.data
    .filter(trace => trace.name === 'Hard execution')
    .every(trace => trace.visible === false)));
});

test('figure three is square and shows adoption, effort, and spending', async () => {
  const spec = plots['work-price-demand-spending.png'];
  const {figure, rendered} = await mountedFigure(spec);
  assert.equal(rendered.length, 4);
  const panels = figure.querySelector('.chart-panels');
  assert.equal(panels.style['--panels'], 2);
  assert.ok(!panels.classList.contains('center-last-panel'));

  const adoption = rendered[0].data.find(trace => trace.name === 'Reference industry');
  assert.equal(rendered[0].layout.yaxis.type, 'linear');
  assert.ok(adoption.y.every(value => value >= 0 && value <= 100));

  const effort = rendered[2].data.find(trace => trace.name === 'Reference industry');
  assert.ok(effort.x[0] < effort.x[effort.x.length - 1]);
  assert.ok(effort.y[0] > effort.y[effort.y.length - 1]);
  assert.ok(rendered[2].layout.xaxis.range[0] > rendered[2].layout.xaxis.range[1]);

  const spending = rendered[3].data.find(trace => trace.name === 'Reference industry');
  assert.ok(spending.y.every(value => value > 0));
});

test('figure four retains and identifies the coincident hurdle-only curves', async () => {
  const spec = plots['work-capability-reservation-price.png'];
  assert.deepEqual(overlappingAdoption(spec), [
    'High adoption hurdle', 'Low adoption hurdle'
  ]);
  const {figure, rendered} = await mountedFigure(spec);
  assert.match(figure.querySelector('.overlap-note').textContent, /separate toggles/);
  assert.deepEqual(new Set(rendered[0].data.map(trace => trace.name)), new Set([
    'Reference industry', 'Low adoption hurdle', 'High adoption hurdle',
    'Hard execution', 'High capability requirement'
  ]));
});

test('actual scale selector, linked zoom, and expand/escape handlers work', async () => {
  const {figure, rendered, document} = await mountedFigure();
  await rendered[0].dispatch('plotly_relayout', {'yaxis.range[0]':-.3, 'yaxis.range[1]':.4});
  assert.ok(rendered.every(plot => plot.layout.yaxis.range[0] === -.3));
  const select = figure.querySelector('select'); select.value = 'independent';
  await select.dispatch('change');
  assert.notDeepEqual(rendered[0].layout.yaxis.range, rendered[1].layout.yaxis.range);
  const panel = figure.querySelector('.chart-panel');
  const expand = panel.querySelector('.expand-panel');
  await expand.dispatch('click');
  assert.equal(panel.getAttribute('aria-modal'), 'true');
  assert.ok(document.body.classList.contains('panel-open'));
  await panel.dispatch('keydown', {key:'Escape', preventDefault(){}});
  assert.equal(panel.getAttribute('aria-modal'), undefined);
  assert.ok(!document.body.classList.contains('panel-open'));
  assert.equal(document.activeElement, expand);
});

test('each panel fits only itself and unlocks shared scales without moving its neighbors', async () => {
  for (const index of [0, 1, 2]) {
    const {figure, rendered} = await mountedFigure();
    const buttons = figure.querySelectorAll('.fit-panel');
    assert.equal(buttons.length, 3);
    assert.ok(buttons.every(node => node.disabled === false));
    await figure.querySelectorAll('button').find(node => node.textContent === 'Reference only').dispatch('click');
    const before = rendered.map(plot => JSON.stringify(plot.layout.yaxis.range));
    const xBefore = rendered.map(plot => JSON.stringify(plot.layout.xaxis.range));
    await buttons[index].dispatch('click');
    assert.equal(figure.querySelector('select').value, 'independent');
    const expected = fitRanges(plots['attention-limited-token-demand-indexed.png'],
      new Set(['Reference industry']), false)[index];
    assert.equal(JSON.stringify(rendered[index].layout.yaxis.range), JSON.stringify(expected));
    for (const other of [0, 1, 2].filter(i => i !== index)) {
      assert.equal(JSON.stringify(rendered[other].layout.yaxis.range), before[other]);
    }
    assert.deepEqual(rendered.map(plot => JSON.stringify(plot.layout.xaxis.range)), xBefore);
    assert.ok(rendered[2].layout.xaxis.range[0] > rendered[2].layout.xaxis.range[1]);
    await rendered[index].dispatch('plotly_relayout', {'yaxis.range[0]':-.1, 'yaxis.range[1]':.2});
    for (const other of [0, 1, 2].filter(i => i !== index)) {
      assert.equal(JSON.stringify(rendered[other].layout.yaxis.range), before[other]);
    }
    const select = figure.querySelector('select'); select.value = 'shared';
    await select.dispatch('change');
    assert.equal(JSON.stringify(rendered[0].layout.yaxis.range), JSON.stringify(rendered[2].layout.yaxis.range));
  }
});

test('both adoption-hurdle labels can be toggled or isolated in all three panels', async () => {
  const {figure, rendered} = await mountedFigure();
  assert.match(figure.querySelector('.overlap-note').textContent, /separate toggles/);
  for (const name of ['High adoption hurdle', 'Low adoption hurdle']) {
    const node = figure.querySelectorAll('.legend-item').find(item => item.getAttribute('aria-label') === name);
    assert.ok(node);
    await node.dispatch('dblclick');
    assert.ok(rendered.every(plot => plot.data.filter(line => line.visible).length === 1));
    assert.ok(rendered.every(plot => plot.data.find(line => line.name === name).visible));
    await node.dispatch('click');
    assert.ok(rendered.every(plot => plot.data.every(line => !line.visible)));
  }
});

test('paradigm gallery preserves mixed axes and linked case isolation', async () => {
  const {figure, rendered} = await mountedFigure(plots['paradigm-work-demand.png']);
  assert.deepEqual(rendered.map(plot => plot.layout.yaxis.type), ['log', 'log', 'log']);
  assert.equal(figure.querySelector('select'), undefined);
  const adoption = figure.querySelectorAll('.legend-item')
    .find(node => node.getAttribute('aria-label') === 'High adoption hurdle');
  await adoption.dispatch('dblclick');
  assert.ok(rendered.every(plot => plot.data.filter(line => line.visible).length === 1));
  assert.ok(rendered.every(plot => plot.data.find(line => line.name === 'High adoption hurdle').visible));
  const before = rendered.map(plot => JSON.stringify(plot.layout.yaxis.range));
  await figure.querySelectorAll('.fit-panel')[1].dispatch('click');
  assert.equal(JSON.stringify(rendered[0].layout.yaxis.range), before[0]);
  assert.equal(JSON.stringify(rendered[2].layout.yaxis.range), before[2]);
});

test('question figures retain two rows and link case controls across both regimes', async () => {
  const {figure, rendered} = await mountedFigure(plots['intervention-efficiency-returns.png']);
  assert.equal(rendered.length, 6);
  assert.equal(figure.querySelector('.chart-panels').style['--panels'], 3);
  assert.equal(figure.querySelector('select'), undefined);
  const middle = figure.querySelectorAll('.legend-item')
    .find(node => node.getAttribute('aria-label') === 'Inference returns α = 0.5');
  await middle.dispatch('dblclick');
  assert.ok(rendered.every(plot => plot.data.filter(line => line.visible).length === 1));
  const before = rendered.map(plot => JSON.stringify(plot.layout.yaxis.range));
  await figure.querySelectorAll('.fit-panel')[4].dispatch('click');
  for (const other of [0, 1, 2, 3, 5]) {
    assert.equal(JSON.stringify(rendered[other].layout.yaxis.range), before[other]);
  }
});
