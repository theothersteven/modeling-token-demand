/* The browser model must reproduce the audited Python results. */
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const M = require('../src/modeling_token_demand/paper_assets/model.js');
const report = JSON.parse(fs.readFileSync(path.join(__dirname, '../figures/paradigms.json'))).main;

const FIELD = {
  capability_horizon_hours: 'lambda', capability_shape: 'nu', execution_scale: 'a',
  inference_returns: 'alpha', verification_fixed_hours: 'h0', verification_scale: 'h1',
  verification_elasticity: 'beta', value_per_work_hour: 'b', human_cost_per_hour: 'w',
  adoption_location: 'mu', adoption_scale: 'sigma', potential_work_hours: 'W',
  human_attention_hours: 'H',
};
const AXIS = {capability: 'm', efficiency: 'eta', price: 'c'};

function industryOf(record) {
  const ind = {};
  for (const [python, key] of Object.entries(FIELD)) ind[key] = record.industry[python];
  if (ind.H === null || ind.H === undefined) ind.H = 100000;
  return ind;
}

function relativeError(actual, expected) {
  return Math.abs(actual - expected) / Math.max(1, Math.abs(expected));
}

test('presets reproduce the calibration table', () => {
  assert.deepEqual(M.presetIndustry('Hard execution').a, 1);
  assert.deepEqual(M.presetIndustry('Slow-growing review').beta, 0.15);
  assert.deepEqual(M.validateIndustry(M.REFERENCE_INDUSTRY), []);
  assert.ok(M.validateIndustry(Object.assign({}, M.REFERENCE_INDUSTRY, {alpha: 1})).length);
});

test('evaluate matches the Python model at a recorded optimum', () => {
  const record = report.curves.find(c => c.regime === 'work' && c.axis === 'capability'
    && c.industry.name === 'Reference industry');
  const i = record.baseline_index;
  const o = M.evaluate(industryOf(record), {m: 1, eta: 1, c: 1, v: 1}, record.s[i], record.x[i]);
  assert.ok(relativeError(o.u, record.surplus[i]) < 1e-12);
  assert.ok(relativeError(o.A, record.adoption[i]) < 1e-12);
  assert.ok(relativeError(o.P, record.success[i]) < 1e-12);
  assert.ok(relativeError(o.workDemand, record.demand[i]) < 1e-12);
  assert.ok(relativeError(o.h, record.verification_hours[i]) < 1e-12);
});

// Independent re-solution at the endpoints, the baseline, and the demand
// extrema of every audited main-figure curve, using the Python tolerances.
for (const record of report.curves) {
  const name = `${record.regime} / ${record.axis} / ${record.industry.name}`;
  test(`solver reproduces ${name}`, () => {
    const ind = industryOf(record);
    const demand = record.demand;
    const indices = new Set([0, record.values.length - 1, record.baseline_index,
      demand.indexOf(Math.max(...demand)), demand.indexOf(Math.min(...demand))]);
    for (const i of indices) {
      const sc = Object.assign({m: 1, eta: 1, c: 1, v: 1}, {[AXIS[record.axis]]: record.values[i]});
      const o = M.solve(ind, sc, record.regime);
      const expectedObjective = record.regime === 'work' ? record.surplus[i]
        : record.attention_value[i] - ind.w;
      const objective = record.regime === 'work' ? o.u : o.J;
      assert.ok(relativeError(objective, expectedObjective) < 1e-8,
        `${name} @ ${record.values[i]}: objective ${objective} vs ${expectedObjective}`);
      const got = record.regime === 'work' ? o.workDemand : o.attentionDemand;
      assert.ok(relativeError(got, demand[i]) < 5e-4,
        `${name} @ ${record.values[i]}: demand ${got} vs ${demand[i]}`);
      assert.deepEqual(o.boundHits, []);
    }
  });
}

for (const record of report.curves.filter(c => c.reservation_price)) {
  const name = `${record.regime} / ${record.industry.name}`;
  test(`reservation price reproduces ${name}`, () => {
    const ind = industryOf(record);
    const target = record.reservation_target_surplus;
    const capabilities = record.reservation_capability;
    for (const i of [0, Math.floor(capabilities.length / 2), capabilities.length - 1]) {
      const result = M.reservationPrice(ind, {m: capabilities[i], eta: 1, c: 1, v: 1}, record.regime, target);
      assert.ok(Math.abs(result.price / record.reservation_price[i] - 1) < 2e-5,
        `${name} @ m=${capabilities[i]}: ${result.price} vs ${record.reservation_price[i]}`);
    }
  });
}

test('sweep exposes every output and flags an undefined reservation price', () => {
  const values = M.axisValues(0.5, 2, 5, 'log', 1);
  assert.ok(values.includes(1));
  const keys = M.OUTPUTS.map(o => o.key);
  const result = M.sweep(M.REFERENCE_INDUSTRY, {key: 'm', values}, 'work', keys, {grid: 9, starts: 2});
  for (const key of keys) assert.equal(result.series[key].length, values.length);
  assert.ok(result.series.cres.every(Number.isFinite));
  const priced = M.sweep(M.REFERENCE_INDUSTRY, {key: 'c', values}, 'work', ['cres'], {grid: 9, starts: 2});
  assert.ok(priced.warnings.some(w => w.includes('undefined')));
  const industrySweep = M.sweep(M.REFERENCE_INDUSTRY, {key: 'beta', values: [0.2, 0.8]}, 'attention', ['D'], {grid: 9, starts: 2});
  assert.ok(industrySweep.series.D[0] > industrySweep.series.D[1]);
});
