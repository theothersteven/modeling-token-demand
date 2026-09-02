/* Client-side port of the single-attempt token-demand model.
 *
 * This file mirrors src/modeling_token_demand/model.py and optimizer.py so the
 * published page can solve the user's policy problem in the browser. It has
 * no DOM dependencies and is also loaded by tests/model_js.test.cjs, which
 * checks it against the audited Python results in figures/paradigms.json.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.TokenDemandModel = api;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  /* ---------- parameter metadata ---------- */

  // Industry parameters, in the order used by the paper's tables.
  const INDUSTRY_PARAMETERS = [
    {key: 'lambda', symbol: 'λ', label: 'Capability horizon', min: 1e-6,
     help: 'Baseline task-scope horizon; larger values make more work feasible at a given capability.',
     equation: 'q(s;m) = exp[−(s/(λm))^ν]'},
    {key: 'nu', symbol: 'ν', label: 'Capability shape', min: 1e-6,
     help: 'Controls how sharply feasibility falls as delegated scope exceeds the capability horizon.',
     equation: 'q(s;m) = exp[−(s/(λm))^ν]'},
    {key: 'a', symbol: 'a', label: 'Execution ease', min: 1e-6,
     help: 'Scales the execution horizon; larger values raise conditional success at a given scope and effort.',
     equation: 'r(s,x;m,η) = exp[−s/(am(ηx)^α)]'},
    {key: 'alpha', symbol: 'α', label: 'Inference returns', min: 1e-6, max: 0.999999,
     help: 'Controls diminishing returns to effective inference ηx; it must remain below one.',
     equation: 'r(s,x;m,η) = exp[−s/(am(ηx)^α)]'},
    {key: 'h0', symbol: 'h₀', label: 'Fixed review time', min: 0,
     help: 'Fixed review hours paid once for each delegated task, regardless of its scope.',
     equation: 'h(s) = κₕ{h₀ + h₁[(1+s)^β − 1]}'},
    {key: 'h1', symbol: 'h₁', label: 'Variable review scale', min: 1e-9,
     help: 'Scales the portion of review time that grows with delegated task scope.',
     equation: 'h(s) = κₕ{h₀ + h₁[(1+s)^β − 1]}'},
    {key: 'beta', symbol: 'β', label: 'Review elasticity', min: 0,
     help: 'Controls how quickly review time grows with scope: zero is fixed and one is proportional.',
     equation: 'h(s) = κₕ{h₀ + h₁[(1+s)^β − 1]}'},
    {key: 'b', symbol: 'b', label: 'Value of successful work', min: 1e-9,
     help: 'Dollar value created by one successfully completed work unit.',
     equation: 'u(s,x) = bP(s,x) − cx − wh(s)/s'},
    {key: 'w', symbol: 'w', label: 'Value of human attention', min: 1e-9,
     help: 'Opportunity cost of one hour of human review and verification.',
     equation: 'u(s,x) = bP(s,x) − cx − wh(s)/s'},
    {key: 'mu', symbol: 'μ', label: 'Hurdle location', min: -Infinity,
     help: 'Center of the adoption-hurdle distribution; adoption is 50% when optimized surplus equals μ.',
     equation: 'A(u) = 1/{1 + exp[−(u−μ)/σ]}'},
    {key: 'sigma', symbol: 'σ', label: 'Hurdle spread', min: 1e-9,
     help: 'Spread of adoption hurdles; smaller values make adoption more threshold-like.',
     equation: 'A(u) = 1/{1 + exp[−(u−μ)/σ]}'},
    {key: 'W', symbol: 'W', label: 'Potential work', min: 1e-9,
     help: 'Total potentially adoptable work available in the work-limited regime.',
     equation: 'Dᵂ = W·A(u*)·x*'},
    {key: 'H', symbol: 'H', label: 'Human attention', min: 1e-9,
     help: 'Total human review hours available in the attention-limited regime.',
     equation: 'Dᴴ = H·[s*/h(s*)]·x*'},
  ];

  // Scenario variables changed by comparative statics.
  const SCENARIO_PARAMETERS = [
    {key: 'm', symbol: 'm', label: 'Model capability', min: 1e-9,
     help: 'Expands both the feasible task frontier and the conditional execution horizon.',
     equation: 'q(s;m) = exp[−(s/(λm))^ν],  r(s,x;m,η) = exp[−s/(am(ηx)^α)]'},
    {key: 'eta', symbol: 'η', label: 'Token efficiency', min: 1e-9,
     help: 'Turns token effort x into effective inference ηx; larger values achieve the same reliability with fewer tokens.',
     equation: 'r(s,x;m,η) = exp[−s/(am(ηx)^α)]'},
    {key: 'c', symbol: 'c', label: 'Token price', min: 1e-9,
     help: 'Price paid for each unit of token effort x.',
     equation: 'u(s,x) = bP(s,x) − cx − wh(s)/s'},
    {key: 'v', symbol: 'κₕ', label: 'Review-time multiplier', min: 1e-9,
     help: 'Multiplies every component of review time; smaller values mean faster verification.',
     equation: 'h(s) = κₕ{h₀ + h₁[(1+s)^β − 1]}'},
  ];

  const REFERENCE_INDUSTRY = {
    lambda: 12, nu: 1.25, a: 4, alpha: 0.5, h0: 0.03, h1: 0.05, beta: 0.5,
    b: 100, w: 100, mu: 81, sigma: 4, W: 1e6, H: 1e5,
  };

  // Named one-parameter cases from calibrations.py.
  const INDUSTRY_PRESETS = [
    {name: 'Reference industry', changes: {}},
    {name: 'Low adoption hurdle', changes: {mu: 40}},
    {name: 'High adoption hurdle', changes: {mu: 95}},
    {name: 'Hard execution', changes: {a: 1}},
    {name: 'High capability requirement', changes: {lambda: 3}},
    {name: 'Low inference returns', changes: {alpha: 0.2}},
    {name: 'Slow-growing review', changes: {beta: 0.15}},
    {name: 'Nearly proportional review', changes: {beta: 0.95}},
  ];

  const BASELINE_SCENARIO = {m: 1, eta: 1, c: 1, v: 1};

  // Numerical search bounds: the same as paradigms.gallery_settings().
  const DEFAULT_SETTINGS = {
    minS: 0.002, maxS: 800, minX: 1, maxX: 2000, grid: 17, starts: 4,
  };

  function presetIndustry(name) {
    const preset = INDUSTRY_PRESETS.find(item => item.name === name);
    if (!preset) throw new Error(`Unknown industry preset: ${name}`);
    return Object.assign({}, REFERENCE_INDUSTRY, preset.changes);
  }

  function validateIndustry(ind) {
    const problems = [];
    for (const spec of INDUSTRY_PARAMETERS) {
      const value = ind[spec.key];
      if (!Number.isFinite(value)) problems.push(`${spec.symbol} must be a number`);
      else if (value < spec.min) problems.push(`${spec.symbol} must be at least ${spec.min}`);
      else if (spec.max !== undefined && value > spec.max) problems.push(`${spec.symbol} must be below one`);
    }
    if (ind.h0 === 0 && ind.beta === 0) problems.push('h₀ and β cannot both be zero');
    return problems;
  }

  /* ---------- the economic model (model.py) ---------- */

  function logistic(z) {
    if (z >= 0) return 1 / (1 + Math.exp(-z));
    const e = Math.exp(z);
    return e / (1 + e);
  }

  /** Evaluate one candidate policy (s, x) for an industry in a scenario. */
  function evaluate(ind, sc, s, x) {
    const q = Math.exp(-Math.pow(s / (ind.lambda * sc.m), ind.nu));
    const effective = sc.eta * x;
    const horizon = ind.a * sc.m * Math.pow(effective, ind.alpha);
    const exponent = -s / horizon;
    const r = exponent < -745 ? 0 : Math.exp(exponent);
    const P = q * r;
    const h = sc.v * (ind.h0 + ind.h1 * Math.expm1(ind.beta * Math.log1p(s)));
    const tokenCost = sc.c * x;
    const reviewCost = ind.w * h / s;
    const u = ind.b * P - tokenCost - reviewCost;   // surplus per work unit
    const J = s * u / h;                             // surplus per review hour
    const A = logistic((u - ind.mu) / ind.sigma);
    const leverage = s / h;
    return {
      s, x, q, r, P, h, tokenCost, reviewCost, u, J, A, leverage,
      effectiveInference: effective,
      rho: J + ind.w,                     // value (shadow price) of one review hour
      workDemand: ind.W * A * x,          // D in the work-limited regime
      attentionDemand: ind.H * leverage * x, // D in the attention-limited regime
      workAssigned: ind.W * A,
      attentionAssigned: ind.H * leverage,
    };
  }

  function objectiveOf(regime) {
    if (regime === 'work') return o => o.u;
    if (regime === 'attention') return o => o.J;
    throw new Error(`Unknown regime: ${regime}`);
  }

  /* ---------- numerical optimisation (optimizer.py) ---------- */

  /** Nelder–Mead on a 2-D box in log space; points outside are clamped. */
  function nelderMead(f, start, lower, upper, options) {
    const maxIterations = (options && options.maxIterations) || 600;
    const tol = (options && options.tolerance) || 1e-12;
    const n = 2;
    const clamp = p => p.map((v, i) => Math.min(upper[i], Math.max(lower[i], v)));
    const step = [(upper[0] - lower[0]) * 0.02, (upper[1] - lower[1]) * 0.02];
    let simplex = [clamp(start)];
    for (let i = 0; i < n; i++) {
      const p = start.slice();
      p[i] += (p[i] + step[i] <= upper[i]) ? step[i] : -step[i];
      simplex.push(clamp(p));
    }
    let values = simplex.map(p => f(p));
    let evaluations = values.length;
    for (let iteration = 0; iteration < maxIterations; iteration++) {
      const order = [0, 1, 2].sort((i, j) => values[i] - values[j]);
      simplex = order.map(i => simplex[i]);
      values = order.map(i => values[i]);
      const spread = Math.max(
        Math.abs(simplex[1][0] - simplex[0][0]), Math.abs(simplex[2][0] - simplex[0][0]),
        Math.abs(simplex[1][1] - simplex[0][1]), Math.abs(simplex[2][1] - simplex[0][1]));
      const valueSpread = Math.abs(values[2] - values[0]);
      if (spread < tol && valueSpread < tol * Math.max(1, Math.abs(values[0]))) break;
      const centroid = [(simplex[0][0] + simplex[1][0]) / 2, (simplex[0][1] + simplex[1][1]) / 2];
      const worst = simplex[2];
      const reflect = clamp([2 * centroid[0] - worst[0], 2 * centroid[1] - worst[1]]);
      const fr = f(reflect); evaluations++;
      if (fr < values[0]) {
        const expand = clamp([3 * centroid[0] - 2 * worst[0], 3 * centroid[1] - 2 * worst[1]]);
        const fe = f(expand); evaluations++;
        if (fe < fr) { simplex[2] = expand; values[2] = fe; }
        else { simplex[2] = reflect; values[2] = fr; }
        continue;
      }
      if (fr < values[1]) { simplex[2] = reflect; values[2] = fr; continue; }
      const outside = fr < values[2];
      const contract = outside
        ? clamp([centroid[0] + (reflect[0] - centroid[0]) / 2, centroid[1] + (reflect[1] - centroid[1]) / 2])
        : clamp([centroid[0] + (worst[0] - centroid[0]) / 2, centroid[1] + (worst[1] - centroid[1]) / 2]);
      const fc = f(contract); evaluations++;
      if (fc < (outside ? fr : values[2])) { simplex[2] = contract; values[2] = fc; continue; }
      for (let i = 1; i < 3; i++) {
        simplex[i] = clamp([simplex[0][0] + (simplex[i][0] - simplex[0][0]) / 2,
                            simplex[0][1] + (simplex[i][1] - simplex[0][1]) / 2]);
        values[i] = f(simplex[i]); evaluations++;
      }
    }
    const best = values.indexOf(Math.min(...values));
    return {point: simplex[best], value: values[best], evaluations};
  }

  /** Golden-section search of a 1-D unimodal minimum on [lo, hi]. */
  function goldenSection(g, lo, hi, tol) {
    const phi = (Math.sqrt(5) - 1) / 2;
    let a = lo, b = hi;
    let c = b - phi * (b - a), d = a + phi * (b - a);
    let fc = g(c), fd = g(d);
    while (b - a > tol) {
      if (fc < fd) { b = d; d = c; fd = fc; c = b - phi * (b - a); fc = g(c); }
      else { a = c; c = d; fc = fd; d = a + phi * (b - a); fd = g(d); }
    }
    return fc < fd ? c : d;
  }

  /**
   * Solve max objective over (s, x) with a log-space grid, local refinement
   * from the best grid points, and a coordinate polish. Returns the outcome.
   */
  function solve(ind, sc, regime, settings) {
    const cfg = Object.assign({}, DEFAULT_SETTINGS, settings || {});
    const objective = objectiveOf(regime);
    const lower = [Math.log(cfg.minS), Math.log(cfg.minX)];
    const upper = [Math.log(cfg.maxS), Math.log(cfg.maxX)];
    const value = p => objective(evaluate(ind, sc, Math.exp(p[0]), Math.exp(p[1])));
    const negative = p => -value(p);

    const candidates = [];
    for (let i = 0; i < cfg.grid; i++) {
      const ls = lower[0] + (upper[0] - lower[0]) * i / (cfg.grid - 1);
      for (let j = 0; j < cfg.grid; j++) {
        const lx = lower[1] + (upper[1] - lower[1]) * j / (cfg.grid - 1);
        candidates.push({point: [ls, lx], value: value([ls, lx])});
      }
    }
    candidates.sort((p, q) => q.value - p.value);
    let best = candidates[0];
    for (const start of candidates.slice(0, cfg.starts)) {
      const local = nelderMead(negative, start.point, lower, upper);
      let point = local.point, current = -local.value;
      // Coordinate polish: exact 1-D searches handle optima on the effort
      // floor x = 1, where a simplex can stall against the clamp.
      for (let pass = 0; pass < 3; pass++) {
        const width = 0.35;
        const s0 = Math.max(lower[0], point[0] - width), s1 = Math.min(upper[0], point[0] + width);
        const ls = goldenSection(t => negative([t, point[1]]), s0, s1, 1e-11);
        const x0 = Math.max(lower[1], point[1] - width), x1 = Math.min(upper[1], point[1] + width);
        const lx = goldenSection(t => negative([ls, t]), x0, x1, 1e-11);
        const polished = value([ls, lx]);
        if (polished >= current) { point = [ls, lx]; current = polished; }
        else break;
      }
      if (current > best.value) best = {point, value: current};
    }
    const outcome = evaluate(ind, sc, Math.exp(best.point[0]), Math.exp(best.point[1]));
    outcome.objective = best.value;
    outcome.boundHits = boundHits(outcome, cfg);
    return outcome;
  }

  /** Numerical bound hits, excluding the economic effort floor x = 1. */
  function boundHits(outcome, cfg) {
    const hits = [];
    if (outcome.s <= cfg.minS * 1.0001 || outcome.s >= cfg.maxS / 1.0001) hits.push('s');
    if (outcome.x >= cfg.maxX / 1.0001) hits.push('x');
    return hits;
  }

  /** Brent's root finder on [a, b] (port of scipy.optimize.brentq). */
  function brentq(f, a, b, fa, fb, xtol, maxIterations) {
    if (fa * fb > 0) throw new Error('root is not bracketed');
    let c = a, fc = fa, d = b - a, e = d;
    for (let i = 0; i < maxIterations; i++) {
      if (fb * fc > 0) { c = a; fc = fa; d = b - a; e = d; }
      if (Math.abs(fc) < Math.abs(fb)) { a = b; b = c; c = a; fa = fb; fb = fc; fc = fa; }
      const tol = 2 * Number.EPSILON * Math.abs(b) + xtol / 2;
      const m = (c - b) / 2;
      if (Math.abs(m) <= tol || fb === 0) return b;
      if (Math.abs(e) >= tol && Math.abs(fa) > Math.abs(fb)) {
        let p, q;
        const s = fb / fa;
        if (a === c) { p = 2 * m * s; q = 1 - s; }
        else {
          const qa = fa / fc, r = fb / fc;
          p = s * (2 * m * qa * (qa - r) - (b - a) * (r - 1));
          q = (qa - 1) * (r - 1) * (s - 1);
        }
        if (p > 0) q = -q; else p = -p;
        if (2 * p < Math.min(3 * m * q - Math.abs(tol * q), Math.abs(e * q))) { e = d; d = p / q; }
        else { d = m; e = m; }
      } else { d = m; e = m; }
      a = b; fa = fb;
      b += Math.abs(d) > tol ? d : (m > 0 ? tol : -tol);
      fb = f(b);
    }
    return b;
  }

  /**
   * Token price at which the optimized objective equals `target`. Every
   * candidate price reoptimizes (s, x). Returns {price, outcome, gap}.
   */
  function reservationPrice(ind, sc, regime, target, settings) {
    if (!Number.isFinite(target)) throw new Error('target value must be finite');
    const objective = objectiveOf(regime);
    const cache = new Map();
    const gap = logPrice => {
      const price = Math.exp(logPrice);
      const outcome = solve(ind, Object.assign({}, sc, {c: price}), regime, settings);
      cache.set(logPrice, outcome);
      return objective(outcome) - target;
    };
    const start = Math.log(sc.c);
    const initial = gap(start);
    const tolerance = 1e-10 * Math.max(1, Math.abs(target));
    if (Math.abs(initial) <= tolerance) return {price: sc.c, outcome: cache.get(start), gap: initial};
    const step = Math.log(2);
    let lower, upper, lowerGap, upperGap;
    if (initial > 0) {
      lower = start; lowerGap = initial; upper = start + step; upperGap = gap(upper);
      for (let i = 0; i < 48 && upperGap > 0; i++) { lower = upper; lowerGap = upperGap; upper += step; upperGap = gap(upper); }
      if (upperGap > 0) throw new Error('could not bracket a finite reservation price');
    } else {
      upper = start; upperGap = initial; lower = start - step; lowerGap = gap(lower);
      for (let i = 0; i < 48 && lowerGap < 0; i++) { upper = lower; upperGap = lowerGap; lower -= step; lowerGap = gap(lower); }
      if (lowerGap < 0) throw new Error('target value is unattainable at a positive token price');
    }
    const root = brentq(gap, lower, upper, lowerGap, upperGap, 1e-6, 64);
    const finalGap = cache.has(root) ? objective(cache.get(root)) - target : gap(root);
    return {price: Math.exp(root), outcome: cache.get(root), gap: finalGap};
  }

  /* ---------- outcome variables the plot builder can show ---------- */

  const OUTPUTS = [
    {key: 'D', label: 'Token demand', unit: 'token units', log: true,
     value: (o, ctx) => ctx.regime === 'work' ? o.workDemand : o.attentionDemand},
    {key: 'R', label: 'Token spending', unit: '$', log: true,
     value: (o, ctx) => ctx.scenario.c * (ctx.regime === 'work' ? o.workDemand : o.attentionDemand)},
    {key: 'A', label: 'Adoption', unit: '% of potential work', log: false, percent: true,
     value: o => o.A},
    {key: 'Q', label: 'Assigned work', unit: 'work units', log: true,
     value: (o, ctx) => ctx.regime === 'work' ? o.workAssigned : o.attentionAssigned},
    {key: 'QP', label: 'Completed work', unit: 'work units', log: true,
     value: (o, ctx) => (ctx.regime === 'work' ? o.workAssigned : o.attentionAssigned) * o.P},
    {key: 'u', label: 'Surplus per work unit, u', unit: '$ per work unit', log: false, value: o => o.u},
    {key: 'J', label: 'Surplus per review hour, J', unit: '$ per hour', log: false, value: o => o.J},
    {key: 'rho', label: 'Value created per review hour, ρ', unit: '$ per hour', log: true, value: o => o.rho},
    {key: 'cres', label: 'Reservation token price, c_res / c₀', unit: 'relative to baseline', log: true,
     reservation: true, value: (o, ctx) => ctx.reservationPrice},
    {key: 's', label: 'Optimal scope, s*', unit: 'work units per task', log: true, value: o => o.s},
    {key: 'x', label: 'Optimal effort, x*', unit: 'token units per work unit', log: true, value: o => o.x},
    {key: 'etax', label: 'Effective inference, η·x*', unit: 'per work unit', log: true, value: o => o.effectiveInference},
    {key: 'sx', label: 'Tokens per task, s*·x*', unit: 'token units', log: true, value: o => o.s * o.x},
    {key: 'P', label: 'Success probability, P', unit: '', log: false, percent: true, value: o => o.P},
    {key: 'q', label: 'Feasible share, q', unit: '', log: false, percent: true, value: o => o.q},
    {key: 'r', label: 'Conditional success, r', unit: '', log: false, percent: true, value: o => o.r},
    {key: 'h', label: 'Review time per task, h(s*)', unit: 'hours', log: true, value: o => o.h},
    {key: 'ell', label: 'Supervisory leverage, s*/h(s*)', unit: 'work units per review hour', log: true, value: o => o.leverage},
    {key: 'hours', label: 'Review hours used', unit: 'hours', log: true,
     value: (o, ctx) => ctx.regime === 'work' ? o.workAssigned * o.h / o.s : ctx.industry.H},
    {key: 'tph', label: 'Tokens per review hour', unit: 'token units per hour', log: true, value: o => o.leverage * o.x},
  ];

  /**
   * Sweep one variable and compute every requested output for one industry.
   * `sweep` = {key, values}; key is a scenario key (m, eta, c, v) or an industry key.
   * Returns {values, series: {outputKey: number[]}, outcomes, warnings}.
   */
  function sweep(industry, sweepSpec, regime, outputKeys, settings) {
    const wantsReservation = outputKeys.some(key => OUTPUTS.find(o => o.key === key).reservation);
    const isScenario = SCENARIO_PARAMETERS.some(p => p.key === sweepSpec.key);
    const series = {}; outputKeys.forEach(key => { series[key] = []; });
    const outcomes = [], warnings = new Set();
    let target = null;
    if (wantsReservation) {
      if (sweepSpec.key === 'c') warnings.add('Reservation price is undefined when token price is the swept variable.');
      else target = objectiveOf(regime)(solve(industry, BASELINE_SCENARIO, regime, settings));
    }
    for (const value of sweepSpec.values) {
      const ind = isScenario ? industry : Object.assign({}, industry, {[sweepSpec.key]: value});
      const sc = isScenario ? Object.assign({}, BASELINE_SCENARIO, {[sweepSpec.key]: value}) : BASELINE_SCENARIO;
      const outcome = solve(ind, sc, regime, settings);
      if (outcome.boundHits.length) warnings.add(`Numerical bound hit for ${outcome.boundHits.join(', ')} at ${sweepSpec.key} = ${Number(value.toPrecision(4))}.`);
      const ctx = {regime, scenario: sc, industry: ind, reservationPrice: NaN};
      if (target !== null) {
        try { ctx.reservationPrice = reservationPrice(ind, sc, regime, target, settings).price / BASELINE_SCENARIO.c; }
        catch (error) { warnings.add(`Reservation price: ${error.message}`); }
      }
      outcomes.push(outcome);
      for (const key of outputKeys) series[key].push(OUTPUTS.find(o => o.key === key).value(outcome, ctx));
    }
    return {values: sweepSpec.values.slice(), series, outcomes, warnings: [...warnings]};
  }

  /** Log- or linear-spaced grid that always contains `anchor` when inside the range. */
  function axisValues(lo, hi, points, scale, anchor) {
    const values = [];
    for (let i = 0; i < points; i++) {
      const t = points === 1 ? 0 : i / (points - 1);
      values.push(scale === 'log' ? Math.exp(Math.log(lo) + (Math.log(hi) - Math.log(lo)) * t) : lo + (hi - lo) * t);
    }
    if (Number.isFinite(anchor) && anchor > Math.min(lo, hi) && anchor < Math.max(lo, hi)
        && !values.some(v => Math.abs(v - anchor) <= 1e-12 * Math.max(1, Math.abs(anchor)))) values.push(anchor);
    return values.sort((p, q) => p - q);
  }

  return {
    INDUSTRY_PARAMETERS, SCENARIO_PARAMETERS, REFERENCE_INDUSTRY, INDUSTRY_PRESETS,
    BASELINE_SCENARIO, DEFAULT_SETTINGS, OUTPUTS,
    presetIndustry, validateIndustry, evaluate, solve, reservationPrice, sweep, axisValues,
    nelderMead, brentq, goldenSection, boundHits,
  };
});
