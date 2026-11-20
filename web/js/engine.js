// The genetic algorithm in the browser.
//
// A line-for-line port of src/tsp_ga/engine.py and ga.py. Given the same
// instance, settings and seed it produces exactly the same populations, which
// web/test/engine.test.js checks against runs exported from Python. Keep the
// two in step.

import { makeRng, randInt, randUnit } from "./rng.js";

export const NEIGHBOURS = 8;
export const FULL_EFFORT = 1000000;
export const CROSSOVERS = { pmx: 0, ox: 1, er: 2 };
export const MUTATIONS = { swap: 0, inversion: 1 };
export const SCHEDULES = ["funnel", "fixed", "reverse", "geometric"];

// Columns of a generation's log row.
export const COL = { KEPT: 0, EVALS: 1, CHECKS: 2, BEST: 3, N_POP: 4, SUM_POP: 5, SUMSQ_POP: 6, SUM_SEL: 7, SUM_OFF: 8, SUM_CHILD: 9, SHARED: 10, MOVES: 11, MADE: 12 };

/** The settings of the 2020 program. */
export const DEFAULTS = Object.freeze({
  generations: 300,
  schedule: "funnel",
  base: 200,
  extra: 500,
  init_size: 500,
  crossover: "pmx",
  mutation: "swap",
  mutation_rate: 0.15,
  effort_start: 0,
  effort_end: 0,
  budget: 0,
});

export function validate(config) {
  const int = (v) => Number.isInteger(v);
  if (!(config.crossover in CROSSOVERS)) throw new Error(`unknown crossover: ${config.crossover}`);
  if (!(config.mutation in MUTATIONS)) throw new Error(`unknown mutation: ${config.mutation}`);
  if (!SCHEDULES.includes(config.schedule)) throw new Error(`unknown schedule: ${config.schedule}`);
  if (!int(config.generations) || config.generations < 1) throw new Error("generations must be a whole number, at least 1");
  if (!int(config.base) || config.base < 2) throw new Error("base must be a whole number, at least 2");
  if (!int(config.extra) || config.extra < 0) throw new Error("extra must be a whole number, not negative");
  if (!int(config.init_size) || config.init_size < 2) throw new Error("init_size must be a whole number, at least 2");
  if (!(config.mutation_rate >= 0 && config.mutation_rate <= 1)) throw new Error("mutation_rate must be between 0 and 1");
  if (!int(config.effort_start) || !int(config.effort_end) || !int(config.budget) || config.effort_start < 0 || config.effort_end < 0 || config.budget < 0) throw new Error("effort and budget must be whole numbers, not negative");
  return config;
}

/** TSPLIB EUC_2D distances, and each city's nearest neighbours (ties to the lower index). */
export function makeInstance(coords, optimum = null) {
  const n = coords.length;
  if (n < 4) throw new Error("an instance needs at least four cities");
  const distance = new Int32Array(n * n);
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      const dx = coords[i][0] - coords[j][0];
      const dy = coords[i][1] - coords[j][1];
      distance[i * n + j] = Math.floor(Math.sqrt(dx * dx + dy * dy) + 0.5);
    }
  }
  const k = Math.min(NEIGHBOURS, n - 1);
  const neighbours = new Int32Array(n * k);
  for (let city = 0; city < n; city += 1) {
    const others = [];
    for (let j = 0; j < n; j += 1) if (j !== city) others.push(j);
    others.sort((a, b) => distance[city * n + a] - distance[city * n + b] || a - b);
    for (let q = 0; q < k; q += 1) neighbours[city * k + q] = others[q];
  }
  return { n, k, coords, distance, neighbours, optimum };
}

/** Individuals kept in each generation. */
export function sizes(config) {
  const total = config.generations;
  const out = new Array(total);
  for (let g = 1; g <= total; g += 1) {
    if (config.schedule === "fixed") out[g - 1] = config.base;
    else if (config.schedule === "funnel") out[g - 1] = config.base + Math.floor((config.extra * (total - g)) / total);
    else if (config.schedule === "reverse") out[g - 1] = config.base + Math.floor((config.extra * (g - 1)) / total);
    else {
      const top = config.base + config.extra;
      out[g - 1] = Math.floor(top * (config.base / top) ** ((g - 1) / Math.max(total - 1, 1)) + 0.5);
    }
  }
  return out;
}

/** 2-opt moves allowed per child in each generation. */
export function effort(config) {
  const span = Math.max(config.generations - 1, 1);
  const out = new Array(config.generations);
  for (let g = 0; g < config.generations; g += 1) {
    out[g] = config.effort_start + Math.floor(((config.effort_end - config.effort_start) * g) / span);
  }
  return out;
}

/** Tour evaluations a run without local search makes. */
export function evaluations(config) {
  return config.init_size + 2 * sizes(config).reduce((a, b) => a + b, 0);
}

export function tourLength(instance, tour) {
  const { n, distance } = instance;
  let total = 0;
  for (let i = 0; i < n - 1; i += 1) total += distance[tour[i] * n + tour[i + 1]];
  return total + distance[tour[n - 1] * n + tour[0]];
}

export function randomTour(tour, rng) {
  const n = tour.length;
  for (let i = 0; i < n; i += 1) tour[i] = i;
  for (let i = n - 1; i > 0; i -= 1) {
    const j = randInt(rng, i + 1);
    const t = tour[i];
    tour[i] = tour[j];
    tour[j] = t;
  }
}

/** Partially mapped crossover: a[lo:hi] in place, the rest from b. */
export function pmx(a, b, child, lo, hi, where) {
  const n = a.length;
  for (let i = 0; i < n; i += 1) where[a[i]] = i;
  for (let i = lo; i < hi; i += 1) child[i] = a[i];
  for (let i = 0; i < n; i += 1) {
    if (i >= lo && i < hi) continue;
    let city = b[i];
    while (where[city] >= lo && where[city] < hi) city = b[where[city]];
    child[i] = city;
  }
}

/** Order crossover: a[lo:hi] in place, the rest in the order they have in b. */
export function ox(a, b, child, lo, hi, where) {
  const n = a.length;
  for (let i = 0; i < n; i += 1) where[a[i]] = i;
  for (let i = lo; i < hi; i += 1) child[i] = a[i];
  let at = hi % n;
  for (let k = 0; k < n; k += 1) {
    const city = b[(hi + k) % n];
    if (where[city] >= lo && where[city] < hi) continue;
    child[at] = city;
    at = (at + 1) % n;
  }
}

function addEdge(adj, deg, u, v) {
  for (let k = 0; k < deg[u]; k += 1) if (adj[u * 4 + k] === v) return;
  adj[u * 4 + deg[u]] = v;
  deg[u] += 1;
}

/** Edge recombination: build the child from the parents' edges. */
export function er(a, b, child, start, rng, s) {
  const n = a.length;
  const { adj, deg, rest, slot } = s;
  for (let i = 0; i < n; i += 1) {
    deg[i] = 0;
    rest[i] = i;
    slot[i] = i;
  }
  for (let i = 0; i < n; i += 1) {
    const j = i + 1 < n ? i + 1 : 0;
    addEdge(adj, deg, a[i], a[j]);
    addEdge(adj, deg, a[j], a[i]);
    addEdge(adj, deg, b[i], b[j]);
    addEdge(adj, deg, b[j], b[i]);
  }
  let left = n;
  let city = start;
  for (let k = 0; k < n; k += 1) {
    child[k] = city;
    const last = rest[left - 1];
    rest[slot[city]] = last;
    slot[last] = slot[city];
    left -= 1;
    for (let q = 0; q < deg[city]; q += 1) {
      const other = adj[city * 4 + q];
      for (let r = 0; r < deg[other]; r += 1) {
        if (adj[other * 4 + r] === city) {
          adj[other * 4 + r] = adj[other * 4 + deg[other] - 1];
          deg[other] -= 1;
          break;
        }
      }
    }
    if (left === 0) break;
    let fewest = 99;
    let ties = 0;
    for (let q = 0; q < deg[city]; q += 1) {
      const d = deg[adj[city * 4 + q]];
      if (d < fewest) {
        fewest = d;
        ties = 1;
      } else if (d === fewest) ties += 1;
    }
    if (ties === 0) {
      city = rest[randInt(rng, left)];
    } else {
      let pick = ties > 1 ? randInt(rng, ties) : 0;
      for (let q = 0; q < deg[city]; q += 1) {
        if (deg[adj[city * 4 + q]] === fewest) {
          if (pick === 0) {
            city = adj[city * 4 + q];
            break;
          }
          pick -= 1;
        }
      }
    }
  }
}

export function mutate(tour, kind, rng) {
  const n = tour.length;
  let i = randInt(rng, n);
  let j = randInt(rng, n);
  if (kind === 0) {
    const t = tour[i];
    tour[i] = tour[j];
    tour[j] = t;
    return;
  }
  if (i > j) [i, j] = [j, i];
  while (i < j) {
    const t = tour[i];
    tour[i] = tour[j];
    tour[j] = t;
    i += 1;
    j -= 1;
  }
}

function reverse(tour, pos, i, j) {
  const n = tour.length;
  let length = ((j - i + n) % n) + 1;
  if (2 * length > n) {
    const i2 = (j + 1) % n;
    j = (i - 1 + n) % n;
    i = i2;
    length = n - length;
  }
  for (let step = Math.floor(length / 2); step > 0; step -= 1) {
    const u = tour[i];
    const v = tour[j];
    tour[i] = v;
    pos[v] = i;
    tour[j] = u;
    pos[u] = j;
    i = i + 1 < n ? i + 1 : 0;
    j = j > 0 ? j - 1 : n - 1;
  }
}

/**
 * Improve a tour in place with 2-opt moves restricted to near neighbours.
 * out receives [change in length, moves made, moves examined].
 */
export function twoOpt(instance, tour, pos, maxMoves, out) {
  const { n, k, distance, neighbours } = instance;
  for (let i = 0; i < n; i += 1) pos[tour[i]] = i;
  let gain = 0;
  let moves = 0;
  let checks = 0;
  let improved = true;
  while (improved && moves < maxMoves) {
    improved = false;
    for (let i = 0; i < n; i += 1) {
      if (moves >= maxMoves) break;
      const a = tour[i];
      let b = tour[i + 1 < n ? i + 1 : 0];
      let dab = distance[a * n + b];
      let done = false;
      for (let q = 0; q < k; q += 1) {
        const c = neighbours[a * k + q];
        const dac = distance[a * n + c];
        if (dac >= dab) break;
        const j = pos[c];
        const d = tour[j + 1 < n ? j + 1 : 0];
        checks += 1;
        const delta = dac + distance[b * n + d] - dab - distance[c * n + d];
        if (delta < 0) {
          reverse(tour, pos, i + 1 < n ? i + 1 : 0, j);
          gain += delta;
          moves += 1;
          improved = true;
          done = true;
          break;
        }
      }
      if (done) continue;
      b = tour[i > 0 ? i - 1 : n - 1];
      dab = distance[a * n + b];
      for (let q = 0; q < k; q += 1) {
        const c = neighbours[a * k + q];
        const dac = distance[a * n + c];
        if (dac >= dab) break;
        const j = pos[c];
        const d = tour[j > 0 ? j - 1 : n - 1];
        checks += 1;
        const delta = dac + distance[b * n + d] - dab - distance[c * n + d];
        if (delta < 0) {
          reverse(tour, pos, i, j > 0 ? j - 1 : n - 1);
          gain += delta;
          moves += 1;
          improved = true;
          break;
        }
      }
    }
  }
  out[0] = gain;
  out[1] = moves;
  out[2] = checks;
}

/** Make two offspring of a and b in c1 and c2. */
export function cross(crossover, a, b, c1, c2, rng, s) {
  const n = a.length;
  if (crossover === 2) {
    er(a, b, c1, a[0], rng, s);
    er(b, a, c2, b[0], rng, s);
    return;
  }
  const i = randInt(rng, n);
  const j = randInt(rng, n);
  const lo = Math.min(i, j);
  const hi = Math.max(i, j) + 1;
  if (crossover === 0) {
    pmx(a, b, c1, lo, hi, s.where);
    pmx(b, a, c2, lo, hi, s.where);
  } else {
    ox(a, b, c1, lo, hi, s.where);
    ox(b, a, c2, lo, hi, s.where);
  }
}

export function scratch(n) {
  return { where: new Int32Array(n), adj: new Int32Array(n * 4), deg: new Int32Array(n), rest: new Int32Array(n), slot: new Int32Array(n) };
}

/**
 * A run that can be advanced one generation at a time.
 *
 *   const run = new Run(instance, config, seed);
 *   while (!run.done) run.step();
 *
 * `sizes` and `effort` may be given to override the schedules computed from
 * the config (the parity test feeds in Python's).
 */
export class Run {
  constructor(instance, config, seed, schedules = {}) {
    validate(config);
    this.instance = instance;
    this.config = config;
    this.sizes = schedules.sizes || sizes(config);
    this.effort = schedules.effort || effort(config);
    this.rng = makeRng(seed);
    const n = instance.n;
    let cap = config.init_size;
    for (const s of this.sizes) cap = Math.max(cap, 2 * s);
    const rows = () => Array.from({ length: cap }, () => new Int32Array(n));
    this.tours = rows();
    this.spare = rows();
    this.lengths = new Float64Array(cap);
    this.spareLengths = new Float64Array(cap);
    this.c1 = new Int32Array(n);
    this.c2 = new Int32Array(n);
    this.pos = new Int32Array(n);
    this.nxt = new Int32Array(n);
    this.prv = new Int32Array(n);
    this.scratch = scratch(n);
    this.ls = [0, 0, 0];
    this.order = new Array(cap);
    this.best = new Int32Array(n);
    this.log = [];
    this.generation = 0;
    this.done = false;

    for (let r = 0; r < config.init_size; r += 1) {
      randomTour(this.tours[r], this.rng);
      this.lengths[r] = tourLength(instance, this.tours[r]);
    }
    this.m = config.init_size;
    this.evals = config.init_size;
    this.checks = 0;
    this.bestLength = this.lengths[0];
    this.best.set(this.tours[0]);
    for (let r = 1; r < this.m; r += 1) {
      if (this.lengths[r] < this.bestLength) {
        this.bestLength = this.lengths[r];
        this.best.set(this.tours[r]);
      }
    }
  }

  /** Work done so far, in tour evaluations. */
  get cost() {
    return this.evals + (4 * this.checks) / this.instance.n;
  }

  /** Advance one generation. Returns its log row, or null if the run is over. */
  step() {
    if (this.done) return null;
    const { instance, config, rng } = this;
    const n = instance.n;
    const g = this.generation;
    const size = this.sizes[g];
    const m = this.m;
    const lengths = this.lengths;
    const order = this.order;
    for (let r = 0; r < m; r += 1) order[r] = r;
    order.length = m;
    order.sort((a, b) => lengths[a] - lengths[b] || a - b);
    const kept = Math.min(size, m);
    let sumPop = 0;
    let sumSqPop = 0;
    for (let r = 0; r < m; r += 1) {
      sumPop += lengths[r];
      sumSqPop += lengths[r] * lengths[r];
    }
    let sumSel = 0;
    for (let r = 0; r < kept; r += 1) {
      this.spare[r].set(this.tours[order[r]]);
      this.spareLengths[r] = lengths[order[r]];
      sumSel += this.spareLengths[r];
    }
    [this.tours, this.spare] = [this.spare, this.tours];
    [this.lengths, this.spareLengths] = [this.spareLengths, this.lengths];
    const tours = this.tours;
    const len = this.lengths;

    const top = tours[0];
    for (let i = 0; i < n; i += 1) {
      const u = top[i];
      const v = top[i + 1 < n ? i + 1 : 0];
      this.nxt[u] = v;
      this.prv[v] = u;
    }
    let shared = 0;
    for (let r = 1; r < kept; r += 1) {
      const t = tours[r];
      for (let i = 0; i < n; i += 1) {
        const u = t[i];
        const v = t[i + 1 < n ? i + 1 : 0];
        if (this.nxt[u] === v || this.prv[u] === v) shared += 1;
      }
    }

    const crossover = CROSSOVERS[config.crossover];
    const mutation = MUTATIONS[config.mutation];
    let sumOff = 0;
    let sumChild = 0;
    let moves = 0;
    let made = 0;
    for (let c = 0; c < size; c += 1) {
      if (config.budget > 0 && this.evals * n + 4 * this.checks >= config.budget * n) break;
      made += 1;
      const pa = randInt(rng, kept);
      const pb = randInt(rng, kept);
      cross(crossover, tours[pa], tours[pb], this.c1, this.c2, rng, this.scratch);
      if (randUnit(rng) < config.mutation_rate) {
        mutate(this.c1, mutation, rng);
        mutate(this.c2, mutation, rng);
      }
      const len1 = tourLength(instance, this.c1);
      const len2 = tourLength(instance, this.c2);
      this.evals += 2;
      sumOff += len1 + len2;
      const row = kept + c;
      if (len1 >= len2) {
        tours[row].set(this.c2);
        len[row] = len2;
      } else {
        tours[row].set(this.c1);
        len[row] = len1;
      }
      if (this.effort[g] > 0) {
        twoOpt(instance, tours[row], this.pos, this.effort[g], this.ls);
        len[row] += this.ls[0];
        moves += this.ls[1];
        this.checks += this.ls[2];
      }
      sumChild += len[row];
      if (len[row] < this.bestLength) {
        this.bestLength = len[row];
        this.best.set(tours[row]);
      }
    }
    this.m = kept + made;
    const row = [kept, this.evals, this.checks, this.bestLength, m, sumPop, sumSqPop, sumSel, sumOff, sumChild, shared, moves, made];
    this.log.push(row);
    this.generation += 1;
    if (made < size || this.generation >= this.sizes.length) this.done = true;
    return row;
  }
}

/** A generation's log row in a breeder's terms. Lengths: smaller is better. */
export function breeding(row, n) {
  const nPop = row[COL.N_POP];
  const kept = row[COL.KEPT];
  const made = row[COL.MADE];
  const mean = row[COL.SUM_POP] / nPop;
  const sd = Math.sqrt(Math.max(row[COL.SUMSQ_POP] / nPop - mean * mean, 0));
  const selected = row[COL.SUM_SEL] / kept;
  const offspring = made > 0 ? row[COL.SUM_OFF] / (2 * made) : NaN;
  return {
    mean,
    sd,
    selected,
    offspring,
    kept,
    made,
    // How much longer the average offspring is than the average parent.
    load: offspring / selected - 1,
    // Share of edges the survivors have in common with the best one.
    kinship: kept > 1 ? row[COL.SHARED] / ((kept - 1) * n) : NaN,
  };
}
