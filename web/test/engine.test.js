import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { uint32, makeRng, randInt, randUnit } from "../js/rng.js";
import * as E from "../js/engine.js";

const load = (path) => JSON.parse(readFileSync(new URL(path, import.meta.url)));
const fixtures = load("./fixtures.json");
const data = load("../data/instances.json");
const instances = Object.fromEntries(Object.entries(data).map(([k, v]) => [k, E.makeInstance(v.coords, v.optimum)]));
const isTour = (tour, n) => tour.length === n && [...tour].sort((a, b) => a - b).every((c, i) => c === i);

test("the random number generator matches the Python one", () => {
  // The same constants are asserted in tests/test_rng.py.
  assert.deepEqual([0, 1, 2].map((i) => uint32(0, 0, i)), [1680814304, 1471256801, 3260664659]);
  assert.deepEqual([0, 59].map((i) => uint32(7, 100, i)), [29833046, 1845834364]);
});

test("random integers are in range and roughly uniform", () => {
  const rng = makeRng(5);
  const counts = new Array(7).fill(0);
  for (let i = 0; i < 70000; i += 1) counts[randInt(rng, 7)] += 1;
  assert.ok(counts.every((c) => c > 9500 && c < 10500));
  assert.equal(rng.counter, 70000);
  const u = randUnit(rng);
  assert.ok(u > 0 && u < 1);
  assert.equal(randInt(makeRng(1), 1), 0);
});

test("distances and neighbour lists match Python's", () => {
  for (const c of fixtures.cases) {
    const inst = instances[c.instance];
    assert.equal(inst.distance.reduce((a, b) => a + b, 0), c.distance_sum, c.instance);
    assert.deepEqual(Array.from(inst.neighbours.slice(0, inst.k)), c.neighbours_of_0, c.instance);
  }
});

test("schedules are computed as in Python", () => {
  for (const c of fixtures.cases) {
    assert.deepEqual(E.sizes(c.config), c.sizes, c.name);
    assert.deepEqual(E.effort(c.config), c.effort, c.name);
  }
  assert.equal(E.evaluations(E.DEFAULTS), 269800);
  assert.deepEqual(E.sizes({ ...E.DEFAULTS, generations: 4 }), [575, 450, 325, 200]);
});

test("every generation of every exported run is reproduced exactly", () => {
  assert.ok(fixtures.cases.length >= 6);
  let generations = 0;
  for (const c of fixtures.cases) {
    const run = new E.Run(instances[c.instance], c.config, c.seed, { sizes: c.sizes, effort: c.effort });
    while (!run.done) run.step();
    assert.equal(run.log.length, c.log.length, `${c.name}: generations run`);
    run.log.forEach((row, g) => assert.deepEqual(row, c.log[g], `${c.name}, generation ${g + 1}`));
    assert.deepEqual(Array.from(run.best), c.tour, c.name);
    assert.equal(run.bestLength, c.length, c.name);
    assert.deepEqual(Array.from(run.lengths.slice(0, run.m)), c.lengths, c.name);
    assert.equal(E.tourLength(instances[c.instance], run.best), c.length);
    generations += run.log.length;
  }
  assert.ok(generations > 50);
  const names = fixtures.cases.map((c) => `${c.config.crossover} ${c.config.mutation} ${c.config.schedule}`).join(" ");
  for (const word of ["pmx", "ox", "er", "swap", "inversion", "funnel", "fixed", "reverse", "geometric"]) assert.ok(names.includes(word), word);
  assert.ok(fixtures.cases.some((c) => c.config.budget > 0 && c.log.length < c.config.generations), "a run cut off by its budget");
});

test("every crossover returns a valid tour", () => {
  const n = 23;
  const rng = makeRng(11);
  const s = E.scratch(n);
  const a = new Int32Array(n);
  const b = new Int32Array(n);
  const c1 = new Int32Array(n);
  const c2 = new Int32Array(n);
  for (let trial = 0; trial < 300; trial += 1) {
    E.randomTour(a, rng);
    E.randomTour(b, rng);
    for (const x of [0, 1, 2]) {
      E.cross(x, a, b, c1, c2, rng, s);
      assert.ok(isTour(c1, n) && isTour(c2, n), `crossover ${x}`);
    }
  }
});

test("crossing a tour with itself gives that tour back", () => {
  const n = 17;
  const rng = makeRng(2);
  const s = E.scratch(n);
  const a = new Int32Array(n);
  const c1 = new Int32Array(n);
  const c2 = new Int32Array(n);
  E.randomTour(a, rng);
  const edges = (t) => new Set(Array.from(t, (u, i) => [u, t[(i + 1) % n]].sort((x, y) => x - y).join("-")));
  for (const x of [0, 1, 2]) {
    E.cross(x, a, a, c1, c2, rng, s);
    assert.deepEqual(edges(c1), edges(a), `crossover ${x}`);
  }
});

test("textbook crossover examples", () => {
  // Goldberg and Lingle's example, zero-based: cut after the 3rd and 6th city.
  const a = Int32Array.from([2, 3, 7, 1, 6, 0, 5, 4]);
  const b = Int32Array.from([3, 1, 4, 0, 5, 7, 2, 6]);
  const child = new Int32Array(8);
  E.pmx(b, a, child, 3, 6, new Int32Array(8));
  assert.deepEqual(Array.from(child), [2, 3, 1, 0, 5, 7, 6, 4]);
  E.ox(a, b, child, 3, 6, new Int32Array(8));
  assert.deepEqual(Array.from(child.slice(3, 6)), [1, 6, 0]);
  assert.ok(isTour(child, 8));
});

test("2-opt only ever shortens a tour, and reports the change exactly", () => {
  const inst = instances.qa194;
  const rng = makeRng(4);
  const tour = new Int32Array(inst.n);
  const pos = new Int32Array(inst.n);
  const out = [0, 0, 0];
  for (let trial = 0; trial < 10; trial += 1) {
    E.randomTour(tour, rng);
    const before = E.tourLength(inst, tour);
    E.twoOpt(inst, tour, pos, trial === 0 ? 5 : E.FULL_EFFORT, out);
    assert.ok(isTour(tour, inst.n));
    assert.equal(E.tourLength(inst, tour), before + out[0]);
    assert.ok(out[0] < 0);
    if (trial === 0) assert.equal(out[1], 5);
    else {
      E.twoOpt(inst, tour, pos, E.FULL_EFFORT, out);
      assert.deepEqual(out.slice(0, 2), [0, 0], "a local optimum stays put");
    }
    tour.forEach((city, i) => assert.equal(pos[city], i));
  }
});

test("a run's log is consistent with itself", () => {
  const inst = instances.dj38;
  const run = new E.Run(inst, { ...E.DEFAULTS, generations: 20, crossover: "er", mutation: "inversion" }, 3);
  let previous = Infinity;
  while (!run.done) {
    const row = run.step();
    assert.ok(row[E.COL.BEST] <= previous);
    previous = row[E.COL.BEST];
    const b = E.breeding(row, inst.n);
    assert.ok(b.selected <= b.mean && b.sd >= 0);
    assert.ok(b.kinship >= 0 && b.kinship <= 1);
    assert.ok(row[E.COL.BEST] >= inst.optimum);
    for (let r = 0; r < run.m; r += 1) assert.equal(E.tourLength(inst, run.tours[r]), run.lengths[r]);
  }
  assert.equal(run.step(), null);
  assert.equal(run.log.length, 20);
  assert.equal(run.evals, E.evaluations({ ...E.DEFAULTS, generations: 20 }));
});

test("local search gets Qatar close to the optimum in the browser engine", () => {
  const inst = instances.qa194;
  const config = { ...E.DEFAULTS, schedule: "fixed", base: 20, extra: 0, init_size: 40, generations: 40, crossover: "er", mutation: "inversion", effort_start: E.FULL_EFFORT, effort_end: E.FULL_EFFORT };
  const run = new E.Run(inst, config, 1);
  while (!run.done) run.step();
  assert.ok(run.bestLength / inst.optimum - 1 < 0.02);
  assert.ok(run.cost > run.evals);
});

test("the budget stops a run, and odd settings are refused", () => {
  const inst = instances.dj38;
  const run = new E.Run(inst, { ...E.DEFAULTS, budget: 3000 }, 1);
  while (!run.done) run.step();
  assert.ok(run.evals >= 3000 && run.evals <= 3002);
  assert.ok(run.log.length < 300);
  for (const bad of [{ crossover: "magic" }, { mutation: "none" }, { schedule: "zigzag" }, { generations: 0 }, { base: 1 }, { mutation_rate: 2 }, { init_size: 2.5 }, { budget: -1 }, { mutation_rate: NaN }]) {
    assert.throws(() => E.validate({ ...E.DEFAULTS, ...bad }), JSON.stringify(bad));
  }
  assert.throws(() => E.makeInstance([[0, 0], [1, 1]]));
});

test("tiny instances and tiny populations do not break", () => {
  const inst = E.makeInstance([[0, 0], [0, 3], [4, 3], [4, 0]]);
  for (const crossover of ["pmx", "ox", "er"]) {
    const run = new E.Run(inst, { ...E.DEFAULTS, generations: 10, base: 2, extra: 0, schedule: "fixed", init_size: 2, crossover, effort_start: 5, effort_end: 5 }, 8);
    while (!run.done) run.step();
    assert.equal(run.bestLength, 14, crossover);
  }
});
