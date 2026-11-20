// Turning the page's controls into a run's settings. Pure functions, so they
// can be tested without a browser.

import { DEFAULTS, FULL_EFFORT, evaluations, sizes } from "./engine.js";

/** Work every run on the page may spend: what the 2020 program spent. */
export const BUDGET = 269800;

/** Population sizes the slider steps through. */
export const SIZE_STEPS = [5, 10, 20, 40, 80, 200, 450];

// Rough cost of one child, in tour evaluations: two offspring are evaluated,
// and local search on the chosen one costs about this much more.
const SEARCH_COST = 16;

/**
 * choices: { crossover, mutation, shape: "fixed" | "funnel" | "reverse", size, search }
 * A funnel starts 3.5 times as large as it ends, as in 2020 (700 to 200).
 * The number of generations is set so the run about uses the budget; the
 * budget itself is enforced by the engine.
 */
export function plan(choices) {
  const base = choices.size;
  const extra = choices.shape === "fixed" ? 0 : Math.round(base * 2.5);
  const average = base + extra / 2;
  const initSize = choices.shape === "funnel" ? Math.max(2, extra) : 2 * base;
  const perChild = 2 + (choices.search ? SEARCH_COST * (choices.shape === "funnel" ? 1.4 : 1) : 0);
  const effort = choices.search ? FULL_EFFORT : 0;
  const config = {
    ...DEFAULTS,
    schedule: choices.shape,
    base,
    extra,
    init_size: initSize,
    generations: Math.max(2, Math.ceil((BUDGET - initSize) / (average * perChild))),
    crossover: choices.crossover,
    mutation: choices.mutation,
    effort_start: effort,
    effort_end: effort,
    budget: BUDGET,
  };
  // Without local search the cost is known exactly: make sure the plan reaches the budget.
  while (!choices.search && evaluations(config) < BUDGET) config.generations += Math.max(1, Math.ceil((BUDGET - evaluations(config)) / (2 * average)));
  return config;
}

/** The 2020 settings exactly (they spend the budget to the last evaluation). */
export const ORIGINAL = { crossover: "pmx", mutation: "swap", shape: "funnel", size: 200, search: false };

export const PRESETS = [
  { name: "2020 original", choices: ORIGINAL },
  { name: "Smaller, longer", choices: { crossover: "pmx", mutation: "swap", shape: "fixed", size: 20, search: false } },
  { name: "Better operators", choices: { crossover: "er", mutation: "inversion", shape: "fixed", size: 40, search: false } },
  { name: "Cross, then refine", choices: { crossover: "er", mutation: "inversion", shape: "fixed", size: 40, search: true } },
];

/** A one-line description of how the population changes over the run. */
export function describe(config) {
  const s = sizes(config);
  const first = s[0];
  const last = s[s.length - 1];
  if (first === last) return `${first} kept each generation, for up to ${s.length.toLocaleString("en")} generations`;
  return `${first} kept at first, ${last} at the end, over ${s.length.toLocaleString("en")} generations`;
}
