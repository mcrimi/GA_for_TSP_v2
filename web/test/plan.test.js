import { test } from "node:test";
import assert from "node:assert/strict";
import { DEFAULTS, evaluations, sizes, validate } from "../js/engine.js";
import { BUDGET, ORIGINAL, PRESETS, SIZE_STEPS, describe, plan } from "../js/plan.js";

test("the 2020 preset is the 2020 program", () => {
  const config = plan(ORIGINAL);
  assert.deepEqual(config, { ...DEFAULTS, budget: BUDGET });
  assert.equal(evaluations(config), BUDGET);
});

test("every combination of controls gives valid settings that reach the budget", () => {
  for (const crossover of ["pmx", "ox", "er"]) {
    for (const shape of ["fixed", "funnel", "reverse"]) {
      for (const size of SIZE_STEPS) {
        for (const search of [false, true]) {
          const config = validate(plan({ crossover, mutation: "swap", shape, size, search }));
          assert.equal(config.budget, BUDGET);
          // Without local search the cost is known exactly: it must not fall short.
          if (!search) assert.ok(evaluations(config) >= BUDGET && evaluations(config) < BUDGET * 1.02, `${shape} ${size}`);
          const s = sizes(config);
          if (shape === "funnel") assert.ok(s[0] > s[s.length - 1]);
          if (shape === "reverse") assert.ok(s[0] < s[s.length - 1]);
          if (shape === "fixed") assert.equal(s[0], s[s.length - 1]);
        }
      }
    }
  }
});

test("presets are valid and described in words", () => {
  assert.equal(PRESETS.length, 4);
  for (const p of PRESETS) validate(plan(p.choices));
  assert.equal(describe(plan(ORIGINAL)), "698 kept at first, 200 at the end, over 300 generations");
  assert.match(describe(plan(PRESETS[1].choices)), /^20 kept each generation, for up to 6,7\d\d generations$/);
});
