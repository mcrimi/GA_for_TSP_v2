// Wiring for /tsp/: the live run and the findings charts.

import { COL, Run, breeding, makeInstance } from "./engine.js";
import { BUDGET, PRESETS, SIZE_STEPS, describe, plan } from "./plan.js";
import { drawFunnel as funnelCanvas, drawMap, drawProgress } from "./view.js";
import { DASHES, drawBars, drawFunnel, drawLines, percent } from "./charts.js";

const $ = (id) => document.getElementById(id);
const fill = (key, text) => document.querySelectorAll(`[data-fill="${key}"]`).forEach((el) => { el.textContent = text; });
const num = (v) => Math.round(v).toLocaleString("en");

const SEGMENTS = [
  { key: "map", label: "Country", options: [["dj38", "Djibouti · 38"], ["qa194", "Qatar · 194"]] },
  { key: "crossover", label: "Crossover", options: [["pmx", "Mapped"], ["ox", "Order"], ["er", "Edges"]] },
  { key: "mutation", label: "Mutation", options: [["swap", "Swap two"], ["inversion", "Reverse a stretch"]] },
  { key: "shape", label: "Population", options: [["funnel", "Funnel"], ["fixed", "Fixed"], ["reverse", "Growing"]] },
  { key: "search", label: "Refine each child (2-opt)", options: [[false, "Off"], [true, "On"]] },
];

const START_PRESET = 0;
const state = {
  map: "qa194", ...PRESETS[START_PRESET].choices, preset: START_PRESET,
  seed: 1, run: null, config: null, curve: null, ghost: null, running: false, pace: 20,
};
let instances = {};
let data = null;

/* ---------- Paper settings (shared with the main site) ---------- */

function initPaper(onChange) {
  const KEY = "mcrimi-site";
  const read = () => { try { return JSON.parse(localStorage.getItem(KEY) || "{}") || {}; } catch (e) { return {}; } };
  const root = document.documentElement;
  const buttons = [...document.querySelectorAll("[data-paper]")];
  const show = (paper) => buttons.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.paper === paper)));
  show(read().paper || "warm");
  buttons.forEach((b) => b.addEventListener("click", () => {
    const paper = b.dataset.paper;
    root.dataset.theme = paper === "night" ? "dark" : "light";
    if (paper === "grey") root.dataset.paper = "grey"; else delete root.dataset.paper;
    try { localStorage.setItem(KEY, JSON.stringify({ ...read(), paper })); } catch (e) { /* private mode: keep it for this visit only */ }
    show(paper);
    onChange();
  }));
  const aa = $("aa");
  const panel = $("panel");
  const toggle = (open) => { panel.hidden = !open; aa.setAttribute("aria-expanded", String(open)); };
  aa.addEventListener("click", () => toggle(panel.hidden));
  document.addEventListener("click", (e) => { if (!panel.hidden && !panel.contains(e.target) && e.target !== aa) toggle(false); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") toggle(false); });
}

/* ---------- The live run ---------- */

const choices = () => ({ crossover: state.crossover, mutation: state.mutation, shape: state.shape, size: state.size, search: state.search });

/** Start a fresh run. keepGhost: leave the previous curve on the chart for comparison. */
function newRun({ freshSeed = false, keepGhost = true } = {}) {
  setRunning(false);
  if (freshSeed) state.seed = (Math.floor(Math.random() * 0xffffffff) >>> 0) || 1;
  const instance = instances[state.map];
  if (!instance) return;
  if (keepGhost && state.curve && state.curve.cost.length > 1 && state.curve.map === state.map) state.ghost = { ...state.curve, dashed: true };
  else if (!keepGhost || (state.curve && state.curve.map !== state.map)) state.ghost = null;
  state.config = plan(choices());
  state.run = new Run(instance, state.config, state.seed);
  const gap = state.run.bestLength / instance.optimum - 1;
  state.curve = { map: state.map, cost: [state.run.cost], gap: [gap], label: label() };
  render();
}

const label = () => (state.preset >= 0 ? PRESETS[state.preset].name : "your settings");

function advance() {
  const run = state.run;
  const before = run.bestLength;
  const row = run.step();
  if (!row) return;
  if (run.bestLength < before) {
    state.curve.cost.push(run.cost);
    state.curve.gap.push(run.bestLength / run.instance.optimum - 1);
  }
  state.curve.costNow = run.cost;
}

function render() {
  const run = state.run;
  if (!run) return;
  const instance = run.instance;
  drawMap($("map"), instance.coords, run.best);
  drawProgress($("progress"), [state.ghost, state.curve], BUDGET);
  funnelCanvas($("funnel"), run.sizes, run.generation);

  const gap = run.bestLength / instance.optimum - 1;
  const last = run.log[run.log.length - 1];
  const b = last ? breeding(last, instance.n) : null;
  $("r-gen").textContent = `${num(run.generation)} of ${num(run.sizes.length)}`;
  $("r-length").textContent = num(run.bestLength);
  $("r-gap").textContent = gap === 0 ? "optimal" : `+${percent(gap)}`;
  $("r-kin").textContent = b && Number.isFinite(b.kinship) ? percent(b.kinship) : "–";
  $("funnel-cap").textContent = `The population: ${describe(state.config)}. Filled as far as the run has got.`;

  let verdict;
  let fine = "";
  if (!run.generation) {
    verdict = `Random tours to begin with: the best is <strong>${percent(gap)}</strong> longer than the shortest possible. Press Run.`;
  } else if (gap === 0) {
    verdict = `<strong>The shortest possible tour</strong>, ${num(instance.optimum)}, found after ${num(state.curve.cost[state.curve.cost.length - 1])} evaluations’ worth of work.`;
  } else {
    verdict = `Best tour so far <strong>${num(run.bestLength)}</strong>: ${percent(gap)} longer than the shortest possible, ${num(instance.optimum)}.`;
  }
  if (run.generation && b) {
    if (b.kinship > 0.9) fine = `The survivors share ${percent(b.kinship)} of their edges with the best tour: the population is inbred, and crossing near-copies produces near-copies.`;
    else if (Number.isFinite(b.load) && b.load > 0.02 && !state.search) fine = `Offspring are on average ${percent(b.load)} longer than their parents. Selection has to win that back every generation.`;
    else if (state.search && last[COL.MADE] > 0) fine = `Each child was improved by ${(last[COL.MOVES] / last[COL.MADE]).toFixed(1)} local moves on average this generation.`;
  }
  if (run.done) fine += run.cost >= BUDGET ? ` Budget of ${num(BUDGET)} evaluations spent.` : ` Finished after ${num(run.cost)} of ${num(BUDGET)} evaluations.`;
  if (state.ghost && run.generation) fine += ` Dashed: the previous run (${state.ghost.label}).`;
  $("verdict").innerHTML = verdict;
  $("verdict-fine").textContent = fine.trim();
}

let frame = null;
let clock = 0;
let target = 0;

function loop(now) {
  if (!state.running) return;
  const run = state.run;
  // Pace the run by work done, so that every setting takes about as long to watch.
  target += ((now - clock) / 1000) * (BUDGET / state.pace);
  clock = now;
  const deadline = performance.now() + 14;
  let did = false;
  while (!run.done && run.cost < target && performance.now() < deadline) { advance(); did = true; }
  if (run.cost < target - BUDGET / 20) target = run.cost; // fell behind: do not sprint later
  if (did) render();
  if (run.done) { setRunning(false); render(); return; }
  frame = requestAnimationFrame(loop);
}

function setRunning(on) {
  state.running = Boolean(on && state.run && !state.run.done);
  $("run").textContent = state.running ? "Pause" : "Run";
  cancelAnimationFrame(frame);
  if (state.running) { clock = performance.now(); target = state.run.cost; frame = requestAnimationFrame(loop); }
}

function syncControls() {
  for (const seg of SEGMENTS) {
    $(`seg-${seg.key}`).querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.value === String(state[seg.key]))));
  }
  $("in-size").value = SIZE_STEPS.indexOf(state.size);
  $("out-size").textContent = state.shape === "fixed" ? state.size : state.shape === "funnel" ? `${state.size + Math.round(state.size * 2.5)} → ${state.size}` : `${state.size} → ${state.size + Math.round(state.size * 2.5)}`;
  document.querySelectorAll("#presets button").forEach((b, i) => b.setAttribute("aria-pressed", String(i === state.preset)));
}

function buildControls() {
  $("presets").innerHTML = PRESETS.map((p, i) => `<button type="button" data-preset="${i}">${p.name}</button>`).join("");
  $("segments").innerHTML = SEGMENTS.map((seg) => `<div class="field${seg.options.length > 2 || seg.key === "mutation" ? " wide" : ""}">
      <span class="label">${seg.label}</span>
      <div class="seg" id="seg-${seg.key}" role="group" aria-label="${seg.label}">${seg.options.map(([value, text]) => `<button type="button" data-value="${value}">${text}</button>`).join("")}</div>
    </div>`).join("");
  $("in-size").max = SIZE_STEPS.length - 1;

  $("presets").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-preset]");
    if (!b) return;
    state.preset = Number(b.dataset.preset);
    Object.assign(state, PRESETS[state.preset].choices);
    syncControls();
    newRun();
  });
  $("segments").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-value]");
    if (!b) return;
    const key = b.parentNode.id.slice(4);
    const value = key === "search" ? b.dataset.value === "true" : b.dataset.value;
    if (state[key] === value) return;
    state[key] = value;
    if (key !== "map") state.preset = PRESETS.findIndex((p) => Object.entries(p.choices).every(([k, v]) => state[k] === v));
    syncControls();
    newRun();
  });
  $("in-size").addEventListener("input", (e) => {
    const size = SIZE_STEPS[Math.min(Math.max(Number(e.target.value) || 0, 0), SIZE_STEPS.length - 1)];
    if (size === state.size) return;
    state.size = size;
    state.preset = PRESETS.findIndex((p) => Object.entries(p.choices).every(([k, v]) => state[k] === v));
    syncControls();
    newRun();
  });
  $("run").addEventListener("click", () => {
    if (state.run && state.run.done) newRun({ freshSeed: true });
    setRunning(!state.running);
  });
  $("step").addEventListener("click", () => { setRunning(false); if (state.run && !state.run.done) { advance(); render(); } });
  $("reset").addEventListener("click", () => newRun({ freshSeed: true }));
  $("speed").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-pace]");
    if (!b) return;
    state.pace = Number(b.dataset.pace);
    $("speed").querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
  });
  $("controls").addEventListener("submit", (e) => e.preventDefault());
}

/* ---------- Findings ---------- */

const armByName = (rows, name) => rows.find((r) => r.arm === name);
const sizeOf = (arm) => Number(arm.arm.replace("Fixed ", ""));
const isFixed = (arm) => /^Fixed \d+$/.test(arm.arm);

function scheduleChart(id, block, yRange, yTicks, places) {
  const fixed = block.arms.filter(isFixed);
  const spans = block.arms.filter((a) => !isFixed(a)).map((a) => {
    const label = a.arm.startsWith("Funnel") ? "2020 funnel" : a.arm.startsWith("Reverse") ? "growing" : "steep funnel";
    return { x0: a.sizes[0], x1: a.sizes[1], y: a.gap, label, place: places[label] };
  });
  drawLines($(id), {
    x: [8, 1700], y: yRange, xLog: true, yTicks, labelWidth: 16, ratio: 0.7,
    xLabel: "Tours kept each generation (log scale)",
    xFormat: (v) => v, yFormat: (v) => `${Math.round(v * 100)}%`, xTicks: [10, 20, 50, 100, 200, 450, 1000],
    series: [{ x: fixed.map(sizeOf), y: fixed.map((a) => a.gap), lo: fixed.map((a) => a.lo), hi: fixed.map((a) => a.hi) }],
    points: fixed.map((a) => ({ x: sizeOf(a), y: a.gap })),
    spans,
  });
}

function drawFindings() {
  if (!data) return;
  const sched = data.schedules;
  drawFunnel($("funnel-chart"), { shapes: [
    { label: "2020 funnel", note: "698 kept, down to 200", sizes: lin(698, 200) },
    { label: "Fixed", note: "450 throughout", sizes: lin(450, 450) },
    { label: "Growing", note: "200 kept, up to 698", sizes: lin(200, 698) },
  ] });
  scheduleChart("sched-qa", sched.qa194, [1, 3], [1, 1.5, 2, 2.5, 3], { "steep funnel": "above", "2020 funnel": "left", growing: "below" });
  scheduleChart("sched-dj", sched.dj38, [0, 0.6], [0, 0.2, 0.4, 0.6], { "steep funnel": "below-start", "2020 funnel": "below", growing: "above" });

  const ops = data.operators.qa194;
  const mem = data.memetic.find((m) => m.instance === "qa194" && m.budget === data.budget);
  const bestFixed = armByName(sched.qa194.arms, sched.qa194.verdict.best_fixed);
  const bestMem = armByName(mem.arms, mem.verdict.best_fixed);
  const rows = [
    { label: "2020, as written", a: armByName(sched.qa194.arms, "Funnel 700 to 200 (2020)") },
    { label: `Right-sized population (${sizeOf(bestFixed)})`, a: bestFixed },
    { label: "Better mutation", a: armByName(ops, "pmx+inversion") },
    { label: "Better crossover", a: armByName(ops, "er+swap") },
    { label: "Both", a: armByName(ops, "er+inversion") },
    { label: "No breeding, only refining", a: mem.restarts, hatched: true },
    { label: "Cross, then refine", a: bestMem },
  ].map((row) => ({ label: row.label, value: row.a.gap, lo: row.a.lo, hi: row.a.hi, hatched: row.hatched, note: percent(row.a.gap) }));
  drawBars($("ladder-chart"), { rows, x: [0.0005, 4], xLog: true, xFormat: percent, xTicks: [0.001, 0.01, 0.1, 1], xLabel: "Longer than the shortest possible tour (log scale). Bars: mean of 40 runs; line: 95% interval." });

  const traces = data.traces.arms.slice(0, 3);
  drawLines($("kin-chart"), {
    x: [0, 300], y: [0, 1], labelWidth: 132, ratio: 0.56, xLabel: "Generation", yFormat: (v) => `${Math.round(v * 100)}%`,
    series: traces.map((t, i) => ({ label: ["Partially mapped", "Order", "Edge recombination"][i], x: t.generation, y: t.kinship, dash: DASHES[i] })),
  });
  const short = { pmx: "Partially mapped", ox: "Order", er: "Edge recombination" };
  const stages = [["After 30 generations", "early"], ["After 100 generations", "later"]];
  const loads = stages.flatMap(([stage, when]) => data.heritability.filter((h) => h.stage === stage)
    .map((h) => ({ label: `${short[h.crossover]}, ${when}`, value: h.load, note: `+${percent(h.load)}`, hatched: when === "early" })));
  drawBars($("load-chart"), {
    rows: loads, x: [0, Math.max(0.5, ...loads.map((h) => h.value)) * 1.1], xFormat: (v) => `${Math.round(v * 100)}%`, labelWidth: 172,
    xLabel: "How much longer children are than their parents",
  });

  const budgets = data.memetic.filter((m) => m.instance === "qa194");
  const pick = (name) => budgets.map((m) => armByName(m.arms, name).gap);
  const bx = budgets.map((m) => m.budget);
  drawLines($("budget-chart"), {
    x: [bx[0], bx[bx.length - 1]], y: [0.0003, 0.1], xLog: true, yLog: true, labelWidth: 150, ratio: 0.56,
    xLabel: "Work allowed, in tour evaluations", xFormat: (v) => `${Math.round(v / 1000)}k`, yFormat: percent, xTicks: bx, yTicks: [0.001, 0.01, 0.1],
    series: [
      { label: "No breeding", x: bx, y: budgets.map((m) => m.restarts.gap), dash: DASHES[2] },
      { label: "Fixed 10", x: bx, y: pick("Fixed 10"), dash: DASHES[1] },
      { label: "Fixed 40", x: bx, y: pick("Fixed 40"), dash: DASHES[3] },
      { label: "Funnel 80 to 10", x: bx, y: pick("Funnel 80 to 10") },
    ],
    points: budgets.flatMap((m) => [{ x: m.budget, y: armByName(m.arms, "Funnel 80 to 10").gap }]),
  });
}

const lin = (a, b) => Array.from({ length: 60 }, (_, i) => a + ((b - a) * i) / 59);

function memeticTable(block) {
  const names = ["Fixed 5", "Fixed 10", "Fixed 20", "Fixed 40", "Fixed 80", "Funnel 80 to 10", "Steep funnel 200 to 5", "Steep funnel, effort 1 to full"];
  const labels = { "Steep funnel, effort 1 to full": "Steep funnel, cheap screening first" };
  const best = Math.min(...names.map((n) => armByName(block.arms, n).gap));
  const row = (a, name) => `<tr${a.gap === best ? ' class="picked"' : ""}><th scope="row">${name}</th><td class="n">${percent(a.gap)}</td><td class="n">${percent(a.lo)} to ${percent(a.hi)}</td><td class="n">${Math.round(a.solved * a.n)} of ${a.n}</td></tr>`;
  return names.map((n) => row(armByName(block.arms, n), labels[n] || n)).join("") + row(block.restarts, "No breeding: refine random tours");
}

function fillFindings() {
  const qa = data.schedules.qa194;
  const dj = data.schedules.dj38;
  const er = data.schedules_er.qa194;
  const funnel = armByName(qa.arms, "Funnel 700 to 200 (2020)");
  const signed = (v) => `${v < 0 ? "−" : "+"}${Math.abs(100 * v).toFixed(0)}`;
  const interval = (c) => `${signed(c.difference)} points, 95% interval ${signed(c.lo)} to ${signed(c.hi)}`;
  fill("seeds", data.seeds);
  fill("budget", num(data.budget));
  fill("gap-2020", percent(funnel.gap));
  fill("gap-2020-dj", percent(armByName(dj.arms, "Funnel 700 to 200 (2020)").gap));
  fill("same-average-qa", interval(qa.verdict.funnel_vs_same_average));
  fill("same-average-dj", interval(dj.verdict.funnel_vs_same_average));
  fill("same-average-er", interval(er.verdict.funnel_vs_same_average));
  fill("best-fixed-qa", qa.verdict.best_fixed.toLowerCase());
  fill("best-fixed-qa-gap", percent(armByName(qa.arms, qa.verdict.best_fixed).gap));
  fill("best-fixed-dj", dj.verdict.best_fixed.toLowerCase());
  fill("best-fixed-dj-gap", percent(armByName(dj.arms, dj.verdict.best_fixed).gap));
  fill("best-fixed-er", er.verdict.best_fixed.toLowerCase());
  fill("best-fixed-er-gap", percent(armByName(er.arms, er.verdict.best_fixed).gap));
  fill("funnel-er-gap", percent(armByName(er.arms, "Funnel 700 to 200 (2020)").gap));

  const ops = data.operators.qa194;
  fill("op-pmx-swap", percent(armByName(ops, "pmx+swap").gap));
  fill("op-pmx-inv", percent(armByName(ops, "pmx+inversion").gap));
  fill("op-er-swap", percent(armByName(ops, "er+swap").gap));
  fill("op-er-inv", percent(armByName(ops, "er+inversion").gap));

  const h = (stage, x) => data.heritability.find((r) => r.stage === stage && r.crossover === x);
  for (const x of ["pmx", "ox", "er"]) {
    fill(`load-${x}`, percent(h("After 100 generations", x).load));
    fill(`odds-${x}`, num(1 / h("After 30 generations", x).better_than_parents));
    fill(`inherit-${x}`, percent(h("After 100 generations", x).inherited));
  }
  const traces = data.traces.arms;
  fill("kin-pmx", percent(traces[0].kinship[traces[0].kinship.length - 1]));
  fill("kin-er", percent(traces[2].kinship[traces[2].kinship.length - 1]));

  const mem = data.memetic.find((m) => m.instance === "qa194" && m.budget === data.budget);
  const uy = data.memetic.find((m) => m.instance === "uy734");
  const bestMem = armByName(mem.arms, mem.verdict.best_fixed);
  fill("mem-best", percent(bestMem.gap));
  fill("mem-best-name", mem.verdict.best_fixed.toLowerCase());
  fill("mem-solved", `${Math.round(bestMem.solved * bestMem.n)} of ${bestMem.n}`);
  fill("mem-restarts", percent(mem.restarts.gap));
  fill("mem-funnel", percent(armByName(mem.arms, "Funnel 80 to 10").gap));
  fill("mem-funnel-solved", `${Math.round(armByName(mem.arms, "Funnel 80 to 10").solved * bestMem.n)} of ${bestMem.n}`);
  fill("mem-staged", percent(armByName(mem.arms, "Steep funnel, effort 1 to full").gap));
  fill("uy-best", percent(armByName(uy.arms, uy.verdict.best_fixed).gap));
  fill("uy-best-name", uy.verdict.best_fixed.toLowerCase());
  fill("uy-funnel", percent(armByName(uy.arms, "Funnel 80 to 10").gap));
  fill("uy-restarts", percent(uy.restarts.gap));
  $("mem-qa").querySelector("tbody").innerHTML = memeticTable(mem);
  $("mem-uy").querySelector("tbody").innerHTML = memeticTable(uy);
}

/* ---------- Start ---------- */

async function getJson(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
  return response.json();
}

async function start() {
  buildControls();
  syncControls();
  initPaper(() => { render(); });

  let width = innerWidth;
  let timer;
  addEventListener("resize", () => {
    if (innerWidth === width) return;
    width = innerWidth;
    clearTimeout(timer);
    timer = setTimeout(() => { render(); drawFindings(); }, 120);
  });
  document.addEventListener("visibilitychange", () => { if (document.hidden) setRunning(false); });

  try {
    const raw = await getJson("data/instances.json");
    instances = Object.fromEntries(Object.entries(raw).map(([k, v]) => [k, makeInstance(v.coords, v.optimum)]));
    newRun({ keepGhost: false });
  } catch (error) {
    console.error(error);
    $("verdict").textContent = "The maps could not be loaded. Reload the page to try again.";
  }
  try {
    data = await getJson("data/experiments.json");
    fillFindings();
    drawFindings();
  } catch (error) {
    console.error(error);
    document.querySelectorAll("#findings .chart").forEach((el) => { el.hidden = true; });
  }
}

start();
